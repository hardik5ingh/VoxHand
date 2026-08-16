import base64
from datetime import datetime
import difflib
import os
import sys
import threading
import time

import cv2
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from wordfreq import top_n_list

# Load environment variables
load_dotenv()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

# ==========================================
# CONFIGURATION & DICTIONARY
# ==========================================
CONFIG = {
    "MODEL_PATH": os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task"),
    "GESTURE_MODEL_PATH": os.path.join(os.path.dirname(__file__), "models", "gesture_model.pkl"),
    "CONFIRMATION_FRAMES": 2,      # Tuned for fast, responsive typing
    "CONFIDENCE_THRESHOLD": 0.35,  # Low-latency probability threshold
}

DICTIONARY = [w.upper() for w in top_n_list('en', 25000) if w.isalpha()]

def get_word_suggestions(prefix, top_k=4):
    if not prefix:
        return []
    prefix = prefix.upper()
    exact_matches = [w for w in DICTIONARY if w.startswith(prefix) and w != prefix]
    if len(exact_matches) >= top_k:
        return exact_matches[:top_k]
    close_matches = difflib.get_close_matches(prefix, DICTIONARY, n=top_k, cutoff=0.6)
    combined = list(dict.fromkeys(exact_matches + close_matches))
    return combined[:top_k]

# ==========================================
# FLASK & DATABASE INITIALIZATION
# ==========================================
app = Flask(__name__)
CORS(app)

# Database Connection (Uses Render PostgreSQL or local SQLite fallback)
database_url = os.getenv("DATABASE_URL", "sqlite:///users.db")
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ==========================================
# DATABASE MODEL
# ==========================================
class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    google_id = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(150), nullable=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    picture = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# ==========================================
# MEDIAPIPE & MODEL LOADING
# ==========================================
if not os.path.exists(CONFIG["GESTURE_MODEL_PATH"]) or not os.path.exists(CONFIG["MODEL_PATH"]):
    raise FileNotFoundError("Error: Missing model files in models/ folder.")

clf = joblib.load(CONFIG["GESTURE_MODEL_PATH"])
print("✅ Gesture model loaded successfully.")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=CONFIG["MODEL_PATH"]),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1
)
landmarker = HandLandmarker.create_from_options(options)

letters_set = {chr(i) for i in range(97, 123)}
numbers_set = {str(i) for i in range(11)}
symbols_set = {'space', 'del', '.'}

letters_symbols_set = letters_set.union(symbols_set)
numbers_symbols_set = numbers_set.union(symbols_set)

MODES = {
    '0': ('ALL GESTURES', set(clf.classes_)),
    '1': ('LETTERS & SYMBOLS', letters_symbols_set),
    '2': ('NUMBERS & SYMBOLS', numbers_symbols_set)
}

# ==========================================
# STATE & BUFFERS
# ==========================================
state_lock = threading.Lock()

last_appended_gesture = None
current_candidate = None
candidate_counter = 0

current_state = {
    "detected_gesture": "None",
    "confidence": 0.0,
    "current_word": "",
    "sentence": "",
    "suggestions": [],
    "mode": "1",
    "hand_detected": False
}

# ==========================================
# 74-FEATURE EXTRACTION PIPELINE
# ==========================================
def extract_features(landmarks, img_w, img_h):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    coords[:, 0] *= (img_w / img_h)
    
    wrist = coords[0]
    coords = coords - wrist
    
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords = coords / max_val
        
    raw_features = coords.flatten()
    joints = coords.reshape(1, 21, 3)
    
    distances = []
    fingertip_indices = [4, 8, 12, 16, 20]
    
    for tip in fingertip_indices:
        distances.append(np.linalg.norm(joints[:, tip, :] - joints[:, 0, :], axis=1))
        
    for i in range(len(fingertip_indices) - 1):
        distances.append(np.linalg.norm(joints[:, fingertip_indices[i], :] - joints[:, fingertip_indices[i+1], :], axis=1))
        
    distances.append(np.linalg.norm(joints[:, 4, :] - joints[:, 8, :], axis=1))
    distances.append(np.linalg.norm(joints[:, 4, :] - joints[:, 12, :], axis=1))

    extra_features = np.column_stack(distances).flatten()
    return np.hstack((raw_features, extra_features)).reshape(1, -1)

# ==========================================
# AUTH & USER ROUTES
# ==========================================
@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        "google_client_id": GOOGLE_CLIENT_ID
    })

@app.route('/api/auth/google', methods=['POST'])
def google_auth():
    data = request.get_json() or {}
    token = data.get('token')
    if not token:
        return jsonify({'error': 'No token provided'}), 400

    try:
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
        
        google_id = idinfo.get("sub")
        email = idinfo.get("email")
        name = idinfo.get("name", "User")
        picture = idinfo.get("picture", "")

        # Persist user in PostgreSQL
        user = User.query.filter_by(google_id=google_id).first()
        if user:
            user.name = name
            user.picture = picture
            user.last_login = datetime.utcnow()
        else:
            user = User(
                google_id=google_id,
                email=email,
                name=name,
                picture=picture,
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow()
            )
            db.session.add(user)

        db.session.commit()

        return jsonify({
            'status': 'success',
            'user': {
                'id': user.id,
                'google_id': user.google_id,
                'email': user.email,
                'name': user.name,
                'picture': user.picture
            }
        }), 200

    except ValueError as e:
        return jsonify({'error': 'Invalid Google token', 'details': str(e)}), 401
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Database error', 'details': str(e)}), 500

@app.route('/api/users', methods=['GET'])
def list_users():
    """Endpoint to view all registered users and their logins."""
    users = User.query.order_by(User.last_login.desc()).all()
    return jsonify([{
        "id": u.id,
        "name": u.name,
        "email": u.email,
        "picture": u.picture,
        "created_at": u.created_at.strftime("%Y-%m-%d %H:%M:%S") if u.created_at else None,
        "last_login": u.last_login.strftime("%Y-%m-%d %H:%M:%S") if u.last_login else None
    } for u in users])

# ==========================================
# PREDICTION & VISION ROUTES
# ==========================================
@app.route('/predict_landmarks', methods=['POST'])
def predict_landmarks():
    """High-speed coordinate prediction route."""
    global last_appended_gesture, current_candidate, candidate_counter
    
    data = request.get_json() or {}
    landmarks_data = data.get('landmarks')
    img_w = data.get('width', 640)
    img_h = data.get('height', 480)

    if not landmarks_data:
        with state_lock:
            last_appended_gesture = None
            current_candidate = None
            candidate_counter = 0
            current_state["detected_gesture"] = "None"
            current_state["confidence"] = 0.0
            current_state["hand_detected"] = False
        return jsonify(current_state)

    class DummyLandmark:
        def __init__(self, pt):
            self.x = pt['x']
            self.y = pt['y']
            self.z = pt['z']

    landmarks = [DummyLandmark(pt) for pt in landmarks_data]
    features = extract_features(landmarks, img_w, img_h)
    
    detected_pred = "None"
    max_prob = 0.0
    action_triggered = "none"

    with state_lock:
        mode_name, allowed_classes = MODES.get(current_state["mode"], MODES['1'])
        probabilities = clf.predict_proba(features)[0]
        valid_indices = [i for i, c in enumerate(clf.classes_) if c in allowed_classes]

        if valid_indices:
            filtered_probs = probabilities[valid_indices]
            filtered_classes = clf.classes_[valid_indices]

            max_prob = float(np.max(filtered_probs))
            detected_pred = str(filtered_classes[np.argmax(filtered_probs)]).lower()

            if max_prob >= CONFIG["CONFIDENCE_THRESHOLD"]:
                if detected_pred == current_candidate:
                    candidate_counter += 1
                else:
                    current_candidate = detected_pred
                    candidate_counter = 1

                if candidate_counter >= CONFIG["CONFIRMATION_FRAMES"]:
                    if detected_pred != last_appended_gesture:
                        last_appended_gesture = detected_pred

                        if detected_pred == 'del':
                            action_triggered = "delete"
                            if len(current_state["current_word"]) > 0:
                                if current_state["current_word"].endswith('10'):
                                    current_state["current_word"] = current_state["current_word"][:-2]
                                else:
                                    current_state["current_word"] = current_state["current_word"][:-1]
                            elif len(current_state["sentence"]) > 0:
                                current_state["sentence"] = current_state["sentence"][:-1]

                        elif detected_pred == 'space':
                            action_triggered = "confirm"
                            if current_state["current_word"]:
                                current_state["sentence"] += current_state["current_word"] + " "
                                current_state["current_word"] = ""

                        elif detected_pred == '.':
                            action_triggered = "speak"
                            if current_state["current_word"]:
                                current_state["sentence"] += current_state["current_word"] + ". "
                                current_state["current_word"] = ""
                            elif current_state["sentence"] and not current_state["sentence"].endswith('. '):
                                current_state["sentence"] = current_state["sentence"].strip() + ". "

                        elif detected_pred in numbers_set:
                            action_triggered = "confirm"
                            current_state["current_word"] += detected_pred.upper()

                        else:
                            action_triggered = "confirm"
                            current_state["current_word"] += detected_pred.upper()

                        current_state["suggestions"] = get_word_suggestions(current_state["current_word"])
            else:
                candidate_counter = 0

        current_state["detected_gesture"] = detected_pred.upper()
        current_state["confidence"] = round(max_prob * 100, 1)
        current_state["hand_detected"] = True

        return jsonify({
            "gesture": current_state["detected_gesture"],
            "confidence": current_state["confidence"],
            "current_word": current_state["current_word"],
            "sentence": current_state["sentence"],
            "suggestions": current_state["suggestions"],
            "mode_key": current_state["mode"],
            "hand_detected": True,
            "action": action_triggered
        })

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Fallback Base64 frame processing route."""
    global last_appended_gesture, current_candidate, candidate_counter

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    try:
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect(mp_image)

        detected_pred = "None"
        max_prob = 0.0
        hand_detected = bool(result.hand_landmarks)
        action_triggered = "none"

        with state_lock:
            mode_name, allowed_classes = MODES.get(current_state["mode"], MODES['1'])

            if hand_detected:
                landmarks = result.hand_landmarks[0]
                features = extract_features(landmarks, w, h)
                probabilities = clf.predict_proba(features)[0]

                valid_indices = [i for i, c in enumerate(clf.classes_) if c in allowed_classes]

                if valid_indices:
                    filtered_probs = probabilities[valid_indices]
                    filtered_classes = clf.classes_[valid_indices]

                    max_prob = float(np.max(filtered_probs))
                    detected_pred = str(filtered_classes[np.argmax(filtered_probs)]).lower()

                    if max_prob >= CONFIG["CONFIDENCE_THRESHOLD"]:
                        if detected_pred == current_candidate:
                            candidate_counter += 1
                        else:
                            current_candidate = detected_pred
                            candidate_counter = 1

                        if candidate_counter >= CONFIG["CONFIRMATION_FRAMES"]:
                            if detected_pred != last_appended_gesture:
                                last_appended_gesture = detected_pred

                                if detected_pred == 'del':
                                    action_triggered = "delete"
                                    if len(current_state["current_word"]) > 0:
                                        if current_state["current_word"].endswith('10'):
                                            current_state["current_word"] = current_state["current_word"][:-2]
                                        else:
                                            current_state["current_word"] = current_state["current_word"][:-1]
                                    elif len(current_state["sentence"]) > 0:
                                        current_state["sentence"] = current_state["sentence"][:-1]

                                elif detected_pred == 'space':
                                    action_triggered = "confirm"
                                    if current_state["current_word"]:
                                        current_state["sentence"] += current_state["current_word"] + " "
                                        current_state["current_word"] = ""

                                elif detected_pred == '.':
                                    action_triggered = "speak"
                                    if current_state["current_word"]:
                                        current_state["sentence"] += current_state["current_word"] + ". "
                                        current_state["current_word"] = ""
                                    elif current_state["sentence"] and not current_state["sentence"].endswith('. '):
                                        current_state["sentence"] = current_state["sentence"].strip() + ". "

                                elif detected_pred in numbers_set:
                                    action_triggered = "confirm"
                                    current_state["current_word"] += detected_pred.upper()

                                else:
                                    action_triggered = "confirm"
                                    current_state["current_word"] += detected_pred.upper()

                                current_state["suggestions"] = get_word_suggestions(current_state["current_word"])
                    else:
                        candidate_counter = 0
            else:
                last_appended_gesture = None
                current_candidate = None
                candidate_counter = 0

            current_state["detected_gesture"] = detected_pred.upper()
            current_state["confidence"] = round(max_prob * 100, 1)
            current_state["hand_detected"] = hand_detected

            return jsonify({
                "gesture": current_state["detected_gesture"],
                "confidence": current_state["confidence"],
                "current_word": current_state["current_word"],
                "sentence": current_state["sentence"],
                "suggestions": current_state["suggestions"],
                "mode_key": current_state["mode"],
                "hand_detected": current_state["hand_detected"],
                "action": action_triggered
            })

    except Exception as e:
        print(f"❌ Error in process_frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_state', methods=['GET'])
def get_state():
    with state_lock:
        return jsonify(current_state)

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global current_candidate, candidate_counter, last_appended_gesture
    data = request.get_json() or {}
    new_mode = str(data.get('mode', '1'))
    with state_lock:
        current_state["mode"] = new_mode
        current_candidate = None
        candidate_counter = 0
        last_appended_gesture = None
    return jsonify({"status": "success", "mode": new_mode})

@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    data = request.get_json() or {}
    selected_word = data.get('word', '')
    if selected_word:
        with state_lock:
            current_state["sentence"] += selected_word.upper() + " "
            current_state["current_word"] = ""
            current_state["suggestions"] = []
    return jsonify({"status": "success", "sentence": current_state["sentence"]})

@app.route('/clear', methods=['POST'])
def clear():
    global last_appended_gesture, current_candidate, candidate_counter
    with state_lock:
        current_state["sentence"] = ""
        current_state["current_word"] = ""
        current_state["suggestions"] = []
        last_appended_gesture = None
        current_candidate = None
        candidate_counter = 0
    return jsonify({"status": "success"})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    print(f"🚀 VoxHand Flask API running at http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)