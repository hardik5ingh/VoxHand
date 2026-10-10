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
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from wordfreq import top_n_list

load_dotenv()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

# ==========================================
# CONFIGURATION & DICTIONARY
# ==========================================
CONFIG = {
    "MODEL_PATH": os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task"),
    "GESTURE_MODEL_PATH": os.path.join(os.path.dirname(__file__), "models", "gesture_model.pkl"),
    "CONFIRMATION_FRAMES": 2,      # Instant confirmation for 120ms network packets
    "CONFIDENCE_THRESHOLD": 0.30,  # Forgiving probability threshold for mobile camera angles
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
# FLASK & CORS INITIALIZATION
# ==========================================
app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin"],
    methods=["GET", "POST", "OPTIONS"]
)

active_users = {}

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
# PER-SESSION STATE STORAGE
# ==========================================
state_lock = threading.Lock()
sessions = {}

def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            "detected_gesture": "None",
            "confidence": 0.0,
            "current_word": "",
            "sentence": "",
            "suggestions": [],
            "mode": "1",
            "hand_detected": False,
            "last_appended_gesture": None,
            "current_candidate": None,
            "candidate_counter": 0
        }
    return sessions[session_id]

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

        user_data = {
            'id': google_id,
            'google_id': google_id,
            'email': email,
            'name': name,
            'picture': picture,
            'last_login': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        }
        active_users[google_id] = user_data

        return jsonify({
            'status': 'success',
            'user': user_data
        }), 200

    except ValueError as e:
        return jsonify({'error': 'Invalid Google token', 'details': str(e)}), 401
    except Exception as e:
        return jsonify({'error': 'Authentication error', 'details': str(e)}), 500

@app.route('/api/users', methods=['GET'])
def list_users():
    return jsonify(list(active_users.values()))

# ==========================================
# PREDICTION & VISION ROUTES
# ==========================================
@app.route('/predict_landmarks', methods=['POST'])
def predict_landmarks():
    data = request.get_json() or {}
    session_id = data.get('session_id', 'default_session')
    landmarks_data = data.get('landmarks')
    img_w = data.get('width', 640)
    img_h = data.get('height', 480)

    with state_lock:
        state = get_session(session_id)
        if not landmarks_data:
            state["last_appended_gesture"] = None
            state["current_candidate"] = None
            state["candidate_counter"] = 0
            state["detected_gesture"] = "None"
            state["confidence"] = 0.0
            state["hand_detected"] = False
            return jsonify(state)

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
        state = get_session(session_id)
        mode_name, allowed_classes = MODES.get(state["mode"], MODES['1'])
        probabilities = clf.predict_proba(features)[0]
        valid_indices = [i for i, c in enumerate(clf.classes_) if c in allowed_classes]

        if valid_indices:
            filtered_probs = probabilities[valid_indices]
            filtered_classes = clf.classes_[valid_indices]

            max_prob = float(np.max(filtered_probs))
            detected_pred = str(filtered_classes[np.argmax(filtered_probs)]).lower()

            if max_prob >= CONFIG["CONFIDENCE_THRESHOLD"]:
                if detected_pred == state["current_candidate"]:
                    state["candidate_counter"] += 1
                else:
                    state["current_candidate"] = detected_pred
                    state["candidate_counter"] = 1

                if state["candidate_counter"] >= CONFIG["CONFIRMATION_FRAMES"]:
                    if detected_pred != state["last_appended_gesture"]:
                        state["last_appended_gesture"] = detected_pred

                        if detected_pred == 'del':
                            action_triggered = "delete"
                            if len(state["current_word"]) > 0:
                                if state["current_word"].endswith('10'):
                                    state["current_word"] = state["current_word"][:-2]
                                else:
                                    state["current_word"] = state["current_word"][:-1]
                            elif len(state["sentence"]) > 0:
                                state["sentence"] = state["sentence"][:-1]

                        elif detected_pred == 'space':
                            action_triggered = "confirm"
                            if state["current_word"]:
                                state["sentence"] += state["current_word"] + " "
                                state["current_word"] = ""

                        elif detected_pred == '.':
                            action_triggered = "speak"
                            if state["current_word"]:
                                state["sentence"] += state["current_word"] + ". "
                                state["current_word"] = ""
                            elif state["sentence"] and not state["sentence"].endswith('. '):
                                state["sentence"] = state["sentence"].strip() + ". "

                        elif detected_pred in numbers_set:
                            action_triggered = "confirm"
                            state["current_word"] += detected_pred.upper()

                        else:
                            action_triggered = "confirm"
                            state["current_word"] += detected_pred.upper()

                        state["suggestions"] = get_word_suggestions(state["current_word"])
            else:
                state["candidate_counter"] = 0
                state["last_appended_gesture"] = None

        state["detected_gesture"] = detected_pred.upper()
        state["confidence"] = round(max_prob * 100, 1)
        state["hand_detected"] = True

        return jsonify({
            "gesture": state["detected_gesture"],
            "confidence": state["confidence"],
            "current_word": state["current_word"],
            "sentence": state["sentence"],
            "suggestions": state["suggestions"],
            "mode_key": state["mode"],
            "hand_detected": True,
            "action": action_triggered
        })

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    session_id = data.get('session_id', 'default_session')

    try:
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect(mp_image)

        detected_pred = "None"
        max_prob = 0.0
        hand_detected = bool(result.hand_landmarks)
        action_triggered = "none"

        with state_lock:
            state = get_session(session_id)
            mode_name, allowed_classes = MODES.get(state["mode"], MODES['1'])

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
                        if detected_pred == state["current_candidate"]:
                            state["candidate_counter"] += 1
                        else:
                            state["current_candidate"] = detected_pred
                            state["candidate_counter"] = 1

                        if state["candidate_counter"] >= CONFIG["CONFIRMATION_FRAMES"]:
                            if detected_pred != state["last_appended_gesture"]:
                                state["last_appended_gesture"] = detected_pred

                                if detected_pred == 'del':
                                    action_triggered = "delete"
                                    if len(state["current_word"]) > 0:
                                        if state["current_word"].endswith('10'):
                                            state["current_word"] = state["current_word"][:-2]
                                        else:
                                            state["current_word"] = state["current_word"][:-1]
                                    elif len(state["sentence"]) > 0:
                                        state["sentence"] = state["sentence"][:-1]

                                elif detected_pred == 'space':
                                    action_triggered = "confirm"
                                    if state["current_word"]:
                                        state["sentence"] += state["current_word"] + " "
                                        state["current_word"] = ""

                                elif detected_pred == '.':
                                    action_triggered = "speak"
                                    if state["current_word"]:
                                        state["sentence"] += state["current_word"] + ". "
                                        state["current_word"] = ""
                                    elif state["sentence"] and not state["sentence"].endswith('. '):
                                        state["sentence"] = state["sentence"].strip() + ". "

                                elif detected_pred in numbers_set:
                                    action_triggered = "confirm"
                                    state["current_word"] += detected_pred.upper()

                                else:
                                    action_triggered = "confirm"
                                    state["current_word"] += detected_pred.upper()

                                state["suggestions"] = get_word_suggestions(state["current_word"])
                    else:
                        state["candidate_counter"] = 0
                        state["last_appended_gesture"] = None
            else:
                state["last_appended_gesture"] = None
                state["current_candidate"] = None
                state["candidate_counter"] = 0

            state["detected_gesture"] = detected_pred.upper()
            state["confidence"] = round(max_prob * 100, 1)
            state["hand_detected"] = hand_detected

            return jsonify({
                "gesture": state["detected_gesture"],
                "confidence": state["confidence"],
                "current_word": state["current_word"],
                "sentence": state["sentence"],
                "suggestions": state["suggestions"],
                "mode_key": state["mode"],
                "hand_detected": state["hand_detected"],
                "action": action_triggered
            })

    except Exception as e:
        print(f"❌ Error in process_frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_state', methods=['GET'])
def get_state():
    session_id = request.args.get('session_id', 'default_session')
    with state_lock:
        return jsonify(get_session(session_id))

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.get_json() or {}
    session_id = data.get('session_id', 'default_session')
    new_mode = str(data.get('mode', '1'))
    with state_lock:
        state = get_session(session_id)
        state["mode"] = new_mode
        state["current_candidate"] = None
        state["candidate_counter"] = 0
        state["last_appended_gesture"] = None
    return jsonify({"status": "success", "mode": new_mode})

@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    data = request.get_json() or {}
    session_id = data.get('session_id', 'default_session')
    selected_word = data.get('word', '')
    with state_lock:
        state = get_session(session_id)
        if selected_word:
            state["sentence"] += selected_word.upper() + " "
            state["current_word"] = ""
            state["suggestions"] = []
    return jsonify({"status": "success", "sentence": state["sentence"]})

@app.route('/clear', methods=['POST'])
def clear():
    data = request.get_json() or {}
    session_id = data.get('session_id', 'default_session')
    with state_lock:
        state = get_session(session_id)
        state["sentence"] = ""
        state["current_word"] = ""
        state["suggestions"] = []
        state["last_appended_gesture"] = None
        state["current_candidate"] = None
        state["candidate_counter"] = 0
    return jsonify({"status": "success"})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    print(f"🚀 VoxHand Flask API running at http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)