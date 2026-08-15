import cv2
import numpy as np
import joblib
import mediapipe as mp
import pyttsx3
import threading
import os
import sys
import time
import winsound
import difflib
from datetime import datetime
from wordfreq import top_n_list

# Optional Clipboard Support
try:
	import pyperclip
	HAS_PYPERCLIP = True
except ImportError:
	HAS_PYPERCLIP = False

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
CONFIG = {
	"MODEL_PATH": "models/hand_landmarker.task",
	"GESTURE_MODEL_PATH": "models/gesture_model.pkl",
	"CONFIRMATION_FRAMES": 6,       # Frames needed to confirm a gesture (~0.2s)
	"CONFIDENCE_THRESHOLD": 0.45,   # Minimum prediction probability
	"VOICE_RATE": 155,              # Text-to-speech speed rate
	"VOICE_KEYWORDS": ["heera", "india", "en-in", "veena"], # Preferred voice keywords
	"LOG_FILE": "voxhand_transcripts.txt"
}

# Loads top 25,000 most common English words in uppercase
DICTIONARY = [w.upper() for w in top_n_list('en', 25000) if w.isalpha()]

def get_word_suggestions(prefix, top_k=3):
    if not prefix:
        return []
    prefix = prefix.upper()

    # 1. Exact prefix matches (already sorted by frequency)
    exact_matches = [w for w in DICTIONARY if w.startswith(prefix) and w != prefix]
    if len(exact_matches) >= top_k:
        return exact_matches[:top_k]

    # 2. Fuzzy matches for typos
    close_matches = difflib.get_close_matches(prefix, DICTIONARY, n=top_k, cutoff=0.6)
    combined = list(dict.fromkeys(exact_matches + close_matches))
    return combined[:top_k]

# ==========================================
# AUDIO FEEDBACK WORKER (WINSOUND BEEP)
# ==========================================
def play_feedback_sound(sound_type="confirm"):
	def sound_worker():
		try:
			if sound_type == "delete":
				winsound.Beep(400, 90)
				winsound.Beep(350, 90)
			elif sound_type == "confirm":
				winsound.Beep(1000, 70)
			elif sound_type == "autocomplete":
				winsound.Beep(1200, 60)
				winsound.Beep(1500, 80)
		except Exception:
			pass
			
	threading.Thread(target=sound_worker, daemon=True).start()

# ==========================================
# CLIPBOARD & EXPORT WORKERS
# ==========================================
def copy_to_clipboard(text):
	if not text.strip():
		return
		
	def clipboard_worker():
		try:
			if HAS_PYPERCLIP:
				pyperclip.copy(text)
			else:
				import tkinter as tk
				root = tk.Tk()
				root.withdraw()
				root.clipboard_clear()
				root.clipboard_append(text)
				root.update()
				root.destroy()
		except Exception:
			pass
			
	threading.Thread(target=clipboard_worker, daemon=True).start()

def log_sentence_to_file(sentence):
	if not sentence.strip():
		return False
		
	try:
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		with open(CONFIG["LOG_FILE"], "a", encoding="utf-8") as f:
			f.write(f"[{timestamp}] {sentence.strip()}\n")
		return True
	except Exception as e:
		print(f"[Log Error]: {e}", file=sys.stderr)
		return False

# ==========================================
# TEXT-TO-SPEECH (TTS) WORKER
# ==========================================
def speak_text_async(text_to_speak, voice_keywords, rate):
	def speech_worker():
		try:
			cleaned_text = str(text_to_speak).strip()
			if not cleaned_text:
				return
				
			engine = pyttsx3.init()
			engine.setProperty('rate', rate)
			
			voices = engine.getProperty('voices')
			selected_voice = None
			
			for voice in voices:
				voice_str = (voice.id + voice.name).lower()
				if any(kw in voice_str for kw in voice_keywords):
					selected_voice = voice.id
					break
			
			if selected_voice:
				engine.setProperty('voice', selected_voice)
				
			engine.say(cleaned_text)
			engine.runAndWait()
		except Exception as e:
			print(f"[TTS Error]: {e}", file=sys.stderr)
			
	threading.Thread(target=speech_worker, daemon=True).start()

# ==========================================
# FEATURE EXTRACTION PIPELINE
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
# MAIN RUNTIME
# ==========================================
def main():
	if not os.path.exists(CONFIG["GESTURE_MODEL_PATH"]) or not os.path.exists(CONFIG["MODEL_PATH"]):
		print("Error: Missing model files. Ensure model task and pkl files exist.")
		sys.exit(1)

	clf = joblib.load(CONFIG["GESTURE_MODEL_PATH"])

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

	print("=" * 60)
	print("VOXHAND TRANSLATOR (WITH REAL-TIME WORD SUGGESTIONS)")
	print("Select Operating Mode:")
	print("  [1] Letters & Symbols (A-Z, Space, Del, .)")
	print("  [2] Numbers & Symbols (0-10, Space, Del, .)")
	print("  [0] All Gestures")
	print("=" * 60)

	start_choice = input("Enter Mode Choice (0-2) [Default 1]: ").strip()
	current_mode_key = start_choice if start_choice in MODES else '1'

	BaseOptions = mp.tasks.BaseOptions
	HandLandmarker = mp.tasks.vision.HandLandmarker
	HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

	options = HandLandmarkerOptions(
		base_options=BaseOptions(model_asset_path=CONFIG["MODEL_PATH"]),
		running_mode=mp.tasks.vision.RunningMode.IMAGE,
		num_hands=1
	)

	cap = cv2.VideoCapture(0)
	# user_input = input("0 for default webcam, 1 for USB camera [Default 0]: ").strip()
	# camera_index = int(user_input) if user_input.isdigit() else 0
	# cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
     

	last_appended_gesture = None
	current_candidate = None
	candidate_counter = 0
	current_word = ""
	full_sentence = ""
	status_msg = ""
	status_time = 0

	with HandLandmarker.create_from_options(options) as landmarker:
		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break
				
			frame = cv2.flip(frame, 1)
			h, w, _ = frame.shape
			rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
			
			result = landmarker.detect(mp_image)
			
			# Compute real-time suggestions
			suggestions = get_word_suggestions(current_word, top_k=3)
			
			key = cv2.waitKeyEx(1)
			
			# OpenCV Extended Keys: ESC = 27, F1 = 7340032 or 112, F2 = 7405568 or 113, F3 = 7471104 or 114
			if key in (27, ord('q'), ord('Q')):  # ESC or Q to exit
				break
			elif key in (ord('c'), ord('C')):  # Clear text
				current_word = ""
				full_sentence = ""
				last_appended_gesture = None
				status_msg = "Cleared All Text!"
				status_time = time.time()
			elif key in (ord('s'), ord('S')):  # Save to File
				if full_sentence.strip():
					if log_sentence_to_file(full_sentence):
						status_msg = f"Saved to {CONFIG['LOG_FILE']}!"
					else:
						status_msg = "Failed to save file."
				else:
					status_msg = "Nothing to save!"
				status_time = time.time()
			# F1 Auto-Complete
			elif key in (7340032, 112) and len(suggestions) >= 1:
				full_sentence += suggestions[0] + " "
				current_word = ""
				play_feedback_sound("autocomplete")
				status_msg = f"Completed: {suggestions[0]}"
				status_time = time.time()
			# F2 Auto-Complete
			elif key in (7405568, 113) and len(suggestions) >= 2:
				full_sentence += suggestions[1] + " "
				current_word = ""
				play_feedback_sound("autocomplete")
				status_msg = f"Completed: {suggestions[1]}"
				status_time = time.time()
			# F3 Auto-Complete
			elif key in (7471104, 114) and len(suggestions) >= 3:
				full_sentence += suggestions[2] + " "
				current_word = ""
				play_feedback_sound("autocomplete")
				status_msg = f"Completed: {suggestions[2]}"
				status_time = time.time()
			elif chr(key & 0xFF) in MODES if 0 <= (key & 0xFF) < 128 else False:
				current_mode_key = chr(key & 0xFF)

			mode_name, allowed_classes = MODES[current_mode_key]

			if result.hand_landmarks:
				landmarks = result.hand_landmarks[0]
				
				for lm in landmarks:
					cx, cy = int(lm.x * w), int(lm.y * h)
					cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
					
				features = extract_features(landmarks, w, h)
				probabilities = clf.predict_proba(features)[0]
				
				valid_indices = [i for i, c in enumerate(clf.classes_) if c in allowed_classes]
				
				if valid_indices:
					filtered_probs = probabilities[valid_indices]
					filtered_classes = clf.classes_[valid_indices]
					
					max_prob = np.max(filtered_probs)
					detected_pred = str(filtered_classes[np.argmax(filtered_probs)]).lower()
					
					if max_prob > CONFIG["CONFIDENCE_THRESHOLD"]:
						cv2.putText(frame, f"Gesture: {detected_pred.upper()} ({max_prob*100:.0f}%)", 
									(20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
						
						if detected_pred == current_candidate:
							candidate_counter += 1
						else:
							current_candidate = detected_pred
							candidate_counter = 0

						if candidate_counter >= CONFIG["CONFIRMATION_FRAMES"]:
							if detected_pred != last_appended_gesture:
								last_appended_gesture = detected_pred
								
								play_feedback_sound("delete" if detected_pred == 'del' else "confirm")

								# EDIT COMMAND: DELETE
								if detected_pred == 'del':
									if len(current_word) > 0:
										if current_word.endswith('10'):
											current_word = current_word[:-2]
										else:
											current_word = current_word[:-1]
									elif len(full_sentence) > 0:
										full_sentence = full_sentence[:-1]
								
								# EDIT COMMAND: SPACE
								elif detected_pred == 'space':
									if current_word:
										full_sentence += current_word + " "
										current_word = ""
								
								# EDIT COMMAND: FULL STOP
								elif detected_pred == '.':
									if current_word:
										full_sentence += current_word + ". "
										current_word = ""
									elif full_sentence and not full_sentence.endswith('. '):
										full_sentence = full_sentence.strip() + ". "
										
									speak_text_async(full_sentence, CONFIG["VOICE_KEYWORDS"], CONFIG["VOICE_RATE"])
									copy_to_clipboard(full_sentence)
									status_msg = "Copied to Clipboard!"
									status_time = time.time()

								# NUMBERS HANDLING
								elif detected_pred in numbers_set:
									speak_text_async(detected_pred, CONFIG["VOICE_KEYWORDS"], CONFIG["VOICE_RATE"])
									current_word += detected_pred.upper()
									
								# LETTERS HANDLING
								else:
									current_word += detected_pred.upper()
					else:
						cv2.putText(frame, "Uncertain Gesture...", (20, 45),
									cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
						candidate_counter = 0
			else:
				cv2.putText(frame, "No Hand Detected", (20, 45),
							cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
				last_appended_gesture = None
				current_candidate = None
				candidate_counter = 0

			# UI: Word Suggestions Overlay Box (Above Sentence Banner)
			cv2.rectangle(frame, (0, h - 130), (w, h - 85), (45, 45, 45), -1)
			suggestion_str = "Suggestions: "
			if suggestions:
				suggestion_str += "  |  ".join([f"[F{i+1}] {word}" for i, word in enumerate(suggestions)])
			else:
				suggestion_str += "None"
			cv2.putText(frame, suggestion_str, (15, h - 100), 
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

			# UI: Sentence & Current Word Banner
			cv2.rectangle(frame, (0, h - 85), (w, h - 30), (25, 25, 25), -1)
			display_text = f"Sentence: {full_sentence} | Current Input: [{current_word}]"
			cv2.putText(frame, display_text, (15, h - 48), 
						cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
				
			# Status Notification Bar
			if status_msg and (time.time() - status_time < 2.5):
				cv2.putText(frame, f"[{status_msg}]", (w - 320, 45), 
							cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

			# Bottom Controls Bar
			cv2.putText(frame, f"Mode: {mode_name} | Hotkeys: 1=Letters, 2=Numbers, 0=All | 'C'=Clear | 'S'=Save Log", 
						(15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

			cv2.imshow("VoxHand Live Translator", frame)

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()