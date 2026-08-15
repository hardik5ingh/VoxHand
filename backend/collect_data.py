import cv2
import csv
import os
import numpy as np
import mediapipe as mp

MODEL_PATH = "models/hand_landmarker.task"
OUTPUT_CSV = "data/custom_asl_dataset.csv"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1
)


if not os.path.exists(OUTPUT_CSV):
    header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')]
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

cap = cv2.VideoCapture(0)

# Vocabulary list: A-Z, 0-9, space, del, .
vocab = [chr(i) for i in range(97, 123)] + [str(i) for i in range(11)] + ['space', 'del', '.']
current_idx = 0

recording = False
saved_count = 0

print("=" * 60)
print("GUI DATASET COLLECTOR (No Terminal Input)")
print("- Press 'S' to START recording current label")
print("- Press 'Q' to STOP recording current label")
print("- Press 'N' to go to NEXT label in list")
print("- Press 'P' to go to PREVIOUS label in list")
print("- Press ESC to EXIT script")
print("=" * 60)

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
        current_label = vocab[current_idx]
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC to exit
            break
        elif key == ord('s') or key == ord('S'):
            recording = True
        elif key == ord('q') or key == ord('Q'):
            recording = False
        elif key == ord('n') or key == ord('N'):
            recording = False
            current_idx = (current_idx + 1) % len(vocab)
            saved_count = 0
        elif key == ord('p') or key == ord('P'):
            recording = False
            current_idx = (current_idx - 1) % len(vocab)
            saved_count = 0

        # Process & Save Frame
        if recording and result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            coords[:, 0] *= (w / h)
            wrist = coords[0]
            coords = coords - wrist
            max_val = np.max(np.abs(coords))
            if max_val > 0:
                coords = coords / max_val
            
            features = coords.flatten().tolist()
            
            with open(OUTPUT_CSV, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_label] + features)
            
            saved_count += 1

        # Render Landmarks
        if result.hand_landmarks:
            for lm in result.hand_landmarks[0]:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # On-Screen Overlay Controls
        if recording:
            status = f"RECORDING '{current_label.upper()}': {saved_count} samples | Press 'Q' to Stop"
            color = (0, 255, 0)
        else:
            status = f"LABEL: '{current_label.upper()}' | 'S'=Start | 'Q'=Stop | 'N'=Next | 'P'=Prev"
            color = (0, 255, 255)

        cv2.putText(frame, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Custom Dataset Collector", frame)

cap.release()
cv2.destroyAllWindows()
