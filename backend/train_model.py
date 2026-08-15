import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

CSV_FILE = 'data/custom_asl_dataset.csv'
MODEL_FILE = 'models/gesture_model.pkl'

if not os.path.exists(CSV_FILE):
    print(f"Error: {CSV_FILE} not found. Run collect_data.py first!")
    exit(1)

print("Loading custom dataset...")
df = pd.read_csv(CSV_FILE)

X_raw = df.drop(columns=['label']).values
y = df['label'].values

def add_distance_features(X_data):
    num_samples = X_data.shape[0]
    joints = X_data.reshape(num_samples, 21, 3)
    
    distances = []
    fingertip_indices = [4, 8, 12, 16, 20]
    
    # Wrist to Fingertips
    for tip in fingertip_indices:
        dist = np.linalg.norm(joints[:, tip, :] - joints[:, 0, :], axis=1)
        distances.append(dist)
        
    # Adjacent Fingertips
    for i in range(len(fingertip_indices) - 1):
        dist = np.linalg.norm(joints[:, fingertip_indices[i], :] - joints[:, fingertip_indices[i+1], :], axis=1)
        distances.append(dist)
        
    # Thumb to Index / Middle
    distances.append(np.linalg.norm(joints[:, 4, :] - joints[:, 8, :], axis=1))
    distances.append(np.linalg.norm(joints[:, 4, :] - joints[:, 12, :], axis=1))

    extra_features = np.column_stack(distances)
    return np.hstack((X_data, extra_features))

print("Engineering spatial distance features...")
X_engineered = add_distance_features(X_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Random Forest Classifier...")
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("=" * 50)
print(f"CUSTOM MODEL ACCURACY: {acc * 100:.2f}%")
print("=" * 50)

joblib.dump(clf, MODEL_FILE)
print(f"Model successfully saved to '{MODEL_FILE}'!")