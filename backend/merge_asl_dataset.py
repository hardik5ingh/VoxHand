import os
import glob
import pandas as pd
import numpy as np

DATASET_DIR = "processed_combine_asl_dataset"
OUTPUT_CSV = "large_gesture_dataset.csv"

# Header with label and 63 landmark coordinates (x, y, z for 21 points)
header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')]

all_rows = []
print("Searching and merging extracted landmark files...")

if not os.path.exists(DATASET_DIR):
    print(f"ERROR: Directory '{DATASET_DIR}' not found in current folder!")
    exit()

for class_label in os.listdir(DATASET_DIR):
    class_folder = os.path.join(DATASET_DIR, class_label)
    if not os.path.isdir(class_folder):
        continue

    # Look for files inside subfolder (.csv, .npy, or .txt)
    file_paths = glob.glob(os.path.join(class_folder, "*.*"))
    print(f"Processing class '{class_label}': found {len(file_paths)} files...")

    for file_path in file_paths:
        try:
            if file_path.endswith('.csv') or file_path.endswith('.txt'):
                data = pd.read_csv(file_path, header=None).values.flatten()
            elif file_path.endswith('.npy'):
                data = np.load(file_path).flatten()
            else:
                continue

            # Check if array contains 63 coordinate values
            if len(data) == 63:
                all_rows.append([class_label] + list(data))
        except Exception:
            continue

if len(all_rows) == 0:
    print("\n[!] No valid landmark files found. Check what file extensions are inside the letter folders!")
else:
    df = pd.DataFrame(all_rows, columns=header)
    df.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "="*50)
    print(f"SUCCESS: Combined {len(df)} total rows across {df['label'].nunique()} classes.")
    print(f"Saved merged dataset to '{OUTPUT_CSV}'")
    print("="*50)