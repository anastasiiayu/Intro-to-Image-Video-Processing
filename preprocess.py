import os
import numpy as np
import pandas as pd
import cv2

DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train", "train") 
TEST_DIR = os.path.join(DATA_DIR, "test", "test")      
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
 
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess(path):
    """Load a single image by path, convert to grayscale, normalize to [0, 1]."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read {path}")
    normalized = gray.astype(np.float32) / 255.0
    return normalized

def find_train_image(image_id, label):
    """Look in the labelled subfolder first, then scan others as a fallback."""
    primary = os.path.join(TRAIN_DIR, str(label), f"{image_id}.png")
    if os.path.exists(primary):
        return primary
    # Fallback: scan all class folders (shouldn't be needed if CSV matches folders)
    for cls in range(10):
        candidate = os.path.join(TRAIN_DIR, str(cls), f"{image_id}.png")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find train image {image_id}.png")
 
 
def process_train():
    print("Loading train.csv ...")
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"  {len(train_df)} training rows")
 
    X = np.empty((len(train_df), 32, 32), dtype=np.float32)
    y = train_df["Category"].values.astype(np.int64)
    ids = train_df["Id"].values
 
    print("Preprocessing training images ...")
    for i, (image_id, label) in enumerate(zip(ids, y)):
        path = find_train_image(image_id, label)
        X[i] = load_and_preprocess(path)
        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1}/{len(train_df)}")
 
    print(f"  X_train shape: {X.shape}, dtype: {X.dtype}")
    print(f"  X_train range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  y_train shape: {y.shape}, classes: {sorted(np.unique(y).tolist())}")
 
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y)
    np.save(os.path.join(OUTPUT_DIR, "train_ids.npy"), ids)
    print(f"  saved -> {OUTPUT_DIR}/X_train.npy, y_train.npy, train_ids.npy")
 
 
def process_test():
    print("\nLoading test.csv ...")
    test_df = pd.read_csv(TEST_CSV)
    print(f"  {len(test_df)} test rows")
 
    ids = test_df["Id"].values
    X = np.empty((len(ids), 32, 32), dtype=np.float32)
 
    print("Preprocessing test images ...")
    for i, image_id in enumerate(ids):
        path = os.path.join(TEST_DIR, f"{image_id}.png")
        X[i] = load_and_preprocess(path)
        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1}/{len(ids)}")
 
    print(f"  X_test shape: {X.shape}")
 
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "test_ids.npy"), ids)
    print(f"  saved -> {OUTPUT_DIR}/X_test.npy, test_ids.npy")
 
 
def main():
    process_train()
    process_test()
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()