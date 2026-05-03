"""
Test-time augmentation (TTA) inference for the ensemble.

For each test image:
  1. Generate the original image plus N augmented copies (mild augmentations,
     smaller than training-time augmentation).
  2. Run all of them through each ensemble member.
  3. Average the softmax probabilities across all augmentations and all models.
  4. Take argmax to get the final prediction.

The point: small affine perturbations let the network "vote" on slightly
different views of the same digit. This averages away cases where the model
is sensitive to a single pixel shift or 1-degree rotation.

This script does not retrain anything. It only does inference using the
weights already saved by train_ensemble.py.
"""

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = "outputs"

X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test.npy")
TEST_IDS_PATH = os.path.join(PROCESSED_DIR, "test_ids.npy")

SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission_ensemble_tta.csv")

MODEL_PATHS = [
    os.path.join(OUTPUT_DIR, "best_cnn_seed_42.pt"),
    os.path.join(OUTPUT_DIR, "best_cnn_seed_123.pt"),
    os.path.join(OUTPUT_DIR, "best_cnn_seed_2026.pt"),
]

BATCH_SIZE = 64

# Number of augmented copies per image (in addition to the original).
# More = slower, marginally better. 7 + the original = 8 views per image.
TTA_COPIES = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Same model definition as in train_ensemble.py
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TestDatasetPlain(Dataset):
    """Test images, no augmentation — produces the 'original view'."""
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)


class TestDatasetAugmented(Dataset):
    """Test images with mild random augmentation — used for TTA copies.

    Augmentation is INTENTIONALLY weaker than training-time augmentation:
    we want small perturbations around the real image, not large ones.
    """
    def __init__(self, X):
        self.X = X
        self.augmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=5,                  # half of training: 10 -> 5
                translate=(0.04, 0.04),     # half of training: 0.08 -> 0.04
                scale=(0.95, 1.05)          # narrower than training: 0.9-1.1 -> 0.95-1.05
            ),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.augmentation(self.X[idx])


def predict_probabilities(model, loader):
    """Run loader through model, return (N, 10) softmax probabilities."""
    model.eval()
    all_probs = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = softmax(outputs)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)


def main():
    print("Loading test data...")
    X_test = np.load(X_TEST_PATH)
    test_ids = np.load(TEST_IDS_PATH)
    print(f"  X_test: {X_test.shape}")
    print(f"  Using device: {DEVICE}")

    # Pre-build the plain (non-augmented) loader once
    plain_dataset = TestDatasetPlain(X_test)
    plain_loader = DataLoader(plain_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # We will accumulate probabilities here
    # Final shape: (n_models * (1 + TTA_COPIES), n_test, 10)
    all_probs = []

    for model_idx, path in enumerate(MODEL_PATHS):
        print(f"\nModel {model_idx + 1}/{len(MODEL_PATHS)}: {path}")
        model = SimpleCNN().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))

        # 1. Original (non-augmented) prediction
        print("  predicting on original images ...")
        probs = predict_probabilities(model, plain_loader)
        all_probs.append(probs)

        # 2. TTA_COPIES augmented predictions
        for copy_idx in range(TTA_COPIES):
            print(f"  TTA copy {copy_idx + 1}/{TTA_COPIES} ...")
            # New dataset each iteration -> new random augmentations
            aug_dataset = TestDatasetAugmented(X_test)
            aug_loader = DataLoader(aug_dataset, batch_size=BATCH_SIZE, shuffle=False)
            probs = predict_probabilities(model, aug_loader)
            all_probs.append(probs)

    # Average across all (models x augmentations)
    print(f"\nAveraging {len(all_probs)} probability matrices ...")
    mean_probs = np.mean(all_probs, axis=0)
    final_preds = np.argmax(mean_probs, axis=1)

    # Save submission
    submission = pd.DataFrame({
        "Id": test_ids,
        "Category": final_preds
    })
    submission.to_csv(SUBMISSION_PATH, index=False)

    print("\n" + "=" * 60)
    print("TTA + Ensemble inference finished")
    print("=" * 60)
    print(f"Models used      : {len(MODEL_PATHS)}")
    print(f"Augmentations    : 1 original + {TTA_COPIES} augmented = {1 + TTA_COPIES} views")
    print(f"Total predictions: {len(MODEL_PATHS) * (1 + TTA_COPIES)} per test image")
    print(f"Submission saved : {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()