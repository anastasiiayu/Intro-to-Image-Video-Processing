"""
Error analysis for the ensemble.

Loads the three trained ensemble models, runs them on the val set with
the SAME split used during training, and produces:

  1. A confusion matrix showing which class pairs the ensemble confuses.
  2. A grid of all misclassified val images, with true and predicted labels,
     so you can visually inspect what's going wrong.

This helps decide whether to push for more accuracy (errors look fixable)
or stop and write the report (errors look genuinely ambiguous).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = "outputs"

X_TRAIN_PATH = os.path.join(PROCESSED_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(PROCESSED_DIR, "y_train.npy")

MODEL_PATHS = [
    os.path.join(OUTPUT_DIR, "best_cnn_seed_42.pt"),
    os.path.join(OUTPUT_DIR, "best_cnn_seed_123.pt"),
    os.path.join(OUTPUT_DIR, "best_cnn_seed_2026.pt"),
]

# Must match the split seed used in train_ensemble.py
SPLIT_SEED = 42
VALID_SIZE = 0.2
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NPYDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def predict_probabilities(model, loader):
    model.eval()
    all_probs = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            probs = softmax(model(images))
            all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data and reproduce the same train/val split used during training
    X = np.load(X_TRAIN_PATH)
    y = np.load(Y_TRAIN_PATH)
    _, X_val, _, y_val = train_test_split(
        X, y,
        test_size=VALID_SIZE,
        random_state=SPLIT_SEED,
        stratify=y,
    )

    val_dataset = NPYDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Get ensemble probabilities (mean of softmax across models)
    print("Running ensemble on validation set...")
    all_probs = []
    for path in MODEL_PATHS:
        print(f"  loading {path}")
        model = SimpleCNN().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        all_probs.append(predict_probabilities(model, val_loader))

    mean_probs = np.mean(all_probs, axis=0)
    preds = np.argmax(mean_probs, axis=1)
    pred_confidences = np.max(mean_probs, axis=1)

    # ---- Classification report ----
    print("\nClassification report:")
    print(classification_report(y_val, preds, digits=4))

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_val, preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=list(range(10)))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Ensemble Confusion Matrix (val acc = {(preds == y_val).mean():.4f})")
    cm_path = os.path.join(OUTPUT_DIR, "cm_ensemble_analysis.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved confusion matrix -> {cm_path}")

    # ---- Per-class error breakdown ----
    print("\nPer-class error breakdown:")
    for true_cls in range(10):
        mask = y_val == true_cls
        n_total = int(mask.sum())
        n_wrong = int(((preds != y_val) & mask).sum())
        if n_wrong > 0:
            wrong_into = preds[(preds != y_val) & mask]
            wrong_classes, wrong_counts = np.unique(wrong_into, return_counts=True)
            confused_with = ", ".join(
                f"{int(c)}x{int(n)}" for c, n in zip(wrong_classes, wrong_counts)
            )
            print(f"  class {true_cls}: {n_wrong}/{n_total} wrong, "
                  f"confused with [{confused_with}]")
        else:
            print(f"  class {true_cls}: 0/{n_total} wrong")

    # ---- Misclassified images grid ----
    error_idx = np.where(preds != y_val)[0]
    n_errors = len(error_idx)
    print(f"\nTotal errors: {n_errors}/{len(y_val)} "
          f"({n_errors / len(y_val):.4f} error rate)")

    if n_errors == 0:
        print("No errors to visualize.")
        return

    # Sort errors by confidence (most confident wrong predictions first --
    # those are the most interesting to inspect)
    error_idx_sorted = error_idx[np.argsort(-pred_confidences[error_idx])]

    n_cols = min(8, n_errors)
    n_rows = int(np.ceil(n_errors / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 2.0))
    axes = np.atleast_2d(axes)

    for plot_pos, idx in enumerate(error_idx_sorted):
        r, c = plot_pos // n_cols, plot_pos % n_cols
        ax = axes[r, c]
        ax.imshow(X_val[idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"true: {y_val[idx]}\npred: {preds[idx]} ({pred_confidences[idx]:.2f})",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for plot_pos in range(n_errors, n_rows * n_cols):
        r, c = plot_pos // n_cols, plot_pos % n_cols
        axes[r, c].axis("off")

    fig.suptitle(
        f"Misclassified validation images ({n_errors} total), sorted by confidence",
        fontsize=12,
    )
    plt.tight_layout()
    err_path = os.path.join(OUTPUT_DIR, "misclassified_val.png")
    plt.savefig(err_path, dpi=150, bbox_inches="tight")
    print(f"Saved misclassified images -> {err_path}")
    plt.show()


if __name__ == "__main__":
    main()