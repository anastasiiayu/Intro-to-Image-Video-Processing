import os
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# =========================
# Config
# =========================
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = "outputs"

X_TRAIN_PATH = os.path.join(PROCESSED_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(PROCESSED_DIR, "y_train.npy")
X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test.npy")
TEST_IDS_PATH = os.path.join(PROCESSED_DIR, "test_ids.npy")

MODEL_PATH = os.path.join(OUTPUT_DIR, "best_cnn.pt")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
VALID_SIZE = 0.2
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Dataset
# =========================
class NPYDataset(Dataset):
    def __init__(self, X, y=None):
        """
        X shape before: (N, 32, 32)
        We convert it to: (N, 1, 32, 32)
        """
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


# =========================
# Model
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),  # 32x32 -> 16x16

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),  # 16x16 -> 8x8

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)   # 8x8 -> 4x4
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


# =========================
# Training / Evaluation
# =========================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# =========================
# Prediction
# =========================
def predict_test(model, loader):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_preds)


# =========================
# Main
# =========================
def main():
    print("=" * 60)
    print("Loading processed data")
    print("=" * 60)

    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    test_ids = np.load(TEST_IDS_PATH)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"test_ids shape: {test_ids.shape}")
    print(f"Using device: {DEVICE}")

    # Train/validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    print("\nAfter split:")
    print(f"Train: {X_tr.shape}, {y_tr.shape}")
    print(f"Val  : {X_val.shape}, {y_val.shape}")

    # Datasets
    train_dataset = NPYDataset(X_tr, y_tr)
    val_dataset = NPYDataset(X_val, y_val)
    test_dataset = NPYDataset(X_test)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

    print("\n" + "=" * 60)
    print("Start training")
    print("=" * 60)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, MODEL_PATH)
            print(f"  -> Best model saved to {MODEL_PATH}")

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Load best model
    model.load_state_dict(best_model_weights)

    # Predict test
    print("\nPredicting test set...")
    test_preds = predict_test(model, test_loader)

    # Create submission
    submission = pd.DataFrame({
        "Id": test_ids,
        "Category": test_preds
    })

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to: {SUBMISSION_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()