import os
import copy
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split


PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = "outputs"

X_TRAIN_PATH = os.path.join(PROCESSED_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(PROCESSED_DIR, "y_train.npy")
X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test.npy")
TEST_IDS_PATH = os.path.join(PROCESSED_DIR, "test_ids.npy")

SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission_ensemble.csv")

BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.001
VALID_SIZE = 0.2

SPLIT_SEED = 42
SEEDS = [42, 123, 2026]
LABEL_SMOOTHING = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_ELASTIC = hasattr(transforms, "ElasticTransform")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NPYDataset(Dataset):
    def __init__(self, X, y=None, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

        self.augmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            make_elastic_transform(alpha=20.0, sigma=4.0)
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]

        if self.augment:
            image = self.augmentation(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if self.y is None:
            return image

        label = torch.tensor(self.y[idx], dtype=torch.long)
        return image, label


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


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0
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

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def predict_probabilities(model, loader):
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


def train_single_model(seed, X_train, y_train, X_test):
    print("\n" + "=" * 60)
    print(f"Training model with seed {seed}")
    print("=" * 60)

    set_seed(seed)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALID_SIZE,
        random_state=SPLIT_SEED,
        stratify=y_train
    )

    train_dataset = NPYDataset(X_tr, y_tr, augment=True)
    val_dataset = NPYDataset(X_val, y_val, augment=False)
    test_dataset = NPYDataset(X_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    best_val_acc = 0
    best_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)

        print(
            f"Seed {seed} | Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping for seed {seed} at epoch {epoch + 1}")
            break

    model.load_state_dict(best_weights)

    model_path = os.path.join(OUTPUT_DIR, f"best_cnn_seed_{seed}.pt")
    torch.save(best_weights, model_path)

    print(f"Best val acc for seed {seed}: {best_val_acc:.4f}")
    print(f"Saved model: {model_path}")

    test_probs = predict_probabilities(model, test_loader)

    return test_probs, best_val_acc


def main():
    print("Loading data...")
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    test_ids = np.load(TEST_IDS_PATH)

    print(f"Using device: {DEVICE}")

    all_test_probs = []
    val_scores = []

    for seed in SEEDS:
        test_probs, val_acc = train_single_model(seed, X_train, y_train, X_test)
        all_test_probs.append(test_probs)
        val_scores.append(val_acc)

    mean_probs = np.mean(all_test_probs, axis=0)
    final_preds = np.argmax(mean_probs, axis=1)

    submission = pd.DataFrame({
        "Id": test_ids,
        "Category": final_preds
    })

    submission.to_csv(SUBMISSION_PATH, index=False)

    print("\n" + "=" * 60)
    print("Ensemble finished")
    print("=" * 60)
    print(f"Validation scores: {val_scores}")
    print(f"Average validation accuracy: {np.mean(val_scores):.4f}")
    print(f"Submission saved to: {SUBMISSION_PATH}")


def make_elastic_transform(alpha=20.0, sigma=4.0):
    """Build an elastic deformation transform.
    If the installed torchvision has transforms.ElasticTransform (>=0.13),
    use it directly. Otherwise fall back to a hand-rolled implementation
    that does the same thing (gaussian-smoothed displacement field +
    grid_sample to warp the image).
    """
    if HAS_ELASTIC:
        return transforms.ElasticTransform(alpha=alpha, sigma=sigma)
    # Fallback: a callable that takes a CxHxW float tensor and returns one.
    import torch.nn.functional as F
    class _Elastic:
        def __init__(self, alpha, sigma):
            self.alpha = alpha
            self.sigma = sigma
        def __call__(self, img):
            # img shape: (C, H, W)
            c, h, w = img.shape
            # Random displacement field, then gaussian-smooth it
            dx = torch.randn(1, 1, h, w) * self.alpha
            dy = torch.randn(1, 1, h, w) * self.alpha
            # Gaussian blur via separable conv
            ks = max(3, int(self.sigma * 4) | 1)  # odd kernel size
            xs = torch.arange(ks, dtype=torch.float32) - ks // 2
            g = torch.exp(-(xs ** 2) / (2 * self.sigma ** 2))
            g = (g / g.sum()).view(1, 1, 1, ks)
            dx = F.conv2d(dx, g, padding=(0, ks // 2))
            dx = F.conv2d(dx, g.transpose(2, 3), padding=(ks // 2, 0))
            dy = F.conv2d(dy, g, padding=(0, ks // 2))
            dy = F.conv2d(dy, g.transpose(2, 3), padding=(ks // 2, 0))
            # Build the sampling grid
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, h),
                torch.linspace(-1, 1, w),
                indexing="ij",
            )
            grid = torch.stack(
                (xx + dx[0, 0] * (2.0 / w), yy + dy[0, 0] * (2.0 / h)),
                dim=-1,
            ).unsqueeze(0)
            warped = F.grid_sample(
                img.unsqueeze(0), grid, mode="bilinear",
                padding_mode="zeros", align_corners=True,
            )
            return warped[0]
    return _Elastic(alpha, sigma)

if __name__ == "__main__":
    main()