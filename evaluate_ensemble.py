import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = "outputs"

X_TRAIN_PATH = os.path.join(PROCESSED_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(PROCESSED_DIR, "y_train.npy")


MODEL_PATHS = [
    os.path.join(OUTPUT_DIR, "best_cnn_seed_42.pt"),
    os.path.join(OUTPUT_DIR, "best_cnn_seed_123.pt"),
    os.path.join(OUTPUT_DIR, "best_cnn_seed_2026.pt"),
]

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
        return self.classifier(self.features(x))



def predict_probabilities(model, loader):
    model.eval()
    all_probs = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = softmax(outputs)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)


def main():
    print("Loading data...")
    X = np.load(X_TRAIN_PATH)
    y = np.load(Y_TRAIN_PATH)


    _, X_val, _, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    val_dataset = NPYDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    all_probs = []

    print("Loading models...")
    for path in MODEL_PATHS:
        print(f"Loading: {path}")

        model = SimpleCNN().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))

        probs = predict_probabilities(model, val_loader)
        all_probs.append(probs)


    mean_probs = np.mean(all_probs, axis=0)
    final_preds = np.argmax(mean_probs, axis=1)


    cm = confusion_matrix(y_val, final_preds)

    disp = ConfusionMatrixDisplay(cm, display_labels=list(range(10)))
    disp.plot()

    plt.title("Confusion Matrix - Ensemble")

    save_path = os.path.join(OUTPUT_DIR, "cm_ensemble.png")
    plt.savefig(save_path, dpi=150)

    print(f"Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()