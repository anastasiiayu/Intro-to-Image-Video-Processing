"""
67676767676767676767
  1. Load the .npy files and print their shapes/dtypes/ranges.
  2. Display 5 random images per class (0-9) in a 10x5 grid
67676767676767676767
Run this after preprocess.py so npy files are generated pls 
"""

import os
import numpy as np
import matplotlib.pyplot as plt


PROCESSED_DIR = os.path.join("data", "processed")
SAMPLES_PER_CLASS = 5
SEED = 42


def check_shapes():
    """Load arrays and print their shapes, dtypes, and value ranges."""
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    test_ids = np.load(os.path.join(PROCESSED_DIR, "test_ids.npy"))

    print("=" * 60)
    print("ARRAY SHAPES AND PROPERTIES")
    print("=" * 60)
    print(f"X_train : shape={X_train.shape}, dtype={X_train.dtype}, "
          f"range=[{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"y_train : shape={y_train.shape}, dtype={y_train.dtype}, "
          f"classes={sorted(np.unique(y_train).tolist())}")
    print(f"X_test  : shape={X_test.shape}, dtype={X_test.dtype}, "
          f"range=[{X_test.min():.3f}, {X_test.max():.3f}]")
    print(f"test_ids: shape={test_ids.shape}, dtype={test_ids.dtype}")

    print("\nClass distribution in y_train:")
    classes, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(classes, counts):
        print(f"  class {cls}: {count} samples")

    # Expectation check
    expected = {
        "X_train shape": (X_train.shape == (17000, 32, 32)),
        "y_train shape": (y_train.shape == (17000,)),
        "X_test shape":  (X_test.shape  == (3000, 32, 32)),
        "X_train range": (0.0 <= X_train.min() and X_train.max() <= 1.0),
        "10 classes":    (len(classes) == 10),
    }
    print("\nChecks:")
    for name, ok in expected.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    return X_train, y_train


def visualize_samples(X_train, y_train):
    """Plot 5 random images per class in a 10-row x 5-col grid."""
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(
        nrows=10, ncols=SAMPLES_PER_CLASS,
        figsize=(SAMPLES_PER_CLASS * 1.5, 10 * 1.5),
    )

    for cls in range(10):
        # Indices of all images with this label
        class_indices = np.where(y_train == cls)[0]
        # Randomly pick SAMPLES_PER_CLASS of them
        chosen = rng.choice(class_indices, size=SAMPLES_PER_CLASS, replace=False)

        for col, idx in enumerate(chosen):
            ax = axes[cls, col]
            ax.imshow(X_train[idx], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"label {cls}", rotation=0, ha="right", va="center",
                              fontsize=12, labelpad=20)

    plt.tight_layout()

    out_path = os.path.join(PROCESSED_DIR, "samples_per_class.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved visualization -> {out_path}")
    plt.show()


def main():
    X_train, y_train = check_shapes()
    visualize_samples(X_train, y_train)
    print("done")


if __name__ == "__main__":
    main()