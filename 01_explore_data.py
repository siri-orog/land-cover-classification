"""
01_explore_data.py — Step 2 & 3 of the plan: download EuroSAT and look at it
before building anything. Run this first.

What it does:
1. Downloads EuroSAT into data/ (only happens once; skipped on later runs)
2. Prints how many images exist per class (checks for class imbalance)
3. Saves a grid of sample images (one per class) to outputs/sample_grid.png
   so you can visually confirm the data looks right
"""

import matplotlib
matplotlib.use("Agg")  # save plots to file instead of trying to pop up a window
import matplotlib.pyplot as plt
from collections import Counter

import config
from dataset import load_full_dataset


def main():
    print(f"Device available: {config.DEVICE}")
    print("Downloading / loading EuroSAT (first run downloads ~90MB)...")
    dataset = load_full_dataset()
    print(f"Total images in dataset: {len(dataset)}")

    # ---- Class distribution ----
    labels = [label for _, label in dataset]
    counts = Counter(labels)
    print("\nImages per class:")
    for class_idx in sorted(counts.keys()):
        class_name = config.CLASS_NAMES[class_idx]
        print(f"  {class_name:22s}: {counts[class_idx]}")

    # ---- Sample image grid: one example per class ----
    print("\nSaving one sample image per class to outputs/sample_grid.png ...")
    seen_classes = {}
    for img_tensor, label in dataset:
        if label not in seen_classes:
            seen_classes[label] = img_tensor
        if len(seen_classes) == len(config.CLASS_NAMES):
            break

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for idx, class_idx in enumerate(sorted(seen_classes.keys())):
        ax = axes[idx // 5][idx % 5]
        img = seen_classes[class_idx]
        # undo normalization just for display purposes
        img = img.permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = img.clip(0, 1)
        ax.imshow(img)
        ax.set_title(config.CLASS_NAMES[class_idx], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out_path = f"{config.OUTPUT_DIR}/sample_grid.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")
    print("\nOpen that file and check: does each image roughly match its label?")


if __name__ == "__main__":
    main()
