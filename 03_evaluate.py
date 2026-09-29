"""
03_evaluate.py — Step 7 of the plan: check how well the trained model
actually does on images it has never seen (the test set).

Run this after 02_train.py has finished and saved a model.
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from dataset import get_train_test_split


def load_trained_model():
    from torchvision.models import resnet18
    import torch.nn as nn

    model = resnet18(weights=None)  # architecture only; we load our own trained weights next
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(config.CLASS_NAMES))
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    return model


def main():
    print(f"Loading trained model from: {config.MODEL_PATH}")
    model = load_trained_model()

    _, test_dataset = get_train_test_split()
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    print(f"Evaluating on {len(test_dataset)} test images...")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest accuracy: {acc*100:.2f}%")

    print("\nPer-class report:")
    print(classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES))

    # ---- Confusion matrix plot ----
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(config.CLASS_NAMES)))
    ax.set_yticks(range(len(config.CLASS_NAMES)))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(config.CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (Test Accuracy: {acc*100:.1f}%)")

    for i in range(len(config.CLASS_NAMES)):
        for j in range(len(config.CLASS_NAMES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)

    plt.colorbar(im)
    plt.tight_layout()
    out_path = f"{config.OUTPUT_DIR}/confusion_matrix.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved confusion matrix to: {out_path}")
    print("Look at the off-diagonal cells — those are the classes the model confuses most.")


if __name__ == "__main__":
    main()
