"""
02_train.py — Steps 4, 5, 6 of the plan: split data, build the model via
transfer learning, and train it.

Run this after 01_explore_data.py has confirmed the data looks right.
On CPU (Intel Iris Xe / no NVIDIA GPU), 8 epochs on the full dataset
typically takes somewhere around 30-60 minutes depending on your CPU.

To do a QUICK first test run before committing to the full training time:
  Open config.py and set SUBSET_SIZE = 3000, then run this script.
  Once you've confirmed it works end-to-end, set SUBSET_SIZE back to None
  and run the full training.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

import config
from dataset import get_train_test_split


def build_model():
    """
    Transfer learning: start from a ResNet18 already trained on ImageNet
    (it already knows how to recognize edges, textures, shapes), then
    replace only its final layer so it outputs 10 EuroSAT classes instead
    of ImageNet's 1000 classes.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(config.CLASS_NAMES))
    return model.to(config.DEVICE)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  training", leave=False):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    print(f"Training on device: {config.DEVICE}")
    if config.SUBSET_SIZE:
        print(f"NOTE: SUBSET_SIZE={config.SUBSET_SIZE} — using a subset for a quick test run.")

    train_dataset, test_dataset = get_train_test_split()
    print(f"Train images: {len(train_dataset)} | Test images: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...\n")
    start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{config.NUM_EPOCHS} — loss: {loss:.4f} — train accuracy: {acc*100:.2f}% — took {epoch_time:.1f}s")

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/60:.1f} minutes.")

    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"Model saved to: {config.MODEL_PATH}")
    print("\nNext step: run 03_evaluate.py to check accuracy on the test set.")


if __name__ == "__main__":
    main()
