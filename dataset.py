"""
dataset.py — loads EuroSAT and splits it into train/test.
Shared by 01_explore_data.py, 02_train.py, and 03_evaluate.py so we never
accidentally use a different split in training vs. evaluation.
"""

import torch
from torch.utils.data import random_split, Subset
from torchvision import datasets, transforms

import config


def get_transforms():
    """
    Resize + normalize images so they match what ResNet18 expects
    (ResNet18 was pretrained on ImageNet, which used these exact mean/std values).
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_full_dataset():
    """
    Downloads EuroSAT (RGB version) into config.DATA_DIR the first time this
    runs, then loads it from disk on every run after that.
    """
    transform = get_transforms()
    dataset = datasets.EuroSAT(
        root=config.DATA_DIR,
        download=True,
        transform=transform,
    )
    return dataset


def get_train_test_split():
    """
    Returns (train_dataset, test_dataset) using a fixed random seed so the
    split is identical every time you run training or evaluation.
    """
    full_dataset = load_full_dataset()

    if config.SUBSET_SIZE is not None:
        # Quick smoke-test mode: only use a random subset of the full data.
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(full_dataset), generator=generator)[:config.SUBSET_SIZE]
        full_dataset = Subset(full_dataset, indices.tolist())

    total = len(full_dataset)
    train_size = int(config.TRAIN_SPLIT * total)
    test_size = total - train_size

    generator = torch.Generator().manual_seed(42)  # fixed seed = reproducible split
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size], generator=generator
    )
    return train_dataset, test_dataset
