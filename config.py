"""
config.py — central settings for the Land Cover Classification project.
Change values here instead of hunting through every script.
"""

import os
import torch

# ---- Paths ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")          # EuroSAT downloads here
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")       # trained model + plots go here
MODEL_PATH = os.path.join(OUTPUT_DIR, "landcover_resnet18.pt")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Device ----
# Intel Iris Xe (or any non-NVIDIA GPU) => CUDA is not available => CPU-only.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Data ----
TRAIN_SPLIT = 0.8          # 80% train, 20% test
BATCH_SIZE = 32            # CPU-friendly. Lower this to 16 if training feels slow/heavy.
IMAGE_SIZE = 64            # EuroSAT RGB images are natively 64x64
NUM_WORKERS = 0            # 0 is safest on Windows; raise to 2-4 if it works fine for you

# ---- Training ----
NUM_EPOCHS = 8             # good balance of CPU time vs. accuracy for a first run
LEARNING_RATE = 1e-3
SUBSET_SIZE = None         # set to e.g. 5000 for a quick smoke-test run on a fraction of the data

# EuroSAT class names (10 land-cover categories), fixed order used by torchvision
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
