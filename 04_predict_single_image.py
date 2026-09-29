"""
04_predict_single_image.py — optional bonus step: use the trained model on
ANY single image you give it (not just EuroSAT test images).

Usage:
    python 04_predict_single_image.py path/to/your_image.jpg

This is the "demo" step — good for showing the project working live,
e.g. to a mentor or in a presentation.
"""

import sys
import torch
from PIL import Image

import config
from dataset import get_transforms
from importlib import import_module

evaluate_module = import_module("03_evaluate")


def predict(image_path):
    model = evaluate_module.load_trained_model()
    transform = get_transforms()

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(config.DEVICE)  # add batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = probs.max(0)

    predicted_class = config.CLASS_NAMES[top_idx.item()]
    print(f"\nImage: {image_path}")
    print(f"Predicted class: {predicted_class}  (confidence: {top_prob.item()*100:.1f}%)")

    print("\nAll class probabilities:")
    sorted_probs = sorted(
        zip(config.CLASS_NAMES, probs.tolist()), key=lambda x: x[1], reverse=True
    )
    for name, p in sorted_probs:
        print(f"  {name:22s}: {p*100:5.1f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 04_predict_single_image.py path/to/image.jpg")
        sys.exit(1)
    predict(sys.argv[1])
