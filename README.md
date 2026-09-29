# Land Cover Classification (EuroSAT) — CPU-only build

Classifies a satellite image patch into one of 10 land-cover types
(Forest, Water, Residential, Industrial, AnnualCrop, etc.) using transfer
learning on a pretrained ResNet18. Built for a laptop with **no NVIDIA
GPU** (Intel Iris Xe integrated graphics) — everything here runs on CPU.

This is Project 1 of a 3-project computer vision series (Land Cover
Classification → Object Detection → Change Detection).

---

## 0. Setup (do this once)

```powershell
cd path\to\landcover
python -m venv landcover_env
landcover_env\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Verify it worked:
```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
`cuda.is_available()` printing `False` is **expected and fine** — we
installed the CPU-only build on purpose.

---

## 1. Run order

1. **Explore the data first.**
   ```powershell
   python 01_explore_data.py
   ```
   Downloads EuroSAT (~90MB, one-time) into `data/`, prints how many
   images are in each of the 10 classes, and saves
   `outputs/sample_grid.png` — open it and sanity-check that each image
   roughly matches its label.

2. **Train the model.**
   ```powershell
   python 02_train.py
   ```
   Splits data 80/20 train/test, builds a ResNet18 with its final layer
   replaced for 10 classes, and trains for 8 epochs. Saves the trained
   weights to `outputs/landcover_resnet18.pt`.

   **First time running this?** Open `config.py`, set `SUBSET_SIZE = 3000`,
   and run it once as a quick smoke test (a few minutes) before committing
   to the full run. Set it back to `None` afterward for real training.

3. **Evaluate on the test set.**
   ```powershell
   python 03_evaluate.py
   ```
   Prints overall accuracy and a per-class precision/recall report, and
   saves `outputs/confusion_matrix.png` showing exactly which classes the
   model confuses (e.g. Forest vs. HerbaceousVegetation, since both are
   green from above).

4. **(Optional) Try it on your own image.**
   ```powershell
   python 04_predict_single_image.py path\to\any_satellite_patch.jpg
   ```
   Prints the predicted class and the confidence for all 10 classes.

---

## 2. Project structure

```
landcover/
├── README.md
├── requirements.txt
├── config.py                    # all settings: paths, device, epochs, batch size
├── dataset.py                   # loads EuroSAT, does the train/test split
├── 01_explore_data.py           # download + inspect the data
├── 02_train.py                  # transfer learning + training loop
├── 03_evaluate.py               # accuracy + confusion matrix
├── 04_predict_single_image.py   # run the trained model on any one image
├── data/                        # EuroSAT downloads here (auto-created)
└── outputs/                     # trained model + plots go here (auto-created)
```

## 3. Why each design choice, in one line

- **ResNet18, not a bigger model** — small enough to train on CPU in
  reasonable time; EuroSAT images are only 64x64, so a huge model is overkill.
- **Transfer learning (ImageNet-pretrained), not training from scratch** —
  the model already knows general visual features (edges, textures);
  we only teach it the mapping to our 10 classes, which needs far less
  data and time.
- **Fixed random seed in the train/test split** — so re-running
  `03_evaluate.py` days later always tests on the same held-out images
  used during training, not a different random split.

## 4. Honest limitations

- EuroSAT patches are single fixed-size crops, not full satellite scenes —
  a real deployment would need to tile a large scene into patches first.
- 8 epochs is a reasonable starting point, not a tuned final number —
  if test accuracy looks low, try raising `NUM_EPOCHS` in `config.py`.
- This only classifies a whole patch as one class — it does not localize
  *where* in the patch each land type is (that's closer to what Project 2,
  object detection, and Project 3, change detection, will build toward).
