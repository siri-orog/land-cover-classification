`README.md`**.

````markdown
# 🌍 Land Cover Classification using Deep Learning

A deep learning-based satellite image classification project that uses **ResNet18** and the **EuroSAT RGB dataset** to classify satellite images into 10 different land-cover categories.

This project provides a complete workflow from dataset exploration and preprocessing to model training, evaluation, and single-image prediction.

---

## 📌 Project Overview

Land cover classification is the process of identifying the type of land represented in a satellite image.

This project uses **ResNet18**, a Convolutional Neural Network (CNN), to classify satellite images into 10 land-cover categories.

### 🎯 Objectives

- Explore satellite image data
- Preprocess and prepare images
- Train a deep learning classification model
- Evaluate the trained model
- Generate a confusion matrix
- Predict the land-cover class of a single satellite image
- Provide a reusable deep learning pipeline

---

## 🛰️ Dataset

This project uses the **EuroSAT RGB dataset**.

The dataset contains satellite images belonging to 10 land-cover categories.

### Dataset Classes

| No. | Class |
|---:|---|
| 1 | Annual Crop |
| 2 | Forest |
| 3 | Herbaceous Vegetation |
| 4 | Highway |
| 5 | Industrial |
| 6 | Pasture |
| 7 | Permanent Crop |
| 8 | Residential |
| 9 | River |
| 10 | SeaLake |

The images are RGB satellite images with a resolution of **64 × 64 pixels**.

> The complete dataset is not included in this repository because it contains approximately 27,000 images and is around 186 MB.

---

## 🧠 Model

The project uses **ResNet18**, a CNN architecture based on residual learning.

ResNet18 is used to learn visual patterns from satellite images and classify them into the 10 land-cover categories.

### Training Configuration

| Parameter | Value |
|---|---|
| Model | ResNet18 |
| Framework | PyTorch |
| Image Size | 64 × 64 |
| Number of Classes | 10 |
| Train/Test Split | 80/20 |
| Batch Size | 32 |
| Epochs | 8 |
| Learning Rate | 0.001 |
| Device | CPU / CUDA when available |

---

# 🚀 Getting Started

## 1. Clone the Repository

Open Command Prompt or Terminal and run:

```bash
git clone https://github.com/siri-orog/land-cover-classification.git
````

Move into the project directory:

```bash
cd land-cover-classification
```

---

## 2. Create a Virtual Environment

A virtual environment keeps the project's Python dependencies separate from other projects.

### Windows

```bash
python -m venv land_cover_env
```

Activate it:

```bash
land_cover_env\Scripts\activate
```

You should see something similar to:

```text
(land_cover_env) D:\land-cover-classification>
```

### Linux / macOS

```bash
python3 -m venv land_cover_env
```

Activate:

```bash
source land_cover_env/bin/activate
```

---

## 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# 🛰️ Dataset Setup

The complete EuroSAT dataset is **not included in this repository** because of its size.

After obtaining the EuroSAT RGB dataset, place it inside:

```text
data/eurosat/
```

The expected folder structure is:

```text
data/
└── eurosat/
    ├── AnnualCrop/
    ├── Forest/
    ├── HerbaceousVegetation/
    ├── Highway/
    ├── Industrial/
    ├── Pasture/
    ├── PermanentCrop/
    ├── Residential/
    ├── River/
    └── SeaLake/
```

Each class folder should contain its corresponding satellite images.

---

# ▶️ How to Run

Run the scripts in the following order.

## Step 1 — Explore the Dataset

```bash
python 01_explore_data.py
```

This script explores the dataset and displays sample satellite images.

A sample image grid is saved as:

```text
outputs/sample_grid.png
```

---

## Step 2 — Train the Model

```bash
python 02_train.py
```

This trains the ResNet18 model using the EuroSAT dataset.

Training settings can be changed in:

```text
config.py
```

Example:

```python
TRAIN_SPLIT = 0.8
BATCH_SIZE = 32
IMAGE_SIZE = 64
NUM_EPOCHS = 8
LEARNING_RATE = 1e-3
```

After training, the model is saved as:

```text
outputs/landcover_resnet18.pt
```

---

## Step 3 — Evaluate the Model

After training is completed:

```bash
python 03_evaluate.py
```

This evaluates the trained model on the test dataset.

The confusion matrix is generated at:

```text
outputs/confusion_matrix.png
```

The confusion matrix shows how the model classifies the different land-cover categories and which classes are commonly confused.

---

## Step 4 — Predict a Single Image

```bash
python 04_predict_single_image.py
```

This loads the trained ResNet18 model and predicts the land-cover category of a single satellite image.

Example:

```text
Predicted Class: Forest
```

---

# 🔄 Complete Run Commands

After cloning the project, the basic workflow is:

```bash
git clone https://github.com/siri-orog/land-cover-classification.git

cd land-cover-classification

python -m venv land_cover_env

land_cover_env\Scripts\activate

pip install -r requirements.txt

python 01_explore_data.py

python 02_train.py

python 03_evaluate.py

python 04_predict_single_image.py
```

> Make sure the EuroSAT dataset has been placed inside `data/eurosat/` before running the Python scripts.

---

# ⚙️ Configuration

All major project settings are centralized in:

```text
config.py
```

Important settings include:

```python
TRAIN_SPLIT = 0.8
BATCH_SIZE = 32
IMAGE_SIZE = 64
NUM_WORKERS = 0

NUM_EPOCHS = 8
LEARNING_RATE = 1e-3
SUBSET_SIZE = None
```

### Quick Experiment

For a smaller training experiment, you can set:

```python
SUBSET_SIZE = 5000
```

For the complete dataset:

```python
SUBSET_SIZE = None
```

---

# 📁 Project Structure

```text
land-cover-classification/
│
├── 01_explore_data.py
├── 02_train.py
├── 03_evaluate.py
├── 04_predict_single_image.py
├── config.py
├── dataset.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── sample_grid.png
│   └── landcover_resnet18.pt
│
└── data/
    └── eurosat/
        ├── AnnualCrop/
        ├── Forest/
        ├── HerbaceousVegetation/
        ├── Highway/
        ├── Industrial/
        ├── Pasture/
        ├── PermanentCrop/
        ├── Residential/
        ├── River/
        └── SeaLake/
```

---

# 📄 File Description

| File                         | Purpose                                      |
| ---------------------------- | -------------------------------------------- |
| `01_explore_data.py`         | Dataset exploration and sample visualization |
| `02_train.py`                | ResNet18 model training                      |
| `03_evaluate.py`             | Model evaluation and confusion matrix        |
| `04_predict_single_image.py` | Single-image prediction                      |
| `dataset.py`                 | Dataset loading and preprocessing            |
| `config.py`                  | Central project configuration                |
| `requirements.txt`           | Python dependencies                          |
| `outputs/`                   | Model and generated results                  |

---

# 📊 Outputs

### Sample Image Grid

```text
outputs/sample_grid.png
```

Displays sample satellite images from the dataset.

### Confusion Matrix

```text
outputs/confusion_matrix.png
```

Visualizes classification results across the 10 land-cover categories.

### Trained Model

```text
outputs/landcover_resnet18.pt
```

Saved PyTorch model used for inference.

---

# 🧩 Project Workflow

```text
                 EuroSAT Dataset
                        │
                        ▼
                Dataset Exploration
                        │
                        ▼
                Image Preprocessing
                        │
                        ▼
                  Train/Test Split
                        │
                        ▼
                    ResNet18
                        │
                        ▼
                  Model Training
                        │
                        ▼
                  Trained Model
                        │
                 ┌──────┴──────┐
                 ▼             ▼
             Evaluation     Prediction
                 │             │
                 ▼             ▼
         Confusion Matrix   Single Image
                               │
                               ▼
                       Land-Cover Class
```

---

# 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **Torchvision**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **Pillow**
* **ResNet18**
* **EuroSAT Dataset**

---

# 💡 How to Develop / Extend the Project

The current project can be further developed into a complete satellite-image analysis application.

## 1. Add a Streamlit Web Application

Create an `app.py` file that allows users to:

* Upload a satellite image
* Preview the uploaded image
* Run the trained model
* Display the predicted land-cover class
* Display prediction confidence
* Show information about the predicted class

Run the application using:

```bash
streamlit run app.py
```

---

## 2. Add Data Augmentation

The training pipeline can be improved using techniques such as:

* Random horizontal flip
* Random vertical flip
* Random rotation
* Random crop
* Color adjustments

This can help the model generalize better to different satellite images.

---

## 3. Compare Multiple Deep Learning Models

The ResNet18 model can be compared with other architectures such as:

* ResNet50
* VGG16
* MobileNetV2
* EfficientNet
* DenseNet

Performance can be compared using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 4. Add Training Graphs

The project can be extended to save:

```text
Training Accuracy
Validation Accuracy
Training Loss
Validation Loss
```

These graphs help visualize model learning over different epochs.

---

## 5. Add Prediction Confidence

Instead of showing only:

```text
Predicted Class: Forest
```

the application can display:

```text
Predicted Class: Forest
Confidence: 94.7%
```

This makes the prediction interface more informative.

---

## 6. Deploy the Application

After adding a Streamlit interface, the project can be deployed as a web application.

Possible platforms include:

* Streamlit Community Cloud
* Hugging Face Spaces
* Cloud hosting platforms

---

# 🌍 Real-World Applications

Land-cover classification can be useful for:

* 🌲 Forest monitoring
* 🌾 Agricultural analysis
* 🏙️ Urban development monitoring
* 🌊 Water-body identification
* 🌱 Environmental monitoring
* 🛰️ Remote sensing
* 🗺️ Geographic Information Systems (GIS)
* 🏞️ Land-use analysis
* Satellite image analysis

---

# 🔮 Future Enhancements

* [ ] Add Streamlit web interface
* [ ] Add satellite image upload
* [ ] Add prediction confidence
* [ ] Add training/validation accuracy graphs
* [ ] Add training/validation loss graphs
* [ ] Add precision, recall and F1-score
* [ ] Compare multiple CNN architectures
* [ ] Add data augmentation
* [ ] Add hyperparameter tuning
* [ ] Optimize the trained model
* [ ] Deploy the application
* [ ] Add interactive satellite image analysis

---

# 👩‍💻 Author

## Siri Lasya Reddy

**B.Tech – Artificial Intelligence & Machine Learning**

GitHub:
[https://github.com/siri-orog](https://github.com/siri-orog)

---

# ⭐ Project

If you find this project useful, feel free to explore the repository and extend it with additional deep-learning and remote-sensing features.

````

In your `D:\landcover` CMD:

```cmd
git add README.md
git commit -m "Improve README documentation"
git push
````

Then refresh your GitHub repository. The **single README** above will contain the clone instructions, installation, dataset setup, complete run commands, project structure, and future development steps.
