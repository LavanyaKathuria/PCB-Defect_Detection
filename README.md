# PCB Defect Detection using Deep Learning and Explainable AI

Automated detection and classification of defects in Printed Circuit Boards (PCBs) using deep learning and explainable AI techniques. The system supports both an end-to-end CNN classifier and a hybrid CNN + XGBoost pipeline, with visual explanations generated using Grad-CAM.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-orange.svg)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-yellow.svg)]()

---

## Table of Contents

- [Abstract](#abstract)
- [Problem Statement and Motivation](#problem-statement-and-motivation)
- [Dataset Description](#dataset-description)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Model Architecture](#model-architecture)
- [Hybrid CNN + XGBoost Approach](#hybrid-cnn--xgboost-approach)
- [Training Strategy](#training-strategy)
- [Explainability with Grad-CAM](#explainability-with-grad-cam)
- [Experimental Observations](#experimental-observations)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Challenges Faced](#challenges-faced)
- [Limitations and Future Work](#limitations-and-future-work)
---

## Abstract

This project presents a deep learning–based approach for automated detection and classification of defects in Printed Circuit Boards (PCBs). The system is designed for industrial visual inspection scenarios, where both classification accuracy and model interpretability are critical. A convolutional neural network based on ResNet-18 is employed for visual feature extraction, and two classification strategies are explored: an end-to-end CNN classifier and a hybrid CNN + XGBoost pipeline. Model decisions are analyzed using Gradient-weighted Class Activation Mapping (Grad-CAM) to ensure explainability.

---

## Problem Statement and Motivation

Printed Circuit Boards are a fundamental component of modern electronic systems. Manufacturing defects such as open circuits, shorts, cracks, and missing holes can severely impact reliability and functionality. Traditional manual inspection methods are time-consuming, subjective, and unsuitable for high-throughput production environments.

The objective of this project is to develop an automated PCB defect detection system that:

- Accurately classifies multiple defect types
- Operates on localized defect regions
- Provides visual explanations for its predictions
- Is suitable for deployment-oriented industrial workflows

---

## Dataset Description

- **Dataset**: DeepPCB
- **Type**: Patch-based PCB defect dataset
- **Annotations**: Bounding boxes with defect class labels

### Defect Classes

1. Open
2. Short
3. Crack
4. Missing Hole
5. Spur
6. Background

**Note**: The DeepPCB dataset is not included in this repository due to licensing and size constraints.

Each PCB image is accompanied by annotation files specifying defect bounding boxes. These bounding boxes are used to extract localized defect patches for training and evaluation.

---

## Data Processing Pipeline

1. Full PCB images are read from the dataset
2. Defect bounding boxes are parsed from annotation files
3. Defect-level patches are cropped from the full image
4. Patches are resized to **224 × 224** pixels
5. Data augmentation is applied during training:
   - Horizontal and vertical flips
   - Random rotations
   - Color jitter
6. Input normalization is performed using ImageNet statistics
7. A custom PyTorch Dataset class is implemented to manage this pipeline efficiently

---

## Model Architecture

### Base CNN Model

- **Backbone**: ResNet-18
- **Framework**: PyTorch
- **Input size**: 224 × 224 RGB images
- **Output**: 6-class probability distribution

The final fully connected layer of ResNet-18 is adapted to match the number of PCB defect classes.

---

## Hybrid CNN + XGBoost Approach

In addition to the end-to-end CNN classifier, a hybrid classification strategy is explored.

### Architecture

1. ResNet-18 is used as a fixed feature extractor
2. The final classification layer is removed
3. Each defect patch is mapped to a **512-dimensional** feature vector
4. These feature vectors are used to train an XGBoost classifier

### Motivation for Hybrid Design

- CNNs excel at learning spatial and semantic representations
- XGBoost provides strong decision boundaries on structured feature vectors
- The hybrid approach can improve robustness, especially under limited data and class imbalance

The hybrid model is evaluated as an alternative to the pure CNN approach.

---

## Training Strategy

- **Loss function**: Cross-entropy loss
- **Optimizer**: Adam
- **Learning rate**: 1e-4
- **Batch size**: Patch-based mini-batches
- **Hardware**: GPU acceleration (CUDA supported)

Training is performed on defect-level patches rather than full PCB images, allowing the model to focus on localized defect patterns.

---

## Explainability with Grad-CAM

To ensure transparency and trustworthiness of the model, Grad-CAM is employed for visual explanation.

### Purpose

- Identify image regions contributing most to a prediction
- Validate whether the model attends to true defect regions
- Support debugging and industrial acceptance of deep learning models

### Implementation

- Grad-CAM is computed on the final convolutional layer of ResNet-18
- Heatmaps are overlaid on defect patches
- For the hybrid pipeline, Grad-CAM explanations are generated from the CNN feature extractor, while final predictions may come from XGBoost

### Generated Visualizations Include:

- Full PCB image with defect bounding box
- Cropped defect patch
- Grad-CAM heatmap overlay

---

## Experimental Observations

- The CNN model successfully learns discriminative features for multiple PCB defect classes
- The hybrid CNN + XGBoost approach demonstrates comparable and, in some cases, more stable performance
- Grad-CAM visualizations confirm that the model focuses on defect-relevant regions rather than background artifacts
- Quantitative metrics can be extended in future work

---

## Project Structure
```
pcb-defect-detection/
│
├── data/                      # Dataset (ignored in git)
├── models/
│   ├── best_pcb_model.pth     # Trained CNN weights
│   ├── xgboost_classifier.pkl
│   └── feature_scaler.pkl
│
├── notebooks/                 # Training and visualization notebooks
├── results/
│   ├── plots/                 # Training curves and analysis
│   └── gradcam_outputs/       # Grad-CAM visualizations
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Technologies Used

- Python
- PyTorch
- Torchvision
- XGBoost
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- PIL
- pytorch-grad-cam
- CUDA (GPU acceleration)
- Git

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/pcb-defect-detection.git
cd pcb-defect-detection
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Training CNN Model

Run training notebook:
```
notebooks/train_cnn.ipynb
```

### Train Hybrid XGBoost Model
```
notebooks/train_xgboost.ipynb
```

### Generate Grad-CAM Visualizations
```
notebooks/gradcam_visualization.ipynb
```

Ensure dataset paths are configured correctly before execution.

---

## Challenges Faced

### Class Imbalance
- The background class and certain rare defect types (e.g., missing hole) were underrepresented in the dataset
- Addressed through data augmentation and weighted loss functions

### Feature Extraction vs End-to-End Learning
- Balancing the trade-off between CNN feature extraction and XGBoost classification required careful experimentation
- Hybrid approach provided more stable results in some scenarios but increased model complexity

### Computational Constraints
- Training deep models on high-resolution PCB images required GPU acceleration
- Patch-based training strategy was adopted to reduce memory overhead

### Model Interpretability
- Ensuring that Grad-CAM heatmaps meaningfully highlighted defect regions required tuning of visualization parameters
- Validation of attention maps against ground truth defect locations was essential for trust in model predictions

---

## Limitations and Future Work

- Extension to full-image object detection (e.g., YOLO, Faster R-CNN)
- Improved handling of background and ambiguous defect classes
- Quantitative benchmarking against industrial baselines
- Real-time deployment on embedded or edge systems

---

