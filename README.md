# PCB Defect Detection using Deep Learning and Explainable AI

## Abstract

This project presents a deep learning–based approach for automated detection and classification of defects in Printed Circuit Boards (PCBs). The system is designed for industrial visual inspection scenarios, where both classification accuracy and model interpretability are critical. A convolutional neural network based on ResNet-18 is employed for visual feature extraction, and two classification strategies are explored: an end-to-end CNN classifier and a hybrid CNN + XGBoost pipeline. Model decisions are analyzed using Gradient-weighted Class Activation Mapping (Grad-CAM) to ensure explainability.

---

## 1. Problem Statement and Motivation

Printed Circuit Boards are a fundamental component of modern electronic systems. Manufacturing defects such as open circuits, shorts, cracks, and missing holes can severely impact reliability and functionality. Traditional manual inspection methods are time-consuming, subjective, and unsuitable for high-throughput production environments.

The objective of this project is to develop an automated PCB defect detection system that:
- Accurately classifies multiple defect types
- Operates on localized defect regions
- Provides visual explanations for its predictions
- Is suitable for deployment-oriented industrial workflows

---

## 2. Dataset Description

- **Dataset:** DeepPCB
- **Type:** Patch-based PCB defect dataset
- **Annotations:** Bounding boxes with defect class labels
- **Classes:**  
  - Open  
  - Short  
  - Crack  
  - Missing Hole  
  - Spur  
  - Background  

> **Note:** The DeepPCB dataset is not included in this repository due to licensing and size constraints.

Each PCB image is accompanied by annotation files specifying defect bounding boxes. These bounding boxes are used to extract localized defect patches for training and evaluation.

---

## 3. Data Processing Pipeline

1. Full PCB images are read from the dataset
2. Defect bounding boxes are parsed from annotation files
3. Defect-level patches are cropped from the full image
4. Patches are resized to 224 × 224 pixels
5. Data augmentation is applied during training:
   - Horizontal and vertical flips
   - Random rotations
   - Color jitter
6. Input normalization is performed using ImageNet statistics

A custom PyTorch `Dataset` class is implemented to manage this pipeline efficiently.

---

## 4. Model Architecture

### 4.1 Base CNN Model

- **Backbone:** ResNet-18
- **Framework:** PyTorch
- **Input size:** 224 × 224 RGB images
- **Output:** 6-class probability distribution

The final fully connected layer of ResNet-18 is adapted to match the number of PCB defect classes.

---

## 5. Hybrid CNN + XGBoost Approach

In addition to the end-to-end CNN classifier, a hybrid classification strategy is explored.

### 5.1 Architecture

- ResNet-18 is used as a **fixed feature extractor**
- The final classification layer is removed
- Each defect patch is mapped to a **512-dimensional feature vector**
- These feature vectors are used to train an **XGBoost classifier**


### 5.2 Motivation for Hybrid Design

- CNNs excel at learning spatial and semantic representations
- XGBoost provides strong decision boundaries on structured feature vectors
- The hybrid approach can improve robustness, especially under limited data and class imbalance

The hybrid model is evaluated as an alternative to the pure CNN approach.

---

## 6. Training Strategy

- **Loss function:** Cross-entropy loss
- **Optimizer:** Adam
- **Learning rate:** 1e-4
- **Batch size:** Patch-based mini-batches
- **Hardware:** GPU acceleration (CUDA supported)

Training is performed on defect-level patches rather than full PCB images, allowing the model to focus on localized defect patterns.

---

## 7. Explainability with Grad-CAM

To ensure transparency and trustworthiness of the model, **Grad-CAM** is employed for visual explanation.

### 7.1 Purpose

- Identify image regions contributing most to a prediction
- Validate whether the model attends to true defect regions
- Support debugging and industrial acceptance of deep learning models

### 7.2 Implementation

- Grad-CAM is computed on the final convolutional layer of ResNet-18
- Heatmaps are overlaid on defect patches
- For the hybrid pipeline, Grad-CAM explanations are generated from the CNN feature extractor, while final predictions may come from XGBoost

Generated visualizations include:
1. Full PCB image with defect bounding box
2. Cropped defect patch
3. Grad-CAM heatmap overlay

---

## 8. Experimental Observations

- The CNN model successfully learns discriminative features for multiple PCB defect classes
- The hybrid CNN + XGBoost approach demonstrates comparable and, in some cases, more stable performance
- Grad-CAM visualizations confirm that the model focuses on defect-relevant regions rather than background artifacts

Quantitative metrics can be extended in future work.

---

## 9. Project Structure

pcb-defect-detection/
│
├── data/ # Dataset (ignored in git)
├── models/
│ ├── best_pcb_model.pth # Trained CNN weights
│ ├── xgboost_classifier.pkl
│ └── feature_scaler.pkl
│
├── notebooks/ # Training and visualization notebooks
├── results/
│ ├── plots/ # Training curves and analysis
│ └── gradcam_outputs/ # Grad-CAM visualizations
│
├── README.md
├── requirements.txt
└── .gitignore

---

## 10. Technologies Used

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

---

## 11. Limitations and Future Work

- Extension to full-image object detection (e.g., YOLO, Faster R-CNN)
- Improved handling of background and ambiguous defect classes
- Quantitative benchmarking against industrial baselines
- Real-time deployment on embedded or edge systems

---

## 12. Author

**Tushar**  
Electronics Engineering  
Focus Areas: Deep Learning, Computer Vision, Explainable AI  

---

## 13. Intended Use

This project is intended for **academic, research, and experimental purposes** and serves as a foundation for further work in automated visual inspection systems.
