# TumorVision – Breast Cancer Classification Using PCA and Machine Learning

> **Disclaimer:** This application is developed strictly for educational and academic purposes. It is not a medical diagnostic tool and must not be used to make medical decisions.

## Project Overview
**TumorVision** is an educational machine learning mini-project that classifies breast tumour samples as **Malignant (`1`)** or **Benign (`0`)** using 30 cell-nucleus measurements, and studies the impact of **Principal Component Analysis (PCA)** dimensionality reduction across four algorithms:
- **Logistic Regression** (Without PCA & With PCA)
- **K-Nearest Neighbours (KNN)** (Without PCA & With PCA)
- **Decision Tree** (Without PCA & With PCA)
- **Random Forest** (Without PCA & With PCA)

## Folder Structure
```text
TumorVision/
│
├── dataset/
│   └── breast_cancer.csv
│
├── notebooks/
│   └── main.ipynb
│
├── pickle/
│   ├── logistic_regression.pkl
│   ├── knn.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── logistic_regression_pca.pkl
│   ├── knn_pca.pkl
│   ├── decision_tree_pca.pkl
│   ├── random_forest_pca.pkl
│   ├── scaler.pkl
│   ├── pca.pkl
│   ├── model_metadata.pkl
│   └── readable/             # Human-readable JSON exports (generated)
│
├── report/
│   ├── figures/
│   └── Project_Report.md
│
├── app.py
├── train.py
├── inspect_pickles.py
├── requirements.txt
└── README.md
```

## Installation & Execution

1. **Navigate to the project folder:**
   ```bash
   cd TumorVision
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train all 8 models and generate pickle files & evaluation plots:**
   ```bash
   python train.py
   ```

4. **Launch the Streamlit web application:**
   ```bash
   streamlit run app.py
   ```

5. **(Optional) Open the Jupyter Notebook:**
   ```bash
   jupyter notebook notebooks/main.ipynb
   ```

## Viewing Saved Model Files

The `.pkl` files are binary serialized models and are not intended to be opened as
text. Keep them unchanged for the application. To export readable JSON descriptions
of the models, preprocessing objects, and metadata, run:

```bash
python3 inspect_pickles.py
```

Open the generated files in `pickle/readable/` with any text editor. The exports
include model settings and learned values; the metadata export contains the full
training and evaluation metadata. Only run this script on trusted pickle files:
loading pickle files from untrusted sources can execute code.

## Key Results Summary
- **PCA Dimensionality Reduction:** 10 principal components retain **95.21%** of the variance from the original 30 features.
- **Top Models:** **Logistic Regression (PCA)** and **Random Forest (No PCA)** both achieved **97.37% accuracy**, **100.00% precision**, **92.86% malignant-class recall**, and **96.30% F1-score** on the stratified 20% test set.
