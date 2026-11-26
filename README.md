# TumorVision 🧬
### Breast Cancer Classification Using Principal Component Analysis (PCA)

## 📌 About the Project

TumorVision is a Machine Learning project developed to classify breast tumours as **Malignant (Cancerous)** or **Benign (Non-Cancerous)** using different classification algorithms.

The main focus of this project is to understand and implement **Principal Component Analysis (PCA)**, a dimensionality reduction technique, and study how it affects the performance of machine learning models.

The project compares classification models with and without PCA to understand the impact of reducing the number of features on prediction performance.

## 🎯 Objectives

- Build a machine learning model to classify breast tumours.
- Understand and implement Principal Component Analysis (PCA).
- Reduce the dimensionality of the dataset while retaining important information.
- Compare machine learning models with and without PCA.
- Evaluate models using different performance metrics.
- Develop an interactive web application using Streamlit.

## 🧠 Machine Learning Algorithms

The project explores the following classification algorithms:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest

Each algorithm is evaluated with and without PCA to understand the effect of dimensionality reduction.

## 🔬 What is PCA?

Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique used to reduce the number of features in a dataset while preserving as much important information as possible.

In this project, PCA helps transform the original dataset into a smaller set of principal components, making it possible to study how reduced features affect classification performance.

## 📊 Model Evaluation

The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics help compare model performance and understand how effectively the models identify malignant and benign tumours.

## 🛠️ Technologies Used

- **Python** – Main programming language
- **Pandas** – Data manipulation
- **NumPy** – Numerical operations
- **Scikit-learn** – Machine learning and PCA
- **Matplotlib & Seaborn** – Data visualization
- **Streamlit** – Interactive web application
- **Pickle** – Saving and loading trained models
- **Jupyter Notebook** – Data analysis and experimentation

## 📁 Project Structure

```text
TumorVision/
│
├── dataset/                 # Dataset files
│
├── notebooks/
│   └── main.ipynb           # Data analysis and experiments
│
├── pickle/                  # Saved trained models
│
├── app.py                   # Streamlit application
├── train.py                 # Model training script
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
│
└── report/                  # Project report
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-repository-link>
```

### 2. Navigate to the Project Folder

```bash
cd TumorVision
```

### 3. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 4. Activate the Virtual Environment

For macOS/Linux:

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Train the Models

```bash
python train.py
```

### 7. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 🌐 Application Features

- **Dataset Overview:** Explore the breast cancer dataset and its features.
- **PCA Visualization:** Understand dimensionality reduction and principal components.
- **Model Comparison:** Compare classification algorithms with and without PCA.
- **Model Evaluation:** View performance metrics and confusion matrices.
- **Tumour Prediction:** Explore model predictions for benign and malignant classifications.

## 💡 Key Learning Outcomes

Through this project, I aim to gain practical experience in:

- Data preprocessing and feature scaling.
- Dimensionality reduction using PCA.
- Implementing different classification algorithms.
- Comparing machine learning models.
- Understanding model evaluation metrics.
- Building an interactive machine learning application using Streamlit.

## ⚠️ Disclaimer

TumorVision is an educational machine learning project developed for academic and learning purposes only. It is not a medically approved diagnostic tool and should not be used as a substitute for professional medical advice or diagnosis.

## 👩‍💻 Author

**Irene Koner**

Machine Learning Project | TumorVision

---

⭐ If you find this project interesting, feel free to explore the repository!
