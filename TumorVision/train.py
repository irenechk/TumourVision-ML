"""
TumorVision – Breast Cancer Classification Using PCA and Machine Learning
Training Script (train.py)

This script:
1. Loads and inspects the dataset dynamically without hardcoding column assumptions.
2. Cleans unnecessary/missing columns and checks for duplicates.
3. Splits data into stratified 80:20 train and test sets.
4. Fits StandardScaler and PCA (retaining >= 95% variance) only on training data.
5. Trains 8 Scikit-learn Pipelines (4 algorithms x 2 configurations: With & Without PCA).
6. Evaluates all 8 models (Accuracy, Precision, Recall, F1-score, Confusion Matrix).
7. Generates comparison charts and heatmaps in report/figures/.
8. Saves all 8 models, scaler, PCA, and metadata into pickle/.
9. Verifies that all saved pickle files load and predict correctly.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
PCA_VARIANCE_THRESHOLD = 0.95

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "breast_cancer.csv")
PICKLE_DIR = os.path.join(BASE_DIR, "pickle")
FIGURES_DIR = os.path.join(BASE_DIR, "report", "figures")


def detect_target_column(df: pd.DataFrame) -> tuple[str, dict]:
    """
    Dynamically detect the target column containing Malignant/Benign labels
    and build a binary mapping where Malignant = 1 (positive class) and Benign = 0.
    """
    for col in df.columns:
        unique_vals = set(df[col].dropna().astype(str).str.strip().unique())
        upper_vals = {v.upper() for v in unique_vals}

        if upper_vals == {"M", "B"}:
            mapping = {}
            for v in unique_vals:
                mapping[v] = 1 if v.strip().upper() == "M" else 0
            return col, mapping

        if upper_vals == {"MALIGNANT", "BENIGN"}:
            mapping = {}
            for v in unique_vals:
                mapping[v] = 1 if v.strip().upper() == "MALIGNANT" else 0
            return col, mapping

    raise ValueError(
        "Could not automatically detect a binary Malignant/Benign target column."
    )


def load_and_clean_dataset(csv_path: str) -> tuple[pd.DataFrame, str, dict, dict]:
    """Load CSV, inspect quality, drop unnecessary columns, and clean missing values."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    raw_shape = df.shape
    print(f"Loaded dataset: {csv_path}")
    print(f"Raw shape: {raw_shape[0]} rows x {raw_shape[1]} columns")

    # Identify completely empty columns (e.g., Unnamed: 32 from trailing commas)
    empty_cols = [c for c in df.columns if df[c].isnull().all()]
    if empty_cols:
        print(f"Dropping completely empty columns: {empty_cols}")
        df = df.drop(columns=empty_cols)

    # Detect target column dynamically
    target_col, target_mapping = detect_target_column(df)
    print(f"Detected target column: '{target_col}' with mapping {target_mapping}")
    print("Positive class: 1 (Malignant) | Negative class: 0 (Benign)")

    # Drop identifier columns (e.g., 'id' with all unique integer values)
    id_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        if col.lower() in {"id", "patient_id", "sample_id"} or (
            pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() == len(df)
        ):
            id_cols.append(col)

    if id_cols:
        print(f"Dropping identifier columns: {id_cols}")
        df = df.drop(columns=id_cols)

    # Check duplicates
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        print(f"Dropping {duplicate_count} duplicate rows.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("Duplicate rows: 0")

    # Drop rows with missing target, if any
    missing_target = int(df[target_col].isnull().sum())
    if missing_target > 0:
        df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # Check feature missing values and impute numeric features with median if needed
    feature_cols = [c for c in df.columns if c != target_col]
    missing_features = df[feature_cols].isnull().sum()
    cols_with_missing = missing_features[missing_features > 0].to_dict()
    if cols_with_missing:
        print(f"Imputing missing feature values with median: {cols_with_missing}")
        for col in cols_with_missing:
            df[col] = df[col].fillna(df[col].median())
    else:
        print("Missing values in cleaned features: 0")

    # Verify all feature columns are numeric
    non_numeric = [
        c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric}")

    class_counts = df[target_col].value_counts().to_dict()
    summary = {
        "raw_rows": int(raw_shape[0]),
        "raw_columns": int(raw_shape[1]),
        "cleaned_rows": int(df.shape[0]),
        "num_features": len(feature_cols),
        "dropped_empty_columns": empty_cols,
        "dropped_id_columns": id_cols,
        "duplicate_rows_removed": duplicate_count,
        "missing_values_after_cleaning": int(df.isnull().sum().sum()),
        "class_counts_raw": {str(k): int(v) for k, v in class_counts.items()},
    }

    return df, target_col, target_mapping, summary


def save_plots(
    y: pd.Series,
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    pca: PCA,
    results: list[dict],
) -> None:
    """Generate and save all required EDA, PCA, and model evaluation plots."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Class Distribution Chart
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts = y.value_counts().sort_index()
    labels = ["Benign (0)", "Malignant (1)"]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(labels, [counts[0], counts[1]], color=colors, width=0.5, edgecolor="#333")
    ax.set_title("Class Distribution (Benign vs Malignant)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Samples")
    ax.set_ylim(0, max(counts) * 1.15)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 6,
            f"{int(height)} ({height / len(y) * 100:.1f}%)",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"), dpi=150)
    plt.close(fig)

    # 2. Cumulative Explained Variance Plot
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_comp = len(cum_var)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x_ticks = np.arange(1, n_comp + 1)
    ax.bar(
        x_ticks,
        pca.explained_variance_ratio_ * 100,
        alpha=0.55,
        color="#3498db",
        label="Individual Explained Variance (%)",
    )
    ax.plot(
        x_ticks,
        cum_var * 100,
        marker="o",
        color="#e74c3c",
        linewidth=2,
        label="Cumulative Explained Variance (%)",
    )
    ax.axhline(95, color="#2c3e50", linestyle="--", linewidth=1.2, label="95% Threshold")
    ax.set_title(
        f"PCA Explained Variance ({n_comp} Components Retain {cum_var[-1]*100:.2f}% Variance)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_xticks(x_ticks)
    ax.set_ylim(0, 105)
    ax.legend(loc="center right")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cumulative_explained_variance.png"), dpi=150)
    plt.close(fig)

    # 3. 2D PCA Scatter Plot (PC1 vs PC2) on training data
    X_train_pca = pca.transform(X_train_scaled)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    benign_mask = y_train.values == 0
    malignant_mask = y_train.values == 1
    ax.scatter(
        X_train_pca[benign_mask, 0],
        X_train_pca[benign_mask, 1],
        c="#2ecc71",
        label="Benign (0)",
        alpha=0.75,
        edgecolors="k",
        linewidths=0.4,
        s=45,
    )
    ax.scatter(
        X_train_pca[malignant_mask, 0],
        X_train_pca[malignant_mask, 1],
        c="#e74c3c",
        label="Malignant (1)",
        alpha=0.75,
        edgecolors="k",
        linewidths=0.4,
        s=45,
    )
    ax.set_title("2D PCA Projection (PC1 vs PC2)", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pca_2d_scatter.png"), dpi=150)
    plt.close(fig)

    # 4. Confusion Matrices Heatmap Grid (2x4)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for idx, res in enumerate(results):
        cm = np.array(res["confusion_matrix"])
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(res["model_name"], fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign (0)", "Malig (1)"], fontsize=8)
        ax.set_yticklabels(["Benign (0)", "Malig (1)"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11,
                    fontweight="bold",
                )
    plt.suptitle("Confusion Matrices for All 8 Models (Test Set)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrices.png"), dpi=150)
    plt.close(fig)

    # 5. Model Accuracy Comparison Bar Chart
    names = [r["model_name"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]
    bar_colors = ["#3498db" if not r["uses_pca"] else "#9b59b6" for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names[::-1], accs[::-1], color=bar_colors[::-1], edgecolor="#333")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison (Without PCA vs With PCA)", fontsize=13, fontweight="bold")
    ax.set_xlim(80, 102)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "accuracy_comparison.png"), dpi=150)
    plt.close(fig)

    # 6. Malignant-Class Recall Comparison Bar Chart
    recalls = [r["recall"] * 100 for r in results]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names[::-1], recalls[::-1], color=bar_colors[::-1], edgecolor="#333")
    ax.set_xlabel("Malignant-Class Recall (%)")
    ax.set_title("Malignant-Class Recall Comparison (Minimizing False Negatives)", fontsize=13, fontweight="bold")
    ax.set_xlim(80, 102)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "malignant_recall_comparison.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(PICKLE_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Load & Clean Dataset
    df, target_col, target_mapping, summary = load_and_clean_dataset(DATASET_PATH)

    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].copy()
    y = df[target_col].map(target_mapping).astype(int)

    # 2. Stratified Train-Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"\nTrain set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 3. Fit standalone StandardScaler and PCA ONLY on training data (for inspection & saving)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, svd_solver="full", random_state=RANDOM_STATE)
    pca.fit(X_train_scaled)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nPCA Components to retain >= 95% variance: {pca.n_components_} (actual: {cum_var[-1]*100:.2f}%)")
    for i, (ev, cv) in enumerate(zip(pca.explained_variance_ratio_, cum_var), start=1):
        print(f"  PC{i:02d}: {ev*100:6.2f}% | Cumulative: {cv*100:6.2f}%")

    # 4. Define the 4 base algorithms and 2 configurations (8 models total)
    model_specs = [
        ("Logistic Regression", "logistic_regression", lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("K-Nearest Neighbours", "knn", lambda: KNeighborsClassifier(n_neighbors=5)),
        ("Decision Tree", "decision_tree", lambda: DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ("Random Forest", "random_forest", lambda: RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ]

    results = []
    saved_model_files = {}

    for display_name, file_stem, estimator_fn in model_specs:
        # Configuration A: Without PCA
        pipe_no_pca = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", estimator_fn()),
        ])
        pipe_no_pca.fit(X_train, y_train)
        y_pred = pipe_no_pca.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        model_key = f"{display_name} (No PCA)"
        pkl_name = f"{file_stem}.pkl"
        with open(os.path.join(PICKLE_DIR, pkl_name), "wb") as f:
            pickle.dump(pipe_no_pca, f)
        saved_model_files[model_key] = pkl_name

        results.append({
            "model_name": model_key,
            "algorithm": display_name,
            "uses_pca": False,
            "pickle_file": pkl_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, pos_label=1)),
            "recall": float(recall_score(y_test, y_pred, pos_label=1)),
            "f1_score": float(f1_score(y_test, y_pred, pos_label=1)),
            "confusion_matrix": cm.tolist(),
            "false_negatives": int(cm[1, 0]),
            "false_positives": int(cm[0, 1]),
        })

        # Configuration B: With PCA
        pipe_pca = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=PCA_VARIANCE_THRESHOLD, svd_solver="full", random_state=RANDOM_STATE)),
            ("classifier", estimator_fn()),
        ])
        pipe_pca.fit(X_train, y_train)
        y_pred_pca = pipe_pca.predict(X_test)
        cm_pca = confusion_matrix(y_test, y_pred_pca)

        model_key_pca = f"{display_name} (PCA)"
        pkl_name_pca = f"{file_stem}_pca.pkl"
        with open(os.path.join(PICKLE_DIR, pkl_name_pca), "wb") as f:
            pickle.dump(pipe_pca, f)
        saved_model_files[model_key_pca] = pkl_name_pca

        results.append({
            "model_name": model_key_pca,
            "algorithm": display_name,
            "uses_pca": True,
            "pickle_file": pkl_name_pca,
            "accuracy": float(accuracy_score(y_test, y_pred_pca)),
            "precision": float(precision_score(y_test, y_pred_pca, pos_label=1)),
            "recall": float(recall_score(y_test, y_pred_pca, pos_label=1)),
            "f1_score": float(f1_score(y_test, y_pred_pca, pos_label=1)),
            "confusion_matrix": cm_pca.tolist(),
            "false_negatives": int(cm_pca[1, 0]),
            "false_positives": int(cm_pca[0, 1]),
        })

    # Identify best model prioritizing Malignant Recall, then F1-score, then Accuracy
    best_model = max(
        results,
        key=lambda r: (round(r["recall"], 6), round(r["f1_score"], 6), round(r["accuracy"], 6)),
    )

    print("\n=== Model Evaluation Comparison (Test Set) ===")
    results_df = pd.DataFrame([
        {
            "Model": r["model_name"],
            "Accuracy": f"{r['accuracy']*100:.2f}%",
            "Precision": f"{r['precision']*100:.2f}%",
            "Malignant Recall": f"{r['recall']*100:.2f}%",
            "F1-Score": f"{r['f1_score']*100:.2f}%",
            "FN": r["false_negatives"],
            "FP": r["false_positives"],
        }
        for r in results
    ])
    print(results_df.to_string(index=False))
    print(
        f"\nBest Performing Model (prioritizing Malignant Recall): {best_model['model_name']} "
        f"(Recall={best_model['recall']*100:.2f}%, Accuracy={best_model['accuracy']*100:.2f}%, "
        f"F1={best_model['f1_score']*100:.2f}%)"
    )

    # Save standalone scaler and pca
    with open(os.path.join(PICKLE_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(PICKLE_DIR, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)

    # Feature statistics from training set for Streamlit input validation & defaults
    feature_stats = {}
    for col in feature_names:
        feature_stats[col] = {
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "mean": float(X_train[col].mean()),
            "median": float(X_train[col].median()),
            "std": float(X_train[col].std()),
        }

    metadata = {
        "project_name": "TumorVision",
        "target_column": target_col,
        "target_mapping": target_mapping,
        "reverse_target_mapping": {0: "Benign", 1: "Malignant"},
        "positive_class": 1,
        "positive_class_label": "Malignant",
        "feature_names": feature_names,
        "feature_stats": feature_stats,
        "dataset_summary": summary,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "pca_info": {
            "variance_threshold": PCA_VARIANCE_THRESHOLD,
            "n_components": int(pca.n_components_),
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "cumulative_variance_ratio": [float(v) for v in cum_var],
        },
        "evaluation_results": results,
        "saved_model_files": saved_model_files,
        "best_model_name": best_model["model_name"],
    }

    with open(os.path.join(PICKLE_DIR, "model_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    # Generate plots
    save_plots(y, X_train_scaled, y_train, pca, results)
    print(f"\nSaved plots to: {FIGURES_DIR}")

    # Verify loading saved pickle files without retraining
    print("\n=== Verifying Saved Pickle Files ===")
    sample_input = X_test.iloc[[0]]
    with open(os.path.join(PICKLE_DIR, "model_metadata.pkl"), "rb") as f:
        loaded_meta = pickle.load(f)
    for m_name, pkl_file in loaded_meta["saved_model_files"].items():
        with open(os.path.join(PICKLE_DIR, pkl_file), "rb") as f:
            loaded_pipe = pickle.load(f)
        pred = loaded_pipe.predict(sample_input[loaded_meta["feature_names"]])[0]
        prob = (
            loaded_pipe.predict_proba(sample_input[loaded_meta["feature_names"]])[0][1]
            if hasattr(loaded_pipe, "predict_proba")
            else None
        )
        print(f"  Verified {pkl_file:30s} -> pred={loaded_meta['reverse_target_mapping'][int(pred)]} (P(Malignant)={prob:.4f})")


if __name__ == "__main__":
    main()
