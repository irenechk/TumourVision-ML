"""
TumorVision – Breast Cancer Classification Using PCA and Machine Learning
Streamlit Web Application (app.py)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "breast_cancer.csv")
PICKLE_DIR = os.path.join(BASE_DIR, "pickle")

DISCLAIMER_TEXT = (
    "This application is developed strictly for educational and academic purposes. "
    "It is not a medical diagnostic tool and must not be used to make medical decisions."
)

st.set_page_config(
    page_title="TumorVision | Breast Cancer ML & PCA",
    page_icon="🔬",
    layout="wide",
)


@st.cache_resource
def load_metadata():
    meta_path = os.path.join(PICKLE_DIR, "model_metadata.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model(pickle_filename: str):
    model_path = os.path.join(PICKLE_DIR, pickle_filename)
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_preprocessing_objects():
    scaler_path = os.path.join(PICKLE_DIR, "scaler.pkl")
    pca_path = os.path.join(PICKLE_DIR, "pca.pkl")
    scaler, pca = None, None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    if os.path.exists(pca_path):
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
    return scaler, pca


@st.cache_data
def load_dataset(target_col: str, target_mapping: dict):
    if not os.path.exists(DATASET_PATH):
        return None
    df = pd.read_csv(DATASET_PATH)
    empty_cols = [c for c in df.columns if df[c].isnull().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
    id_cols = [
        c for c in df.columns
        if c != target_col and (
            c.lower() in {"id", "patient_id", "sample_id"}
            or (pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() == len(df))
        )
    ]
    if id_cols:
        df = df.drop(columns=id_cols)
    df["target_label"] = df[target_col].map(target_mapping).map({0: "Benign", 1: "Malignant"})
    return df


def render_disclaimer():
    st.warning(f"⚠️ **Educational Disclaimer:** {DISCLAIMER_TEXT}")


def main():
    st.title("🔬 TumorVision – Breast Cancer Classification Using PCA & ML")
    render_disclaimer()

    metadata = load_metadata()
    if metadata is None:
        st.error(
            "Saved model metadata (`pickle/model_metadata.pkl`) was not found. "
            "Please run `python train.py` first to train and save all models."
        )
        st.stop()

    scaler, pca = load_preprocessing_objects()
    df = load_dataset(metadata["target_column"], metadata["target_mapping"])

    st.sidebar.header("Navigation")
    section = st.sidebar.radio(
        "Go to Section",
        [
            "1. Home / Dashboard",
            "2. Data Visualisation",
            "3. Model Comparison",
            "4. Prediction",
            "5. About / Limitations",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Best Model:** {metadata['best_model_name']}\n\n"
        f"**PCA Components (≥95% var):** {metadata['pca_info']['n_components']} "
        f"(from {metadata['dataset_summary']['num_features']} features)"
    )

    # ---------------------------------------------------------
    # 1. HOME / DASHBOARD
    # ---------------------------------------------------------
    if section == "1. Home / Dashboard":
        st.header("Home / Dashboard")
        st.markdown(
            """
            **TumorVision** is an educational machine learning classification system that classifies
            breast tumour samples as **Malignant** or **Benign** using measured cell-nucleus features,
            and studies the impact of dimensionality reduction via **Principal Component Analysis (PCA)**.
            """
        )

        summary = metadata["dataset_summary"]
        class_counts = summary["class_counts_raw"]
        # Map raw counts to Benign / Malignant
        benign_count = sum(
            cnt for k, cnt in class_counts.items() if metadata["target_mapping"].get(k) == 0
        )
        malignant_count = sum(
            cnt for k, cnt in class_counts.items() if metadata["target_mapping"].get(k) == 1
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", summary["cleaned_rows"])
        col2.metric("Input Features", summary["num_features"])
        col3.metric("Benign Samples (0)", benign_count)
        col4.metric("Malignant Samples (1)", malignant_count)

        st.subheader("Dataset Overview & Quality Summary")
        st.markdown(
            f"""
            - **Target Column:** `{metadata['target_column']}`
            - **Positive Class Mapping:** `1 = Malignant` (priority class to minimize false negatives), `0 = Benign`
            - **Train / Test Split:** Stratified 80:20 ({metadata['train_samples']} train / {metadata['test_samples']} test samples)
            - **Dropped Unnecessary Columns:** Empty: `{summary['dropped_empty_columns']}`, IDs: `{summary['dropped_id_columns']}`
            - **Duplicate Rows Removed:** {summary['duplicate_rows_removed']}
            - **Missing Values After Cleaning:** {summary['missing_values_after_cleaning']}
            """
        )

        if df is not None:
            st.subheader("Sample Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

    # ---------------------------------------------------------
    # 2. DATA VISUALISATION
    # ---------------------------------------------------------
    elif section == "2. Data Visualisation":
        st.header("Data Visualisation & PCA Exploration")
        if df is None:
            st.error("Dataset file `dataset/breast_cancer.csv` not found.")
            st.stop()

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Class Distribution")
            counts = df["target_label"].value_counts()
            fig, ax = plt.subplots(figsize=(5.5, 4))
            bars = ax.bar(
                ["Benign", "Malignant"],
                [counts.get("Benign", 0), counts.get("Malignant", 0)],
                color=["#2ecc71", "#e74c3c"],
                width=0.5,
                edgecolor="#333",
            )
            ax.set_ylabel("Number of Samples")
            ax.set_title("Benign vs Malignant Samples")
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + 5, str(int(h)), ha="center", fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            st.subheader("PCA Explained Variance")
            evr = np.array(metadata["pca_info"]["explained_variance_ratio"]) * 100
            cum_evr = np.array(metadata["pca_info"]["cumulative_variance_ratio"]) * 100
            x_ticks = np.arange(1, len(evr) + 1)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(x_ticks, evr, alpha=0.6, color="#3498db", label="Individual (%)")
            ax.plot(x_ticks, cum_evr, marker="o", color="#e74c3c", label="Cumulative (%)")
            ax.axhline(95, color="#2c3e50", linestyle="--", label="95% Threshold")
            ax.set_xlabel("Principal Component")
            ax.set_ylabel("Explained Variance (%)")
            ax.set_xticks(x_ticks)
            ax.set_ylim(0, 105)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("PCA 2D Scatter Plot (PC1 vs PC2)")
        if scaler is not None and pca is not None:
            X_all = df[metadata["feature_names"]]
            X_pca_all = pca.transform(scaler.transform(X_all))
            fig, ax = plt.subplots(figsize=(8, 4.8))
            for label, color in [("Benign", "#2ecc71"), ("Malignant", "#e74c3c")]:
                mask = df["target_label"].values == label
                ax.scatter(
                    X_pca_all[mask, 0],
                    X_pca_all[mask, 1],
                    c=color,
                    label=label,
                    alpha=0.75,
                    edgecolors="k",
                    linewidths=0.4,
                    s=40,
                )
            ax.set_xlabel(f"PC1 ({metadata['pca_info']['explained_variance_ratio'][0]*100:.2f}% variance)")
            ax.set_ylabel(f"PC2 ({metadata['pca_info']['explained_variance_ratio'][1]*100:.2f}% variance)")
            ax.set_title("2D PCA Projection of Breast Tumour Features")
            ax.legend()
            ax.grid(alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Interactive Feature Distribution")
        selected_feature = st.selectbox("Select a feature to inspect", metadata["feature_names"])
        fig, ax = plt.subplots(figsize=(8, 4))
        for label, color in [("Benign", "#2ecc71"), ("Malignant", "#e74c3c")]:
            vals = df.loc[df["target_label"] == label, selected_feature]
            ax.hist(vals, bins=25, alpha=0.55, color=color, label=label, edgecolor="#333")
        ax.set_title(f"Distribution of '{selected_feature}' by Class")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------------------------------------
    # 3. MODEL COMPARISON
    # ---------------------------------------------------------
    elif section == "3. Model Comparison":
        st.header("Model Comparison (8 Trained Models)")
        st.markdown(
            """
            Each of the four algorithms (**Logistic Regression**, **KNN**, **Decision Tree**, **Random Forest**)
            is evaluated **Without PCA** (30 scaled features) and **With PCA** (10 principal components retaining ≥95% variance).
            Special importance is given to **Malignant-Class Recall** because missing a malignant tumour (False Negative)
            is the most critical error in this educational classification task.
            """
        )

        results = metadata["evaluation_results"]
        comp_df = pd.DataFrame([
            {
                "Model": r["model_name"],
                "PCA Used": "Yes" if r["uses_pca"] else "No",
                "Accuracy (%)": round(r["accuracy"] * 100, 2),
                "Precision (%)": round(r["precision"] * 100, 2),
                "Malignant Recall (%)": round(r["recall"] * 100, 2),
                "F1-Score (%)": round(r["f1_score"] * 100, 2),
                "False Negatives (FN)": r["false_negatives"],
                "False Positives (FP)": r["false_positives"],
            }
            for r in results
        ])

        st.success(f"🏆 **Best-Performing Model (prioritizing Malignant Recall & F1):** {metadata['best_model_name']}")
        st.dataframe(comp_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            names = [r["model_name"] for r in results][::-1]
            accs = [r["accuracy"] * 100 for r in results][::-1]
            colors = ["#9b59b6" if r["uses_pca"] else "#3498db" for r in results][::-1]
            bars = ax.barh(names, accs, color=colors, edgecolor="#333")
            ax.set_xlim(80, 102)
            ax.set_xlabel("Accuracy (%)")
            for b in bars:
                ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, f"{b.get_width():.2f}%", va="center", fontsize=8)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Malignant-Class Recall Comparison")
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            recalls = [r["recall"] * 100 for r in results][::-1]
            bars = ax.barh(names, recalls, color=colors, edgecolor="#333")
            ax.set_xlim(80, 102)
            ax.set_xlabel("Malignant Recall (%)")
            for b in bars:
                ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, f"{b.get_width():.2f}%", va="center", fontsize=8)
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Confusion Matrices (Test Set)")
        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        axes = axes.flatten()
        for idx, r in enumerate(results):
            cm = np.array(r["confusion_matrix"])
            ax = axes[idx]
            ax.imshow(cm, cmap="Blues")
            ax.set_title(r["model_name"], fontsize=9, fontweight="bold")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Benign", "Malig"], fontsize=8)
            ax.set_yticklabels(["Benign", "Malig"], fontsize=8)
            thresh = cm.max() / 2.0
            for i in range(2):
                for j in range(2):
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontweight="bold",
                    )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------------------------------------
    # 4. PREDICTION
    # ---------------------------------------------------------
    elif section == "4. Prediction":
        st.header("Tumour Classification Prediction")
        st.markdown(
            "Select one of the **8 saved models** and enter cell-nucleus feature measurements below. "
            "Features are validated and passed in the exact training order through the saved preprocessing pipeline."
        )

        model_names = list(metadata["saved_model_files"].keys())
        default_idx = model_names.index(metadata["best_model_name"]) if metadata["best_model_name"] in model_names else 0
        selected_model_name = st.selectbox("Choose Saved Model", model_names, index=default_idx)

        # Optional quick-fill from dataset sample
        preset_values = {}
        if df is not None:
            col_p1, col_p2, col_p3 = st.columns(3)
            if col_p1.button("Load Median Feature Values"):
                st.session_state["preset_mode"] = "median"
            if col_p2.button("Load Sample Benign Row"):
                st.session_state["preset_mode"] = "benign"
            if col_p3.button("Load Sample Malignant Row"):
                st.session_state["preset_mode"] = "malignant"

            mode = st.session_state.get("preset_mode", "median")
            if mode == "benign":
                row = df[df["target_label"] == "Benign"].iloc[0]
                preset_values = {f: float(row[f]) for f in metadata["feature_names"]}
            elif mode == "malignant":
                row = df[df["target_label"] == "Malignant"].iloc[0]
                preset_values = {f: float(row[f]) for f in metadata["feature_names"]}

        feature_names = metadata["feature_names"]
        feature_stats = metadata["feature_stats"]
        user_inputs = {}

        mean_feats = [f for f in feature_names if f.endswith("_mean")]
        se_feats = [f for f in feature_names if f.endswith("_se")]
        worst_feats = [f for f in feature_names if f.endswith("_worst")]
        other_feats = [f for f in feature_names if f not in mean_feats + se_feats + worst_feats]

        tab1, tab2, tab3 = st.tabs(["Mean Features", "Standard Error (SE) Features", "Worst Features"])
        groups = [
            (tab1, mean_feats + other_feats),
            (tab2, se_feats),
            (tab3, worst_feats),
        ]

        for tab, feats in groups:
            with tab:
                cols = st.columns(3)
                for i, feat in enumerate(feats):
                    stats = feature_stats[feat]
                    default_val = preset_values.get(feat, stats["median"])
                    with cols[i % 3]:
                        user_inputs[feat] = st.number_input(
                            label=feat,
                            min_value=0.0,
                            max_value=float(stats["max"] * 5.0),
                            value=float(default_val),
                            format="%.5f",
                            key=f"input_{feat}",
                        )

        if st.button("🔍 Predict Tumour Class", type="primary"):
            try:
                # Build DataFrame in exact training feature order
                ordered_row = {feat: float(user_inputs[feat]) for feat in feature_names}
                input_df = pd.DataFrame([ordered_row], columns=feature_names)

                if input_df.isnull().any().any() or np.isinf(input_df.values).any():
                    st.error("Invalid input detected (NaN or Infinity). Please check your measurements.")
                    st.stop()

                pkl_file = metadata["saved_model_files"][selected_model_name]
                pipeline = load_model(pkl_file)
                if pipeline is None:
                    st.error(f"Could not load model file `{pkl_file}`.")
                    st.stop()

                pred_class = int(pipeline.predict(input_df)[0])
                pred_label = metadata["reverse_target_mapping"][pred_class]

                if pred_class == 1:
                    st.error(f"### Prediction: **{pred_label}** (Class 1)")
                else:
                    st.success(f"### Prediction: **{pred_label}** (Class 0)")

                if hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba(input_df)[0]
                    c1, c2 = st.columns(2)
                    c1.metric("Benign Probability (0)", f"{probs[0]*100:.2f}%")
                    c2.metric("Malignant Probability (1)", f"{probs[1]*100:.2f}%")

                st.caption(f"Model used: `{selected_model_name}` (`pickle/{pkl_file}`)")
            except Exception as exc:
                st.error(f"Prediction failed due to an unexpected error: {exc}")

    # ---------------------------------------------------------
    # 5. ABOUT / LIMITATIONS
    # ---------------------------------------------------------
    elif section == "5. About / Limitations":
        st.header("About TumorVision & Limitations")
        render_disclaimer()
        st.subheader("Methodology")
        st.markdown(
            """
            1. **Data Cleaning:** Removed empty (`Unnamed: 32`) and identifier (`id`) columns; verified zero duplicates and zero missing feature values.
            2. **Target Encoding:** Mapped `B` (Benign) → `0` and `M` (Malignant) → `1` (positive class).
            3. **Leakage-Free Preprocessing:** Stratified 80:20 train-test split, followed by `StandardScaler` fit strictly on training data.
            4. **PCA Dimensionality Reduction:** Fit `PCA` on scaled training data to retain at least 95% cumulative explained variance.
            5. **Model Training:** Trained 4 algorithms (Logistic Regression, KNN, Decision Tree, Random Forest) both without PCA and with PCA using Scikit-learn `Pipeline`.
            """
        )

        st.subheader("How PCA Works & Information Loss")
        n_comp = metadata["pca_info"]["n_components"]
        cum_pct = metadata["pca_info"]["cumulative_variance_ratio"][-1] * 100
        discarded_pct = 100.0 - cum_pct
        n_feats = metadata["dataset_summary"]["num_features"]
        st.markdown(
            f"""
            **Principal Component Analysis (PCA)** transforms correlated numerical features into a smaller set of
            uncorrelated orthogonal axes (Principal Components) ordered by how much variance they capture.
            Here, **{n_comp} principal components** capture **{cum_pct:.2f}%** of the variance from the original **{n_feats} features**.
            The remaining ~{discarded_pct:.2f}% variance is discarded, which may remove subtle fine-grained signal or noise,
            and principal components are linear combinations of original features—making individual components less
            directly interpretable than raw measurements like `radius_mean`.
            """
        )

        st.subheader("Limitations")
        st.markdown(
            """
            - **Educational Scope:** Trained on a single 569-sample public dataset; not validated for clinical deployment.
            - **Feature Dependency:** Requires all 30 cell-nucleus measurements in the exact units and definitions of the dataset.
            - **No Clinical Context:** Does not incorporate patient history, imaging, or biopsy pathology reports.
            """
        )


if __name__ == "__main__":
    main()
