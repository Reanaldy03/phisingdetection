import os
import time
import argparse
import warnings
import joblib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

import shap


warnings.filterwarnings("ignore")


RANDOM_STATE = 42
SHAP_MAX_SAMPLES = 3000
TOP_K_LIST = [5, 10, 15, 20, 25, 30, 40, 50]
F1_TOLERANCE = 0.002

OUTPUT_DIR = "outputs"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")


def make_dirs():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Klasifikasi URL Phishing Menggunakan LightGBM dengan Seleksi Fitur Berbasis SHAP"
    )

    parser.add_argument(
        "--train",
        type=str,
        default="Training.parquet",
        help="Path file Training.parquet"
    )

    parser.add_argument(
        "--test",
        type=str,
        default="Testing.parquet",
        help="Path file Testing.parquet"
    )

    parser.add_argument(
        "--target",
        type=str,
        default="status",
        help="Nama kolom target"
    )

    parser.add_argument(
        "--drop_constant",
        type=str,
        default="yes",
        choices=["yes", "no"],
        help="Buang fitur konstan dari data training"
    )

    parser.add_argument(
        "--shap_sample",
        type=int,
        default=SHAP_MAX_SAMPLES,
        help="Jumlah sampel data training untuk perhitungan SHAP"
    )

    return parser.parse_args()


def read_parquet_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    try:
        return pd.read_parquet(path)
    except Exception:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        return table.to_pandas()


def save_excel(df, filename, index=False):
    path = os.path.join(TABLE_DIR, filename)
    df.to_excel(path, index=index)
    print(f"Tabel disimpan: {path}")


# ── Academic plot theme ────────────────────────────────────────────────────────
ACADEMIC_PALETTE = ["#1B4F72", "#E74C3C", "#1ABC9C", "#884EA0", "#D68910",
                    "#2874A6", "#117A65", "#943126", "#7D6608", "#4A235A"]

def set_academic_style():
    """Apply a clean, publication-quality matplotlib style."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#cccccc",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.6,
    })


def save_figure(filename):
    path = os.path.join(FIGURE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    plt.rcdefaults()          # reset so next figure starts fresh
    print(f"Grafik disimpan: {path}")


def clean_target(series):
    y = series.copy()

    if y.dtype == "object":
        y = y.astype(str).str.lower().str.strip()

        mapping = {
            "legitimate": 0,
            "legit": 0,
            "benign": 0,
            "safe": 0,
            "normal": 0,
            "0": 0,
            "phishing": 1,
            "phish": 1,
            "malicious": 1,
            "unsafe": 1,
            "attack": 1,
            "1": 1
        }

        y = y.map(mapping)

        if y.isna().sum() > 0:
            raise ValueError("Target memiliki nilai teks yang belum bisa dipetakan ke 0 dan 1.")

        return y.astype(int)

    values = sorted(pd.Series(y.dropna().unique()).tolist())

    if set(values) == {0, 1}:
        return y.astype(int)

    if len(values) == 2:
        return y.map({values[0]: 0, values[1]: 1}).astype(int)

    raise ValueError("Target harus punya dua kelas.")


def prepare_data(train_df, test_df, target_col, drop_constant=True):
    if target_col not in train_df.columns:
        raise ValueError(f"Kolom target {target_col} tidak ada di data training.")

    if target_col not in test_df.columns:
        raise ValueError(f"Kolom target {target_col} tidak ada di data testing.")

    y_train = clean_target(train_df[target_col])
    y_test = clean_target(test_df[target_col])

    drop_cols = [target_col]

    if "url" in train_df.columns:
        drop_cols.append("url")

    X_train_raw = train_df.drop(columns=drop_cols, errors="ignore")
    X_test_raw = test_df.drop(columns=drop_cols, errors="ignore")

    numeric_cols = X_train_raw.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    non_numeric_cols = [col for col in X_train_raw.columns if col not in numeric_cols]

    X_train = X_train_raw[numeric_cols].copy()
    X_test = X_test_raw[numeric_cols].copy()

    for col in X_train.columns:
        if X_train[col].dtype == "bool":
            X_train[col] = X_train[col].astype(int)
            X_test[col] = X_test[col].astype(int)

    constant_cols = []

    if drop_constant:
        constant_cols = [
            col for col in X_train.columns
            if X_train[col].nunique(dropna=False) <= 1
        ]

        X_train = X_train.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols)

    for col in X_train.columns:
        if X_train[col].isna().sum() > 0:
            median_value = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_value)
            X_test[col] = X_test[col].fillna(median_value)

    metadata = {
        "numeric_cols": numeric_cols,
        "non_numeric_cols": non_numeric_cols,
        "constant_cols": constant_cols,
        "final_features": X_train.columns.tolist()
    }

    return X_train, X_test, y_train, y_test, metadata


def create_dataset_tables(train_df, test_df, X_train, X_test, y_train, y_test, metadata, target_col):
    info = pd.DataFrame({
        "item": [
            "training_rows",
            "training_columns",
            "testing_rows",
            "testing_columns",
            "target_column",
            "numeric_features_before_constant_drop",
            "constant_features_dropped",
            "final_features_used",
            "training_missing_total",
            "testing_missing_total",
            "training_duplicate_rows",
            "testing_duplicate_rows"
        ],
        "value": [
            train_df.shape[0],
            train_df.shape[1],
            test_df.shape[0],
            test_df.shape[1],
            target_col,
            len(metadata["numeric_cols"]),
            len(metadata["constant_cols"]),
            X_train.shape[1],
            int(train_df.isna().sum().sum()),
            int(test_df.isna().sum().sum()),
            int(train_df.duplicated().sum()),
            int(test_df.duplicated().sum())
        ]
    })

    save_excel(info, "table_00_dataset_information.xlsx")

    train_structure = pd.DataFrame({
        "column": train_df.columns,
        "dtype": train_df.dtypes.astype(str).values,
        "missing_value": train_df.isna().sum().values,
        "missing_percent": (train_df.isna().sum().values / len(train_df) * 100).round(4),
        "unique_value": train_df.nunique().values
    })

    save_excel(train_structure, "table_01_training_structure.xlsx")

    test_structure = pd.DataFrame({
        "column": test_df.columns,
        "dtype": test_df.dtypes.astype(str).values,
        "missing_value": test_df.isna().sum().values,
        "missing_percent": (test_df.isna().sum().values / len(test_df) * 100).round(4),
        "unique_value": test_df.nunique().values
    })

    save_excel(test_structure, "table_02_testing_structure.xlsx")

    class_distribution = pd.DataFrame({
        "dataset": ["Training", "Training", "Testing", "Testing"],
        "class": [0, 1, 0, 1],
        "label": ["Legitimate", "Phishing", "Legitimate", "Phishing"],
        "count": [
            int((y_train == 0).sum()),
            int((y_train == 1).sum()),
            int((y_test == 0).sum()),
            int((y_test == 1).sum())
        ]
    })

    class_distribution["percentage"] = class_distribution.groupby("dataset")["count"].transform(
        lambda x: (x / x.sum() * 100).round(4)
    )

    save_excel(class_distribution, "table_03_class_distribution.xlsx")

    desc = X_train.describe().T.reset_index().rename(columns={"index": "feature"})
    save_excel(desc, "table_04_descriptive_statistics_training.xlsx")

    non_numeric_table = pd.DataFrame({
        "non_numeric_column_dropped": metadata["non_numeric_cols"]
    })

    save_excel(non_numeric_table, "table_05_non_numeric_columns_dropped.xlsx")

    constant_table = pd.DataFrame({
        "constant_feature_dropped": metadata["constant_cols"]
    })

    save_excel(constant_table, "table_06_constant_features_dropped.xlsx")


def create_eda_figures(train_df, X_train, y_train, target_col):
    class_plot_df = pd.DataFrame({
        "status": y_train.map({0: "Legitimate", 1: "Phishing"})
    })

    counts = class_plot_df["status"].value_counts().reindex(["Legitimate", "Phishing"])

    # ── Figure 01: Class Distribution ────────────────────────────────────────
    set_academic_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = [ACADEMIC_PALETTE[0], ACADEMIC_PALETTE[1]]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5,
                  edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Class Distribution in Training Data", pad=12)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, counts.max() * 1.12)
    save_figure("figure_01_class_distribution_training.png")

    # ── Figure 02: Missing Values ─────────────────────────────────────────────
    missing = train_df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    set_academic_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    if len(missing) == 0:
        ax.text(0.5, 0.5, "No missing values detected in training data",
                ha="center", va="center", fontsize=13,
                style="italic", color="#555555",
                transform=ax.transAxes)
        ax.axis("off")
    else:
        ax.bar(range(len(missing)), missing.values, color=ACADEMIC_PALETTE[2],
               edgecolor="white", linewidth=0.6)
        ax.set_xticks(range(len(missing)))
        ax.set_xticklabels(missing.index, rotation=45, ha="right", fontsize=8)
        ax.set_title("Missing Values in Training Data", pad=12)
        ax.set_xlabel("Column")
        ax.set_ylabel("Missing Count")

    save_figure("figure_02_missing_values_training.png")

    # ── Figure 03: Top-20 Feature-Target Correlation ──────────────────────────
    corr_df = pd.concat([X_train, y_train.rename(target_col)], axis=1)
    corr_target = corr_df.corr(numeric_only=True)[target_col].drop(target_col)

    corr_table = corr_target.abs().sort_values(ascending=False).reset_index()
    corr_table.columns = ["feature", "absolute_correlation_with_target"]

    save_excel(corr_table, "table_07_feature_target_correlation.xlsx")

    top_corr = corr_table.head(20).sort_values("absolute_correlation_with_target")

    set_academic_style()
    fig, ax = plt.subplots(figsize=(9, 7))
    norm_vals = top_corr["absolute_correlation_with_target"].values
    cmap = LinearSegmentedColormap.from_list("acad", ["#AED6F1", ACADEMIC_PALETTE[0]])
    bar_colors = [cmap(v / norm_vals.max()) for v in norm_vals]
    ax.barh(top_corr["feature"], norm_vals, color=bar_colors,
            edgecolor="white", linewidth=0.6)
    ax.set_title("Top 20 Feature–Target Absolute Correlation", pad=12)
    ax.set_xlabel("Absolute Pearson Correlation")
    ax.set_ylabel("Feature")
    ax.set_xlim(0, norm_vals.max() * 1.12)
    for i, val in enumerate(norm_vals):
        ax.text(val + norm_vals.max() * 0.01, i, f"{val:.3f}",
                va="center", fontsize=8)
    save_figure("figure_03_top20_feature_target_correlation.png")


def create_model():
    model = LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.90,
        colsample_bytree=0.90,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        importance_type="gain",
        verbose=-1
    )

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    # Convert DataFrame to a contiguous numpy array before timing to eliminate Pandas indexing overhead
    X_test_np = np.ascontiguousarray(X_test.values, dtype=np.float32)

    # Warm-up run to initialize thread pools and memory structures
    _ = model.predict(X_test_np)

    # Measure prediction time over multiple runs to filter out OS/CPU scaling noise
    n_runs = 10
    pred_times = []
    for _ in range(n_runs):
        t0 = time.time()
        y_pred = model.predict(X_test_np)
        y_prob = model.predict_proba(X_test_np)[:, 1]
        pred_times.append(time.time() - t0)
    
    # Take the minimum run time (standard in system benchmarking to represent peak speed)
    prediction_time = min(pred_times)

    result = {
        "model": model_name,
        "n_features": X_train.shape[1],
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "train_time_second": train_time,
        "prediction_time_second": prediction_time
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Phishing"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T
    report_df.insert(0, "model", model_name)

    return result, y_pred, y_prob, report_df


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Legitimate", "Phishing"]

    set_academic_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.grid(False)  # Remove the white gridlines over the matrix blocks

    # Custom blue colormap for academic style
    cmap = LinearSegmentedColormap.from_list("cm_blue", ["#EBF5FB", ACADEMIC_PALETTE[0]])
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / cm[i].sum() * 100
            ax.text(j, i, f"{cm[i, j]:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "#1B4F72")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=10, rotation=90, va="center")
    ax.set_title(title, pad=12)
    ax.set_xlabel("Predicted Label", labelpad=8)
    ax.set_ylabel("True Label", labelpad=8)
    save_figure(filename)


def plot_roc_curve(y_true, y_prob, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    set_academic_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=ACADEMIC_PALETTE[0], linewidth=2.0,
            label=f"LightGBM (AUC = {auc_value:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999",
            linewidth=1.2, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color=ACADEMIC_PALETTE[0])
    ax.set_title(title, pad=12)
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.legend(loc="lower right")
    save_figure(filename)


def create_lightgbm_importance(model, feature_names):
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance_gain": model.feature_importances_
    }).sort_values("importance_gain", ascending=False)

    importance.insert(0, "rank", range(1, len(importance) + 1))

    save_excel(importance, "table_09_lightgbm_feature_importance.xlsx")

    top_importance = importance.head(20).sort_values("importance_gain")

    set_academic_style()
    fig, ax = plt.subplots(figsize=(9, 7))
    vals = top_importance["importance_gain"].values
    cmap = LinearSegmentedColormap.from_list("imp", ["#AED6F1", ACADEMIC_PALETTE[0]])
    bar_colors = [cmap(v / vals.max()) for v in vals]
    ax.barh(top_importance["feature"], vals, color=bar_colors,
            edgecolor="white", linewidth=0.6)
    ax.set_title("Top 20 Feature Importance — LightGBM (Gain)", pad=12)
    ax.set_xlabel("Importance Score (Gain)")
    ax.set_ylabel("Feature")
    ax.set_xlim(0, vals.max() * 1.12)
    for i, val in enumerate(vals):
        ax.text(val + vals.max() * 0.01, i, f"{val:.1f}",
                va="center", fontsize=8)
    save_figure("figure_06_top20_lightgbm_feature_importance.png")

    return importance


def compute_shap_importance(model, X_train, shap_sample):
    sample_size = min(shap_sample, len(X_train))
    X_shap = X_train.sample(sample_size, random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        shap_values_class = shap_values[1]
    else:
        shap_values_class = shap_values

    shap_values_class = np.array(shap_values_class)

    if len(shap_values_class.shape) == 3:
        shap_values_class = shap_values_class[:, :, 1]

    shap_importance = pd.DataFrame({
        "feature": X_shap.columns,
        "mean_abs_shap": np.abs(shap_values_class).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    shap_importance.insert(0, "rank", range(1, len(shap_importance) + 1))

    save_excel(shap_importance, "table_10_shap_feature_importance.xlsx")

    set_academic_style()
    plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_values_class,
        X_shap,
        plot_type="bar",
        show=False,
        max_display=20,
        color=ACADEMIC_PALETTE[0]
    )
    ax = plt.gca()
    ax.set_title("Top 20 SHAP Feature Importance (Mean |SHAP Value|)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean |SHAP Value|  (Average Impact on Model Output)",
                 fontsize=11, fontweight="bold")
    save_figure("figure_07_shap_bar_top20.png")

    set_academic_style()
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_class,
        X_shap,
        show=False,
        max_display=20
    )
    ax = plt.gca()
    ax.set_title("SHAP Beeswarm Plot — Top 20 Features",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("SHAP Value  (Impact on Model Output Magnitude)",
                 fontsize=11, fontweight="bold")
    save_figure("figure_08_shap_beeswarm_top20.png")

    return shap_importance


def run_shap_selection_experiments(X_train, y_train, X_test, y_test, shap_importance, baseline_result):
    results = [baseline_result]
    store = {}

    total_features = X_train.shape[1]
    valid_top_k = [k for k in TOP_K_LIST if k < total_features]

    for k in valid_top_k:
        selected_features = shap_importance.head(k)["feature"].tolist()

        model = create_model()
        model_name = f"LightGBM_SHAP_Top_{k}"

        result, pred, prob, report = evaluate_model(
            model,
            X_train[selected_features],
            y_train,
            X_test[selected_features],
            y_test,
            model_name
        )

        results.append(result)

        store[model_name] = {
            "model": model,
            "features": selected_features,
            "pred": pred,
            "prob": prob,
            "report": report
        }

        save_excel(report, f"table_classification_report_{model_name}.xlsx", index=True)

        print(
            f"{model_name}: "
            f"F1={result['f1_score']:.4f}, "
            f"ROC-AUC={result['roc_auc']:.4f}"
        )

    result_df = pd.DataFrame(results)
    save_excel(result_df, "table_11_model_comparison_original_order.xlsx")

    sorted_df = result_df.sort_values(
        by=["f1_score", "roc_auc", "n_features", "prediction_time_second"],
        ascending=[False, False, True, True]
    ).reset_index(drop=True)

    save_excel(sorted_df, "table_12_model_comparison_sorted.xlsx")

    return result_df, sorted_df, store


def plot_model_comparison(result_df):
    # Sort result_df by n_features to ensure the models are plotted in ascending order of feature size
    result_df_sorted = result_df.sort_values(by="n_features")
    model_order = result_df_sorted["model"].tolist()

    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    melted = result_df.melt(
        id_vars=["model", "n_features"],
        value_vars=metrics,
        var_name="metric",
        value_name="score"
    )

    pivot = melted.pivot_table(
        index="model",
        columns="metric",
        values="score"
    )[metrics]  # keep consistent column order

    # Reindex pivot to follow the sorted order of n_features (Top-5, Top-10, ..., All Features)
    pivot = pivot.reindex(model_order)

    # Rename columns for display
    pivot.columns = metric_labels

    # Shorten model names for x-axis readability
    short_names = [
        m.replace("LightGBM_SHAP_Top_", "Top-").replace("LightGBM_All_Features", "All Features")
        for m in pivot.index
    ]

    # ── Figure 09: Grouped Bar — Model Comparison ─────────────────────────────
    set_academic_style()
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(pivot))
    n_metrics = len(metric_labels)
    width = 0.14
    offsets = np.linspace(-(n_metrics - 1) * width / 2, (n_metrics - 1) * width / 2, n_metrics)

    for offset, col, color in zip(offsets, metric_labels, ACADEMIC_PALETTE[:n_metrics]):
        ax.bar(x + offset, pivot[col], width, label=col, color=color,
               edgecolor="white", linewidth=0.5)

    ax.set_title("Model Performance Comparison Across Feature Subsets", pad=12)
    ax.set_xlabel("Model", labelpad=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0.85, 1.005)   # zoom in to highlight differences
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=0, ha="center", fontsize=9.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=n_metrics, fontsize=9.5, frameon=True)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    save_figure("figure_09_model_performance_comparison.png")

    # ── Line plots: F1, ROC-AUC, Prediction Time vs Feature Count ─────────────
    ordered = result_df.sort_values("n_features")

    line_cfg = [
        ("f1_score",              "F1-Score",          "figure_10_feature_count_vs_f1.png"),
        ("roc_auc",               "ROC-AUC",           "figure_11_feature_count_vs_roc_auc.png"),
        ("prediction_time_second","Prediction Time (s)","figure_12_feature_count_vs_prediction_time.png"),
    ]

    for col, ylabel, fname in line_cfg:
        set_academic_style()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ordered["n_features"], ordered[col],
                marker="o", markersize=6, linewidth=1.8,
                color=ACADEMIC_PALETTE[0], markerfacecolor="white",
                markeredgewidth=1.5, markeredgecolor=ACADEMIC_PALETTE[0])
        ax.set_title(f"Number of Features vs. {ylabel}", pad=12)
        ax.set_xlabel("Number of Features")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="#dddddd")
        # Annotate each point
        for xv, yv in zip(ordered["n_features"], ordered[col]):
            ax.annotate(f"{yv:.4f}", (xv, yv),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5, color="#333333")
        save_figure(fname)


def choose_final_model(sorted_df, baseline_model, baseline_pred, baseline_prob, X_train, store):
    # Cari nilai F1-score tertinggi dari seluruh model
    max_f1 = sorted_df["f1_score"].max()
    
    # Ambil semua model yang performanya dalam batas toleransi F1-score tertinggi
    eligible = sorted_df[sorted_df["f1_score"] >= (max_f1 - F1_TOLERANCE)]
    
    # Dari model yang memenuhi syarat tersebut, pilih yang memiliki n_features terkecil
    best_row = eligible.sort_values(
        by=["n_features", "f1_score", "roc_auc"], 
        ascending=[True, False, False]
    ).iloc[0]
    
    name = best_row["model"]
    
    print(f"\n[Model Selection] F1-score tertinggi: {max_f1:.4f}")
    print(f"[Model Selection] Batas toleransi F1-score: {max_f1 - F1_TOLERANCE:.4f}")
    print(f"[Model Selection] Model terpilih (Opsi A): {name} dengan {best_row['n_features']} fitur (F1={best_row['f1_score']:.4f})")

    if name == "LightGBM_All_Features":
        final_model = baseline_model
        final_features = X_train.columns.tolist()
        final_pred = baseline_pred
        final_prob = baseline_prob
    else:
        final_model = store[name]["model"]
        final_features = store[name]["features"]
        final_pred = store[name]["pred"]
        final_prob = store[name]["prob"]

    selected_feature_table = pd.DataFrame({
        "selected_feature": final_features
    })

    save_excel(selected_feature_table, "table_13_selected_features_final.xlsx")

    joblib.dump(
        final_model,
        os.path.join(MODEL_DIR, "final_lightgbm_shap_model.pkl")
    )

    best_summary = pd.DataFrame({
        "item": best_row.index,
        "value": best_row.values
    })

    save_excel(best_summary, "table_14_best_model_summary.xlsx")

    return name, final_model, final_features, final_pred, final_prob


def run_cross_validation_on_training(X_train, y_train, final_features):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    model = create_model()

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    scores = cross_validate(
        model,
        X_train[final_features],
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    scores_df = pd.DataFrame(scores)

    summary = scores_df.agg(["mean", "std"]).T.reset_index()
    summary = summary.rename(columns={"index": "metric"})

    save_excel(scores_df, "table_15_cross_validation_training_scores.xlsx")
    save_excel(summary, "table_16_cross_validation_training_summary.xlsx")

    return scores_df, summary


def save_text_summary(train_df, test_df, X_train, X_test, y_train, y_test, metadata, best_name, sorted_df, final_features):
    best = sorted_df[sorted_df["model"] == best_name].iloc[0]

    text = f"""
RINGKASAN HASIL EKSPERIMEN

Judul:
Klasifikasi URL Phishing Menggunakan LightGBM dengan Seleksi Fitur Berbasis SHAP

Data:
Training: {train_df.shape[0]} baris dan {train_df.shape[1]} kolom
Testing: {test_df.shape[0]} baris dan {test_df.shape[1]} kolom
Target: status
Kolom URL mentah tidak dipakai sebagai fitur model
Jumlah fitur numerik awal: {len(metadata["numeric_cols"])}
Jumlah fitur konstan yang dibuang: {len(metadata["constant_cols"])}
Jumlah fitur final sebelum seleksi SHAP: {X_train.shape[1]}
Missing value training: {int(train_df.isna().sum().sum())}
Missing value testing: {int(test_df.isna().sum().sum())}

Distribusi kelas training:
Legitimate: {int((y_train == 0).sum())}
Phishing: {int((y_train == 1).sum())}

Distribusi kelas testing:
Legitimate: {int((y_test == 0).sum())}
Phishing: {int((y_test == 1).sum())}

Model terbaik:
{best_name}

Performa model terbaik pada data testing:
Accuracy: {best["accuracy"]:.4f}
Precision: {best["precision"]:.4f}
Recall: {best["recall"]:.4f}
F1-score: {best["f1_score"]:.4f}
ROC-AUC: {best["roc_auc"]:.4f}
Jumlah fitur: {int(best["n_features"])}
Training time: {best["train_time_second"]:.4f} detik
Prediction time: {best["prediction_time_second"]:.4f} detik

Fitur model terbaik:
{", ".join(final_features)}

Kalimat hasil awal untuk jurnal:
Penelitian ini membandingkan LightGBM dengan seluruh fitur numerik dan LightGBM dengan fitur terpilih berdasarkan nilai mean absolute SHAP. Hasil evaluasi menunjukkan model terbaik adalah {best_name} dengan F1-score sebesar {best["f1_score"]:.4f} dan ROC-AUC sebesar {best["roc_auc"]:.4f}. Temuan ini menunjukkan seleksi fitur berbasis SHAP dapat digunakan untuk memilih fitur URL yang paling berkontribusi terhadap deteksi phishing.
"""

    path = os.path.join(OUTPUT_DIR, "ringkasan_hasil_eksperimen.txt")

    with open(path, "w", encoding="utf-8") as file:
        file.write(text)

    print(f"Ringkasan disimpan: {path}")


def main():
    args = parse_args()

    make_dirs()

    train_df = read_parquet_file(args.train)
    test_df = read_parquet_file(args.test)

    print("DATA BERHASIL DIBACA")
    print(f"Training: {train_df.shape}")
    print(f"Testing: {test_df.shape}")

    if list(train_df.columns) != list(test_df.columns):
        raise ValueError("Kolom Training.parquet dan Testing.parquet tidak sama.")

    X_train, X_test, y_train, y_test, metadata = prepare_data(
        train_df,
        test_df,
        args.target,
        drop_constant=(args.drop_constant == "yes")
    )

    print("DATA SIAP MODEL")
    print(f"Fitur training: {X_train.shape}")
    print(f"Fitur testing: {X_test.shape}")

    create_dataset_tables(
        train_df,
        test_df,
        X_train,
        X_test,
        y_train,
        y_test,
        metadata,
        args.target
    )

    create_eda_figures(
        train_df,
        X_train,
        y_train,
        args.target
    )

    baseline_model = create_model()

    baseline_result, baseline_pred, baseline_prob, baseline_report = evaluate_model(
        baseline_model,
        X_train,
        y_train,
        X_test,
        y_test,
        "LightGBM_All_Features"
    )

    save_excel(
        baseline_report,
        "table_08_classification_report_baseline.xlsx",
        index=True
    )

    plot_confusion_matrix(
        y_test,
        baseline_pred,
        "Confusion Matrix LightGBM Baseline",
        "figure_04_confusion_matrix_baseline.png"
    )

    plot_roc_curve(
        y_test,
        baseline_prob,
        "ROC Curve LightGBM Baseline",
        "figure_05_roc_curve_baseline.png"
    )

    create_lightgbm_importance(
        baseline_model,
        X_train.columns
    )

    shap_importance = compute_shap_importance(
        baseline_model,
        X_train,
        args.shap_sample
    )

    result_df, sorted_df, store = run_shap_selection_experiments(
        X_train,
        y_train,
        X_test,
        y_test,
        shap_importance,
        baseline_result
    )

    plot_model_comparison(result_df)

    best_name, final_model, final_features, final_pred, final_prob = choose_final_model(
        sorted_df,
        baseline_model,
        baseline_pred,
        baseline_prob,
        X_train,
        store
    )

    plot_confusion_matrix(
        y_test,
        final_pred,
        "Confusion Matrix Model Terbaik",
        "figure_13_confusion_matrix_final_model.png"
    )

    plot_roc_curve(
        y_test,
        final_prob,
        "ROC Curve Model Terbaik",
        "figure_14_roc_curve_final_model.png"
    )

    final_report = classification_report(
        y_test,
        final_pred,
        target_names=["Legitimate", "Phishing"],
        output_dict=True,
        zero_division=0
    )

    final_report_df = pd.DataFrame(final_report).T
    final_report_df.insert(0, "model", best_name)

    save_excel(
        final_report_df,
        "table_17_classification_report_final_model.xlsx",
        index=True
    )

    run_cross_validation_on_training(
        X_train,
        y_train,
        final_features
    )

    save_text_summary(
        train_df,
        test_df,
        X_train,
        X_test,
        y_train,
        y_test,
        metadata,
        best_name,
        sorted_df,
        final_features
    )

    print("SELESAI")
    print(f"Tabel: {TABLE_DIR}")
    print(f"Grafik: {FIGURE_DIR}")
    print(f"Model: {MODEL_DIR}")
    print("Ringkasan: outputs/ringkasan_hasil_eksperimen.txt")


if __name__ == "__main__":
    main()