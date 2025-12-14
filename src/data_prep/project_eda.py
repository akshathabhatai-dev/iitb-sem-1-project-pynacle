import os
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone


# ============================================================
# 1. BASIC DATA CHECKS: MISSING VALUES & DUPLICATES
# ============================================================

def check_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reports missing data by column and row, then imputes missing values
    using the MEAN of each numeric column. Returns the cleaned DataFrame.
    (Non-numeric columns are left as-is.)
    """
    cleaned_df = df.copy()

    print("=== Missing Data: By Column ===")
    missing_by_col = cleaned_df.isnull().sum()
    missing_cols = missing_by_col[missing_by_col > 0]

    if len(missing_cols) > 0:
        print(missing_cols)
    else:
        print("No columns have missing data.")

    print("\nTotal columns with missing values:", (missing_by_col > 0).sum())

    print("\n=== Missing Data: By Row ===")
    missing_by_row = cleaned_df.isnull().sum(axis=1)
    missing_rows = missing_by_row[missing_by_row > 0]

    if len(missing_rows) > 0:
        print(missing_rows.head())
    else:
        print("No rows have missing data.")

    print("\nTotal rows with missing values:", len(missing_rows))

    # Impute missing numeric values with mean
    print("\n=== Imputation Summary (mean per numeric column) ===")
    imputations = {}

    for col in cleaned_df.columns:
        if cleaned_df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                mean_value = cleaned_df[col].mean()
                if pd.isna(mean_value):
                    print(f"Column '{col}': all values are missing, no mean available. Left as-is.")
                    continue
                imputations[col] = mean_value
                cleaned_df[col] = cleaned_df[col].fillna(mean_value)
            else:
                print(f"Column '{col}' is non-numeric. Imputation skipped.")

    if imputations:
        for col, val in imputations.items():
            print(f"Column '{col}': filled missing values with mean = {val}")
    else:
        print("No imputations performed (no missing values).")

    return cleaned_df


def report_duplicates(df: pd.DataFrame) -> None:
    """
    Reports duplicate rows and columns, but DOES NOT remove them.
    """
    print("\n=== Duplicate Rows (report only) ===")
    duplicate_row_mask = df.duplicated()
    dup_row_count = duplicate_row_mask.sum()
    print("Number of duplicate rows:", dup_row_count)

    if dup_row_count > 0:
        print("\nSample duplicate rows:")
        print(df[duplicate_row_mask].head())
    else:
        print("No duplicate rows found.")

    print("\n=== Duplicate Columns (report only) ===")
    cols = df.columns
    duplicate_cols_pairs = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if df[cols[i]].equals(df[cols[j]]):
                duplicate_cols_pairs.append((cols[i], cols[j]))

    if duplicate_cols_pairs:
        print("Duplicate column pairs (same data):")
        for c1, c2 in duplicate_cols_pairs:
            print(f"{c1} <==> {c2}")
    else:
        print("No duplicate columns found.")


# ============================================================
# 2. DESCRIPTIVE STATS & VISUALS
# ============================================================

def generate_descriptive_analysis(
    df: pd.DataFrame,
    output_dir: str,
    exclude_cols=None,
    file_name: str = "descriptive_analysis.xlsx"
):
    """
    Generates descriptive analysis for numeric and categorical features
    and saves the results to an Excel file. Returns feature-only dataframe.
    """
    if exclude_cols is None:
        exclude_cols = ["label", "image_name", "cell_index"]

    exclude_lower = {c.lower() for c in exclude_cols}
    excluded_columns_found = [c for c in df.columns if c.lower() in exclude_lower]

    df_features = df.drop(columns=excluded_columns_found, errors="ignore")

    numeric_cols = df_features.select_dtypes(include=["number"]).columns
    categorical_cols = df_features.select_dtypes(exclude=["number"]).columns

    numeric_stats = pd.DataFrame()
    categorical_stats = pd.DataFrame()

    if len(numeric_cols) > 0:
        numeric_stats = df_features[numeric_cols].describe().T
        numeric_stats["missing_count"] = df_features[numeric_cols].isnull().sum()
        numeric_stats["missing_pct"] = (
            numeric_stats["missing_count"] / len(df_features)
        ) * 100
        numeric_stats["n_unique"] = df_features[numeric_cols].nunique()

    if len(categorical_cols) > 0:
        categorical_stats = df_features[categorical_cols].describe().T
        categorical_stats["missing_count"] = df_features[categorical_cols].isnull().sum()
        categorical_stats["missing_pct"] = (
            categorical_stats["missing_count"] / len(df_features)
        ) * 100

    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, file_name)

    with pd.ExcelWriter(full_path, engine="openpyxl") as writer:
        if not numeric_stats.empty:
            numeric_stats.to_excel(writer, sheet_name="Numeric_Features")
        if not categorical_stats.empty:
            categorical_stats.to_excel(writer, sheet_name="Categorical_Features")
        pd.DataFrame({"excluded_columns": excluded_columns_found}).to_excel(
            writer, sheet_name="Excluded_Columns", index=False
        )

    print(f"Descriptive analysis file generated at: {full_path}")
    return df_features


def generate_feature_histograms(
    df: pd.DataFrame,
    output_dir: str,
    exclude_cols=None,
    file_name: str = "feature_histograms.png",
    bins: int = 30
):
    """Big grid of histograms for numeric features."""
    if exclude_cols is None:
        exclude_cols = ["label", "image_name", "cell_index"]

    exclude_lower = {c.lower() for c in exclude_cols}
    cols_to_use = [c for c in df.columns if c.lower() not in exclude_lower]

    df_numeric = df[cols_to_use].select_dtypes(include=["number"])
    if df_numeric.shape[1] == 0:
        print("No numeric features available for histogram generation.")
        return

    feature_cols = df_numeric.columns.tolist()
    n_features = len(feature_cols)

    n_cols = 8
    n_rows = math.ceil(n_features / n_cols)

    plt.figure(figsize=(n_cols * 3, n_rows * 2.5))

    for idx, col in enumerate(feature_cols, start=1):
        ax = plt.subplot(n_rows, n_cols, idx)
        data = df_numeric[col].dropna()
        ax.hist(data, bins=bins)
        ax.set_title(col, fontsize=8)
        ax.tick_params(axis="both", labelsize=6)

    total_plots = n_rows * n_cols
    if total_plots > n_features:
        for idx in range(n_features + 1, total_plots + 1):
            ax = plt.subplot(n_rows, n_cols, idx)
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Histogram grid image generated at: {output_path}")


def generate_feature_boxplots(
    df: pd.DataFrame,
    output_dir: str,
    exclude_cols=None,
    file_name: str = "feature_boxplots.png"
):
    """Big grid of boxplots for numeric features."""
    if exclude_cols is None:
        exclude_cols = ["label", "image_name", "cell_index"]

    exclude_lower = {c.lower() for c in exclude_cols}
    cols_to_use = [c for c in df.columns if c.lower() not in exclude_lower]

    df_numeric = df[cols_to_use].select_dtypes(include=["number"])
    if df_numeric.shape[1] == 0:
        print("No numeric features available for boxplot generation.")
        return

    feature_cols = df_numeric.columns.tolist()
    n_features = len(feature_cols)

    n_cols = 8
    n_rows = math.ceil(n_features / n_cols)

    plt.figure(figsize=(n_cols * 3, n_rows * 3))

    for idx, col in enumerate(feature_cols, start=1):
        ax = plt.subplot(n_rows, n_cols, idx)
        data = df_numeric[col].dropna()
        if data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
            ax.set_xticks([])
        else:
            ax.boxplot(data, vert=True)
            ax.set_xticks([])
        ax.set_title(col, fontsize=8)
        ax.tick_params(axis="y", labelsize=6)

    total_plots = n_rows * n_cols
    if total_plots > n_features:
        for idx in range(n_features + 1, total_plots + 1):
            ax = plt.subplot(n_rows, n_cols, idx)
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Boxplot grid image generated at: {output_path}")


def generate_r2_correlation_heatmap(
    df: pd.DataFrame,
    output_dir: str,
    exclude_cols=None,
    file_name: str = "feature_correlation_heatmap_r2.png"
):
    """R² heatmap (squared Pearson correlation) for numeric features."""
    if exclude_cols is None:
        exclude_cols = ["label", "image_name", "cell_index"]

    exclude_lower = {c.lower() for c in exclude_cols}
    cols_to_use = [c for c in df.columns if c.lower() not in exclude_lower]

    df_numeric = df[cols_to_use].select_dtypes(include=["number"])
    if df_numeric.shape[1] == 0:
        print("No numeric features available for correlation heatmap.")
        return

    feature_cols = df_numeric.columns.tolist()
    n_features = len(feature_cols)

    corr_matrix = df_numeric.corr(method="pearson")
    r2_matrix = corr_matrix ** 2

    base_size = max(10, n_features * 0.4)
    fig, ax = plt.subplots(figsize=(base_size, base_size))

    cax = ax.imshow(r2_matrix.values, interpolation="nearest", cmap="viridis")
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_title("R² Correlation Heatmap", fontsize=12)

    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_features))
    ax.set_xticklabels(feature_cols, rotation=90, fontsize=6)
    ax.set_yticklabels(feature_cols, fontsize=6)

    for i in range(n_features):
        for j in range(n_features):
            value = r2_matrix.values[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                fontsize=5,
                color="white" if value > 0.5 else "black"
            )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"R² correlation heatmap image generated at: {output_path}")


# ============================================================
# 3. CLASS BALANCE / UNDERSAMPLING
# ============================================================

def print_class_distribution(df: pd.DataFrame, label_col: str = "label"):
    if label_col not in df.columns:
        print(f"ERROR: Column '{label_col}' not found.")
        return

    class_counts = df[label_col].value_counts()
    class_percent = df[label_col].value_counts(normalize=True) * 100

    print("\n=== Class Distribution ===")
    for cls in class_counts.index:
        print(f"Class {cls}: {class_counts[cls]} records ({class_percent[cls]:.2f}%)")
    print("\nTotal records:", len(df))


def undersample_class(
    df: pd.DataFrame,
    target_class,
    retain_count: int,
    label_col: str = "label",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Undersample the specified target class to retain only retain_count rows.
    Other classes are left unchanged.
    """
    if label_col not in df.columns:
        print(f"ERROR: Column '{label_col}' not found.")
        return df

    df_target = df[df[label_col] == target_class]
    df_others = df[df[label_col] != target_class]

    current_count = len(df_target)
    if current_count < retain_count:
        print(
            f"WARNING: Class {target_class} has only {current_count} records; "
            f"retain_count={retain_count}. Returning original dataframe."
        )
        return df.copy()

    df_target_sampled = df_target.sample(
        n=retain_count, random_state=random_state, replace=False
    )

    df_balanced = pd.concat([df_target_sampled, df_others], axis=0).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    print(
        f"Undersampling: retained {retain_count} from class {target_class} "
        f"(original {current_count})."
    )
    return df_balanced


# ============================================================
# 4. STANDARDIZATION + PCA
# ============================================================

def standardize_features(
    df: pd.DataFrame,
    exclude_cols=None,
    label_col: str = "label"
):
    """
    Standardize numeric features (z-score). Returns dict with:
    - df_scaled
    - feature_cols
    - scaler
    """
    if exclude_cols is None:
        exclude_cols = ["label", "image_name", "cell_index"]

    exclude_lower = {c.lower() for c in exclude_cols}
    feature_cols = [
        c for c in df.columns
        if c.lower() not in exclude_lower and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(feature_cols) == 0:
        print("No numeric columns available for standardization.")
        return {}

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[feature_cols])

    df_scaled = pd.DataFrame(scaled_values, columns=feature_cols, index=df.index)
    if label_col in df.columns:
        df_scaled[label_col] = df[label_col].values

    print(f"Standardization completed. {len(feature_cols)} numeric features scaled.")
    return {"df_scaled": df_scaled, "feature_cols": feature_cols, "scaler": scaler}


def run_pca_on_standardized(
    df: pd.DataFrame,
    output_dir: str,
    exclude_cols=None,
    label_col: str = "label",
    n_components: int | None = None,
    explained_var_file: str = "pca_explained_variance.png",
    cumulative_var_file: str = "pca_cumulative_variance.png",
    loadings_file: str = "pca_loadings.csv",
    transformed_file: str = "pca_transformed.csv",
    variance_excel_file: str = "pca_variance_summary.xlsx"
):
    """
    Run PCA on standardized features and save:
    - explained variance plot
    - cumulative variance plot
    - loadings CSV
    - PCA-transformed CSV
    - variance tables to Excel
    """
    if exclude_cols is None:
        exclude_cols = ["label", "image_name", "cell_index"]

    exclude_lower = {c.lower() for c in exclude_cols}
    feature_cols = [
        c for c in df.columns
        if c.lower() not in exclude_lower and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(feature_cols) == 0:
        print("No numeric features available for PCA.")
        return {}

    X = df[feature_cols].values
    n_features = X.shape[1]

    if n_components is None or n_components > n_features:
        n_components = n_features

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
    if label_col in df.columns:
        pca_df[label_col] = df[label_col].values

    loadings = pd.DataFrame(pca.components_.T, index=feature_cols, columns=pca_cols)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    components = np.arange(1, n_components + 1)

    os.makedirs(output_dir, exist_ok=True)

    # Explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(components, explained_var, marker="o")
    plt.title("PCA: Explained Variance Ratio")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xticks(components)
    ev_path = os.path.join(output_dir, explained_var_file)
    plt.tight_layout()
    plt.savefig(ev_path, dpi=150)
    plt.close()

    # Cumulative variance
    plt.figure(figsize=(8, 5))
    plt.plot(components, cumulative_var, marker="o", color="green")
    plt.title("PCA: Cumulative Variance Ratio")
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Variance")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xticks(components)
    cv_path = os.path.join(output_dir, cumulative_var_file)
    plt.tight_layout()
    plt.savefig(cv_path, dpi=150)
    plt.close()

    loadings_path = os.path.join(output_dir, loadings_file)
    transformed_path = os.path.join(output_dir, transformed_file)
    loadings.to_csv(loadings_path)
    pca_df.to_csv(transformed_path, index=False)

    variance_summary = pd.DataFrame({
        "Component": components,
        "Explained Variance Ratio": explained_var,
        "Cumulative Variance Ratio": cumulative_var
    })

    excel_path = os.path.join(output_dir, variance_excel_file)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        variance_summary.to_excel(writer, sheet_name="PCA Summary", index=False)
        pd.DataFrame({"Explained Variance Ratio": explained_var}).to_excel(
            writer, sheet_name="Explained Variance", index=False
        )
        pd.DataFrame({"Cumulative Variance Ratio": cumulative_var}).to_excel(
            writer, sheet_name="Cumulative Variance", index=False
        )

    print("PCA completed. Files generated:")
    print(f" - Explained variance plot: {ev_path}")
    print(f" - Cumulative variance plot: {cv_path}")
    print(f" - PCA loadings: {loadings_path}")
    print(f" - PCA transformed data: {transformed_path}")
    print(f" - PCA variance Excel summary: {excel_path}")

    return {"pca_df": pca_df, "pca_model": pca, "loadings": loadings}


# ============================================================
# 5. MODEL DEFINITIONS
# ============================================================

def get_classification_models():
    """
    Returns a dict of models (LogReg, several KNN, several RF, several MLP).
    """
    models = {}

    models["LogisticRegression"] = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        n_jobs=-1
    )

    models["KNN_k3"] = KNeighborsClassifier(n_neighbors=3)
    models["KNN_k5"] = KNeighborsClassifier(n_neighbors=5)
    models["KNN_k7"] = KNeighborsClassifier(n_neighbors=7)
    models["KNN_k11"] = KNeighborsClassifier(n_neighbors=11)

    rf_settings = [
        ("RF_depth10_leaf1",   {"max_depth": 10, "min_samples_leaf": 1}),
        ("RF_depth10_leaf3",   {"max_depth": 10, "min_samples_leaf": 3}),
        ("RF_depth20_leaf1",   {"max_depth": 20, "min_samples_leaf": 1}),
        ("RF_depth20_leaf5",   {"max_depth": 20, "min_samples_leaf": 5}),
        ("RF_depthNone_leaf2", {"max_depth": None, "min_samples_leaf": 2}),
    ]
    for name, params in rf_settings:
        models[name] = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            **params
        )

    nn_settings = {
        "NN_1layer_32":        (32,),
        "NN_1layer_64":        (64,),
        "NN_2layers_32_16":    (32, 16),
        "NN_2layers_64_32":    (64, 32),
        "NN_3layers_32_16_8":  (32, 16, 8),
    }
    for name, structure in nn_settings.items():
        models[name] = MLPClassifier(
            hidden_layer_sizes=structure,
            activation="relu",
            max_iter=300,
            random_state=42
        )

    print("\nModels configured:", list(models.keys()))
    return models


# ============================================================
# 6. MODEL EVALUATION (PCA + NON-PCA)
# ============================================================

def _compute_split_metrics(model_name, split_name, classes, y_true, y_pred, y_proba):
    row = {"model": model_name, "dataset": split_name}

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    macro_prec = np.mean(prec)
    macro_rec = np.mean(rec)
    macro_f1 = np.mean(f1)

    auc_macro = roc_auc_score(
        y_true, y_proba, multi_class="ovr", average="macro"
    )

    y_true_bin = label_binarize(y_true, classes=classes)
    auc_per_class = []
    for i, cls in enumerate(classes):
        if len(np.unique(y_true_bin[:, i])) == 1:
            auc_c = np.nan
        else:
            auc_c = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
        auc_per_class.append(auc_c)

    row["overall_accuracy"] = acc
    row["macro_precision"] = macro_prec
    row["macro_recall"] = macro_rec
    row["macro_f1"] = macro_f1
    row["macro_auc"] = auc_macro

    for idx, cls in enumerate(classes):
        row[f"class_{cls}_precision"] = prec[idx]
        row[f"class_{cls}_recall"] = rec[idx]
        row[f"class_{cls}_f1"] = f1[idx]
        row[f"class_{cls}_auc"] = auc_per_class[idx]
        row[f"class_{cls}_support"] = support[idx]

    return row


def evaluate_models_with_metrics_and_plots(
    pca_train_df: pd.DataFrame,
    pca_test_df: pd.DataFrame,
    models: dict,
    output_dir: str,
    n_components: int = 29,
    label_col: str = "label",
    metrics_excel_file: str = "model_metrics_summary.xlsx",
    big_plot_file: str = "model_boundaries_and_roc.png",
):
    """
    Evaluate models on PCA features:
    - metrics to Excel
    - big image with decision boundaries + ROC curves
    """
    os.makedirs(output_dir, exist_ok=True)

    pc_cols_all = sorted(
        [c for c in pca_train_df.columns if c.startswith("PC")],
        key=lambda x: int(x.replace("PC", "")),
    )
    if len(pc_cols_all) == 0:
        print("No PCA columns found (PC1, PC2, ...).")
        return

    if n_components > len(pc_cols_all):
        n_components = len(pc_cols_all)

    pc_cols_keep = pc_cols_all[:n_components]
    pc_plot_cols = pc_cols_all[:2]

    X_train = pca_train_df[pc_cols_keep].values
    y_train = pca_train_df[label_col].values

    X_test = pca_test_df[pc_cols_keep].values
    y_test = pca_test_df[label_col].values

    classes = np.unique(y_train)
    X_train_2d = pca_train_df[pc_plot_cols].values if len(pc_plot_cols) >= 2 else None
    X_test_2d = pca_test_df[pc_plot_cols].values if len(pc_plot_cols) >= 2 else None

    metric_rows = []
    roc_curves = {"train": {}, "test": {}}

    for model_name, model in models.items():
        print(f"\n=== Training model (PCA): {model_name} ===")
        model.fit(X_train, y_train)

        if not hasattr(model, "predict_proba"):
            print(f"Model {model_name} lacks predict_proba; skipping.")
            continue

        y_train_proba = model.predict_proba(X_train)
        y_test_proba = model.predict_proba(X_test)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metric_rows.append(
            _compute_split_metrics(model_name, "train", classes, y_train, y_train_pred, y_train_proba)
        )
        metric_rows.append(
            _compute_split_metrics(model_name, "test", classes, y_test, y_test_pred, y_test_proba)
        )

        def macro_roc(y_true, y_proba):
            y_true_bin = label_binarize(y_true, classes=classes)
            fpr_dict, tpr_dict = {}, {}

            for i, cls in enumerate(classes):
                if len(np.unique(y_true_bin[:, i])) == 1:
                    continue
                fpr_dict[cls], tpr_dict[cls], _ = roc_curve(
                    y_true_bin[:, i], y_proba[:, i]
                )

            all_fpr = np.unique(
                np.concatenate([fpr for fpr in fpr_dict.values()])
            )
            mean_tpr = np.zeros_like(all_fpr)
            for cls in fpr_dict:
                mean_tpr += np.interp(all_fpr, fpr_dict[cls], tpr_dict[cls])
            mean_tpr /= len(fpr_dict)

            auc_macro = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
            return all_fpr, mean_tpr, auc_macro

        fpr_train, tpr_train, auc_train_macro = macro_roc(y_train, y_train_proba)
        fpr_test, tpr_test, auc_test_macro = macro_roc(y_test, y_test_proba)

        roc_curves["train"][model_name] = (fpr_train, tpr_train, auc_train_macro)
        roc_curves["test"][model_name] = (fpr_test, tpr_test, auc_test_macro)

    metrics_df = pd.DataFrame(metric_rows)
    excel_path = os.path.join(output_dir, metrics_excel_file)
    metrics_df.to_excel(excel_path, index=False, engine="openpyxl")
    print(f"\nPCA metrics Excel saved to: {excel_path}")

    n_models_plot = len(roc_curves["train"])
    if n_models_plot == 0 or X_train_2d is None:
        print("Skipping big PCA plot (no models with ROC or no 2D PCA).")
        return

    fig, axes = plt.subplots(n_models_plot, 3, figsize=(12, 4 * n_models_plot))
    if n_models_plot == 1:
        axes = np.expand_dims(axes, axis=0)

    cmap = plt.cm.get_cmap("tab10", len(classes))

    for row_idx, model_name in enumerate(roc_curves["train"].keys()):
        model = models[model_name]

        # decision boundary
        ax_boundary = axes[row_idx, 0]
        model_2d = clone(model)
        model_2d.fit(X_train_2d, y_train)

        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model_2d.predict(grid).reshape(xx.shape)

        ax_boundary.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
        for i, cls in enumerate(classes):
            m = y_train == cls
            ax_boundary.scatter(
                X_train_2d[m, 0], X_train_2d[m, 1],
                s=10, color=cmap(i), label=str(cls),
                edgecolor="k", linewidth=0.2,
            )
        ax_boundary.set_title(f"{model_name} - Decision Boundary")
        ax_boundary.set_xlabel("PC1")
        ax_boundary.set_ylabel("PC2")
        ax_boundary.legend(fontsize=6)

        # ROC train
        ax_roc_train = axes[row_idx, 1]
        fpr_tr, tpr_tr, auc_tr = roc_curves["train"][model_name]
        ax_roc_train.plot(fpr_tr, tpr_tr, label=f"AUC = {auc_tr:.4f}")
        ax_roc_train.plot([0, 1], [0, 1], "k--", linewidth=0.5)
        ax_roc_train.set_title("ROC Curve - Train")
        ax_roc_train.set_xlabel("FPR")
        ax_roc_train.set_ylabel("TPR")
        ax_roc_train.legend(loc="lower right", fontsize=7)

        # ROC test
        ax_roc_test = axes[row_idx, 2]
        fpr_te, tpr_te, auc_te = roc_curves["test"][model_name]
        ax_roc_test.plot(fpr_te, tpr_te, label=f"AUC = {auc_te:.4f}")
        ax_roc_test.plot([0, 1], [0, 1], "k--", linewidth=0.5)
        ax_roc_test.set_title("ROC Curve - Test")
        ax_roc_test.set_xlabel("FPR")
        ax_roc_test.set_ylabel("TPR")
        ax_roc_test.legend(loc="lower right", fontsize=7)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, big_plot_file)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"PCA decision-boundary + ROC plot saved to: {plot_path}")


def evaluate_models_on_standardized(
    df_train_std: pd.DataFrame,
    df_test_std: pd.DataFrame,
    models: dict,
    output_dir: str,
    feature_cols=None,
    label_col: str = "label",
    metrics_excel_file: str = "nonpca_model_metrics_summary.xlsx",
    big_plot_file: str = "nonpca_model_boundaries_and_roc.png",
    exclude_cols=None,
):
    """
    Evaluate models on NON-PCA standardized features
    (full feature space). Also makes 2D PCA projection for plotting.
    """
    os.makedirs(output_dir, exist_ok=True)

    if exclude_cols is None:
        exclude_cols = [label_col, "image_name", "cell_index"]

    exclude_lower = {c.lower() for c in exclude_cols}

    if feature_cols is None:
        feature_cols = [
            c for c in df_train_std.columns
            if c.lower() not in exclude_lower and pd.api.types.is_numeric_dtype(df_train_std[c])
        ]

    if len(feature_cols) == 0:
        print("No numeric feature columns found for standardized data.")
        return

    X_train = df_train_std[feature_cols].values
    y_train = df_train_std[label_col].values

    X_test = df_test_std[feature_cols].values
    y_test = df_test_std[label_col].values

    classes = np.unique(y_train)

    pca_2d = PCA(n_components=2, random_state=42)
    X_train_2d = pca_2d.fit_transform(X_train)
    X_test_2d = pca_2d.transform(X_test)

    metric_rows = []
    roc_curves = {"train": {}, "test": {}}

    for model_name, model in models.items():
        print(f"\n=== Training model (non-PCA): {model_name} ===")
        model.fit(X_train, y_train)

        if not hasattr(model, "predict_proba"):
            print(f"Model {model_name} has no predict_proba; skipping.")
            continue

        y_train_proba = model.predict_proba(X_train)
        y_test_proba = model.predict_proba(X_test)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metric_rows.append(
            _compute_split_metrics(model_name, "train", classes, y_train, y_train_pred, y_train_proba)
        )
        metric_rows.append(
            _compute_split_metrics(model_name, "test", classes, y_test, y_test_pred, y_test_proba)
        )

        def macro_roc(y_true, y_proba):
            y_true_bin = label_binarize(y_true, classes=classes)
            fpr_dict, tpr_dict = {}, {}
            for i, cls in enumerate(classes):
                if len(np.unique(y_true_bin[:, i])) == 1:
                    continue
                fpr_dict[cls], tpr_dict[cls], _ = roc_curve(
                    y_true_bin[:, i], y_proba[:, i]
                )
            all_fpr = np.unique(
                np.concatenate([fpr for fpr in fpr_dict.values()])
            )
            mean_tpr = np.zeros_like(all_fpr)
            for cls in fpr_dict:
                mean_tpr += np.interp(all_fpr, fpr_dict[cls], tpr_dict[cls])
            mean_tpr /= len(fpr_dict)
            auc_macro = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
            return all_fpr, mean_tpr, auc_macro

        fpr_train, tpr_train, auc_train_macro = macro_roc(y_train, y_train_proba)
        fpr_test, tpr_test, auc_test_macro = macro_roc(y_test, y_test_proba)

        roc_curves["train"][model_name] = (fpr_train, tpr_train, auc_train_macro)
        roc_curves["test"][model_name] = (fpr_test, tpr_test, auc_test_macro)

    metrics_df = pd.DataFrame(metric_rows)
    excel_path = os.path.join(output_dir, metrics_excel_file)
    metrics_df.to_excel(excel_path, index=False, engine="openpyxl")
    print(f"\nNon-PCA metrics Excel saved to: {excel_path}")

    n_models_plot = len(roc_curves["train"])
    if n_models_plot == 0:
        print("No non-PCA models with ROC data; skipping big plot.")
        return

    fig, axes = plt.subplots(n_models_plot, 3, figsize=(12, 4 * n_models_plot))
    if n_models_plot == 1:
        axes = np.expand_dims(axes, axis=0)

    cmap = plt.cm.get_cmap("tab10", len(classes))

    for row_idx, model_name in enumerate(roc_curves["train"].keys()):
        model = models[model_name]

        ax_boundary = axes[row_idx, 0]
        model_2d = clone(model)
        model_2d.fit(X_train_2d, y_train)

        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model_2d.predict(grid).reshape(xx.shape)

        ax_boundary.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
        for i, cls in enumerate(classes):
            m = y_train == cls
            ax_boundary.scatter(
                X_train_2d[m, 0], X_train_2d[m, 1],
                s=10, color=cmap(i), label=str(cls),
                edgecolor="k", linewidth=0.2,
            )
        ax_boundary.set_title(f"{model_name} - Decision Boundary (non-PCA)")
        ax_boundary.set_xlabel("PCA-2D PC1")
        ax_boundary.set_ylabel("PCA-2D PC2")
        ax_boundary.legend(fontsize=6)

        ax_roc_train = axes[row_idx, 1]
        fpr_tr, tpr_tr, auc_tr = roc_curves["train"][model_name]
        ax_roc_train.plot(fpr_tr, tpr_tr, label=f"AUC = {auc_tr:.4f}")
        ax_roc_train.plot([0, 1], [0, 1], "k--", linewidth=0.5)
        ax_roc_train.set_title("ROC Curve - Train (non-PCA)")
        ax_roc_train.set_xlabel("FPR")
        ax_roc_train.set_ylabel("TPR")
        ax_roc_train.legend(loc="lower right", fontsize=7)

        ax_roc_test = axes[row_idx, 2]
        fpr_te, tpr_te, auc_te = roc_curves["test"][model_name]
        ax_roc_test.plot(fpr_te, tpr_te, label=f"AUC = {auc_te:.4f}")
        ax_roc_test.plot([0, 1], [0, 1], "k--", linewidth=0.5)
        ax_roc_test.set_title("ROC Curve - Test (non-PCA)")
        ax_roc_test.set_xlabel("FPR")
        ax_roc_test.set_ylabel("TPR")
        ax_roc_test.legend(loc="lower right", fontsize=7)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, big_plot_file)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Non-PCA decision-boundary + ROC plot saved to: {plot_path}")


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def run_eda_pipeline(train_csv, test_csv, output_dir):
    """
    Run complete EDA pipeline on extracted features.

    Args:
        train_csv: Path to training features CSV
        test_csv: Path to test/validation features CSV
        output_dir: Directory for EDA outputs
    """
    print(f"Loading train from: {train_csv}")
    print(f"Loading test  from: {test_csv}")

    df_raw = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # 1) Missing + duplicates
    df_no_missing = check_missing_data(df_raw)
    report_duplicates(df_no_missing)

    # 2) Descriptive stats + visuals (original, imbalanced)
    df_features = generate_descriptive_analysis(df_no_missing, output_dir=output_dir)
    generate_feature_histograms(df_no_missing, output_dir=output_dir)
    generate_feature_boxplots(df_no_missing, output_dir=output_dir)
    # generate_r2_correlation_heatmap(df_no_missing, output_dir=output_dir) # Causes Memory Error for 1500 features

    # 3) Class distribution + undersampling majority class 0
    print_class_distribution(df_no_missing)
    df_balanced = undersample_class(df_no_missing, target_class=0, retain_count=2000)
    print_class_distribution(df_balanced)

    # 4) Visuals on balanced data
    balanced_output_dir = f"{output_dir}_bal"
    generate_feature_histograms(df_balanced, output_dir=balanced_output_dir)
    generate_feature_boxplots(df_balanced, output_dir=balanced_output_dir)
    # generate_r2_correlation_heatmap(df_balanced, output_dir=balanced_output_dir)  # Causes Memory Error for 1500 features

    # 5) Standardize balanced train
    std_result = standardize_features(df_balanced)
    df_standardized = std_result["df_scaled"]
    scaled_feature_cols = std_result["feature_cols"]
    scaler = std_result["scaler"]

    # 6) PCA on standardized train
    pca_result = run_pca_on_standardized(
        df=df_standardized,
        output_dir=output_dir
    )
    pca_train_df = pca_result["pca_df"]
    pca_model = pca_result["pca_model"]

    # 7) Prepare standardized + PCA test using SAME scaler + PCA
    df_test_features = df_test[scaled_feature_cols]
    test_scaled_values = scaler.transform(df_test_features)
    df_test_standardized = pd.DataFrame(
        test_scaled_values,
        columns=scaled_feature_cols,
        index=df_test.index
    )
    if "label" in df_test.columns:
        df_test_standardized["label"] = df_test["label"].values

    test_pca = pca_model.transform(df_test_standardized[scaled_feature_cols].values)
    pca_cols = [f"PC{i+1}" for i in range(pca_model.n_components_)]
    pca_test_df = pd.DataFrame(test_pca, columns=pca_cols, index=df_test.index)
    if "label" in df_test.columns:
        pca_test_df["label"] = df_test["label"].values

    # 8) Models
    models = get_classification_models()

    # 9) Evaluate on PCA features
    evaluate_models_with_metrics_and_plots(
        pca_train_df=pca_train_df,
        pca_test_df=pca_test_df,
        models=models,
        output_dir=output_dir,
        n_components=29,
        label_col="label",
        metrics_excel_file="model_metrics_summary.xlsx",
        big_plot_file="model_boundaries_and_roc.png"
    )

    # 10) Evaluate on non-PCA standardized features
    evaluate_models_on_standardized(
        df_train_std=df_standardized,
        df_test_std=df_test_standardized,
        models=models,
        output_dir=output_dir,
        label_col="label",
        metrics_excel_file="nonpca_model_metrics_summary.xlsx",
        big_plot_file="nonpca_model_boundaries_and_roc.png"
    )


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description='EDA for Cricket Object Detection Features')
    parser.add_argument('--train_features', type=str,
                       default='outputs/train_features.csv',
                       help='Path to training features CSV')
    parser.add_argument('--validate_features', type=str,
                       default='outputs/validate_features.csv',
                       help='Path to validation features CSV')
    parser.add_argument('--output_dir', type=str,
                       default='eda_reports',
                       help='Directory for EDA outputs')

    args = parser.parse_args()

    # Call the extracted pipeline function
    run_eda_pipeline(
        train_csv=args.train_features,
        test_csv=args.validate_features,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
