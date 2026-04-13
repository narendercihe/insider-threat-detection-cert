from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.data_loader import load_all_data
from src.preprocess import (
    preprocess_logon,
    preprocess_device,
    preprocess_psychometric,
    preprocess_users,
)
from src.features import (
    build_logon_features,
    build_device_features,
    build_final_feature_table,
)
from src.label_builder import build_pseudo_labels
from src.baseline_iforest import train_iforest, save_iforest_artifacts
from src.autoencoder_model import train_autoencoder, save_autoencoder_artifacts
from src.vae_model import train_vae

warnings.filterwarnings("ignore")


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
IFOREST_ARTIFACTS_DIR = ARTIFACTS_DIR / "iforest"
AE_ARTIFACTS_DIR = ARTIFACTS_DIR / "autoencoder"
VAE_ARTIFACTS_DIR = ARTIFACTS_DIR / "vae"

for d in [
    PROCESSED_DIR,
    FIGURES_DIR,
    PREDICTIONS_DIR,
    TABLES_DIR,
    MODELS_DIR,
    IFOREST_ARTIFACTS_DIR,
    AE_ARTIFACTS_DIR,
    VAE_ARTIFACTS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(message, flush=True)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix_plot(cm: np.ndarray, title: str, save_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Normal", "Suspicious"])
    plt.yticks(tick_marks, ["Normal", "Suspicious"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_metric_comparison_plot(df_metrics: pd.DataFrame, save_path: Path) -> None:
    metric_cols = ["accuracy", "precision", "recall", "f1"]
    plot_df = df_metrics.set_index("model")[metric_cols]

    ax = plot_df.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_predictions_file(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None,
    model_name: str,
    save_path: Path,
) -> None:
    temp = df.copy()
    temp["true_label"] = y_true
    temp[f"{model_name}_pred"] = y_pred
    if scores is not None:
        temp[f"{model_name}_score"] = scores
    temp.to_csv(save_path, index=False)


def main():
    # -----------------------------
    # Load raw data
    # -----------------------------
    log("Loading datasets...")
    datasets = load_all_data(RAW_DIR)

    # -----------------------------
    # Preprocess
    # -----------------------------
    log("Preprocessing datasets...")
    logon_df = preprocess_logon(datasets["logon"])
    device_df = preprocess_device(datasets["device"])
    psychometric_df = preprocess_psychometric(datasets["psychometric"])
    users_df = preprocess_users(datasets.get("users"))

    # -----------------------------
    # Build features
    # -----------------------------
    log("Building features...")
    logon_features = build_logon_features(logon_df)
    device_features = build_device_features(device_df)

    feature_df = build_final_feature_table(
        logon_features=logon_features,
        device_features=device_features,
        psychometric_df=psychometric_df,
        users_df=users_df,
    )

    if not isinstance(feature_df, pd.DataFrame):
        raise ValueError("Feature builder did not return a pandas DataFrame.")

    if feature_df.empty:
        raise ValueError("Feature dataframe is empty.")

    feature_table_path = PROCESSED_DIR / "final_feature_table.csv"
    feature_df.to_csv(feature_table_path, index=False)
    log(f"Saved feature table to: {feature_table_path}")

    # -----------------------------
    # Build pseudo labels
    # -----------------------------
    log("Building proxy labels...")
    labeled_df = build_pseudo_labels(feature_df)

    if "label" not in labeled_df.columns:
        raise ValueError("Label column was not created.")

    labeled_feature_table_path = PROCESSED_DIR / "final_feature_table_with_labels.csv"
    labeled_df.to_csv(labeled_feature_table_path, index=False)
    log(f"Saved labeled feature table to: {labeled_feature_table_path}")

    y_true = labeled_df["label"].astype(int).values
    results_summary = []

    # -----------------------------
    # Isolation Forest
    # -----------------------------
    log("Training Isolation Forest...")
    if_scaler, if_model, if_result_df = train_iforest(labeled_df)

    iforest_y_pred = if_result_df["iforest_pred"].astype(int).values
    iforest_scores = if_result_df["iforest_score"].astype(float).values

    iforest_metrics = evaluate_predictions(y_true, iforest_y_pred)
    results_summary.append(
        {
            "model": "Isolation Forest",
            "accuracy": iforest_metrics["accuracy"],
            "precision": iforest_metrics["precision"],
            "recall": iforest_metrics["recall"],
            "f1": iforest_metrics["f1"],
            "threshold": np.nan,
        }
    )

    save_predictions_file(
        labeled_df,
        y_true,
        iforest_y_pred,
        iforest_scores,
        "iforest",
        PREDICTIONS_DIR / "iforest_predictions.csv",
    )

    save_confusion_matrix_plot(
        np.array(iforest_metrics["confusion_matrix"]),
        "Isolation Forest Confusion Matrix",
        FIGURES_DIR / "iforest_confusion_matrix.png",
    )

    save_iforest_artifacts(if_scaler, if_model, IFOREST_ARTIFACTS_DIR)

    with open(TABLES_DIR / "iforest_metrics.json", "w") as f:
        json.dump(iforest_metrics, f, indent=2)

    # -----------------------------
    # Autoencoder
    # -----------------------------
    log("Training Autoencoder...")
    ae_scaler, ae_model, ae_threshold, ae_result_df = train_autoencoder(labeled_df)

    ae_y_pred = ae_result_df["ae_pred"].astype(int).values
    ae_scores = ae_result_df["ae_score"].astype(float).values

    ae_metrics = evaluate_predictions(y_true, ae_y_pred)
    results_summary.append(
        {
            "model": "Autoencoder",
            "accuracy": ae_metrics["accuracy"],
            "precision": ae_metrics["precision"],
            "recall": ae_metrics["recall"],
            "f1": ae_metrics["f1"],
            "threshold": float(ae_threshold),
        }
    )

    save_predictions_file(
        labeled_df,
        y_true,
        ae_y_pred,
        ae_scores,
        "autoencoder",
        PREDICTIONS_DIR / "autoencoder_predictions.csv",
    )

    save_confusion_matrix_plot(
        np.array(ae_metrics["confusion_matrix"]),
        "Autoencoder Confusion Matrix",
        FIGURES_DIR / "autoencoder_confusion_matrix.png",
    )

    save_autoencoder_artifacts(ae_scaler, ae_model, ae_threshold, AE_ARTIFACTS_DIR)

    with open(TABLES_DIR / "autoencoder_metrics.json", "w") as f:
        json.dump(ae_metrics, f, indent=2)

    # -----------------------------
    # VAE
    # -----------------------------
    log("Training VAE...")
    feature_cols = [
        c for c in labeled_df.columns
        if c not in {"label", "user", "day"}
        and pd.api.types.is_numeric_dtype(labeled_df[c])
    ]

    vae_scores, vae_y_pred, vae_threshold, _ = train_vae(
        labeled_df=labeled_df,
        feature_cols=feature_cols,
        label_col="label",
        model_dir=str(VAE_ARTIFACTS_DIR),
    )

    vae_y_pred = np.asarray(vae_y_pred).astype(int)
    vae_scores = np.asarray(vae_scores).astype(float)

    vae_metrics = evaluate_predictions(y_true, vae_y_pred)
    results_summary.append(
        {
            "model": "VAE",
            "accuracy": vae_metrics["accuracy"],
            "precision": vae_metrics["precision"],
            "recall": vae_metrics["recall"],
            "f1": vae_metrics["f1"],
            "threshold": float(vae_threshold),
        }
    )

    save_predictions_file(
        labeled_df,
        y_true,
        vae_y_pred,
        vae_scores,
        "vae",
        PREDICTIONS_DIR / "vae_predictions.csv",
    )

    save_confusion_matrix_plot(
        np.array(vae_metrics["confusion_matrix"]),
        "VAE Confusion Matrix",
        FIGURES_DIR / "vae_confusion_matrix.png",
    )

    with open(TABLES_DIR / "vae_metrics.json", "w") as f:
        json.dump(vae_metrics, f, indent=2)

    # -----------------------------
    # Final comparison
    # -----------------------------
    comparison_df = pd.DataFrame(results_summary)
    comparison_csv_path = TABLES_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)

    comparison_xlsx_path = None
    try:
        comparison_xlsx_path = TABLES_DIR / "model_comparison.xlsx"
        comparison_df.to_excel(comparison_xlsx_path, index=False)
    except Exception:
        pass

    save_metric_comparison_plot(
        comparison_df,
        FIGURES_DIR / "model_comparison.png",
    )

    log("\nDone.")
    log(f"Feature table: {feature_table_path}")
    log(f"Labeled feature table: {labeled_feature_table_path}")
    log(f"Comparison CSV: {comparison_csv_path}")
    if comparison_xlsx_path is not None:
        log(f"Comparison Excel: {comparison_xlsx_path}")
    log(f"Figures folder: {FIGURES_DIR}")
    log(f"Predictions folder: {PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()
