from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.data_loader import load_all_data
from src.preprocess import preprocess_logon, preprocess_device, preprocess_psychometric, preprocess_users
from src.features import build_logon_features, build_device_features, build_final_feature_table
from src.label_builder import build_pseudo_labels
from src.baseline_iforest import train_iforest, save_iforest_artifacts
from src.autoencoder_model import train_autoencoder, save_autoencoder_artifacts
from src.vae_model import train_vae, save_vae_artifacts
from src.metrics import build_comparison_table
from src.visualize import (
    plot_label_distribution,
    plot_metric_comparison,
    plot_confusion,
    plot_score_histogram,
)


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
MODELS_DIR = BASE_DIR / "outputs" / "models"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"
TABLES_DIR = BASE_DIR / "outputs" / "tables"


def ensure_dirs():
    for folder in [PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, PREDICTIONS_DIR, TABLES_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def main():
    ensure_dirs()

    print("Loading datasets...")
    datasets = load_all_data(RAW_DIR)

    print("Preprocessing datasets...")
    logon_df = preprocess_logon(datasets["logon"])
    device_df = preprocess_device(datasets["device"])
    psychometric_df = preprocess_psychometric(datasets["psychometric"])
    users_df = preprocess_users(datasets["users"])

    print("Building features...")
    logon_features = build_logon_features(logon_df)
    device_features = build_device_features(device_df)
    feature_df = build_final_feature_table(
        logon_features=logon_features,
        device_features=device_features,
        psychometric_df=psychometric_df,
        users_df=users_df,
    )

    print("Building pseudo labels...")
    labeled_df = build_pseudo_labels(feature_df)

    processed_path = PROCESSED_DIR / "final_feature_table_with_labels.csv"
    labeled_df.to_csv(processed_path, index=False)
    print(f"Saved processed feature table to: {processed_path}")

    print("Training Isolation Forest...")
    if_scaler, if_model, if_df = train_iforest(labeled_df, contamination=0.10)
    save_iforest_artifacts(if_scaler, if_model, MODELS_DIR)

    print("Training Autoencoder...")
    ae_scaler, ae_model, ae_threshold, ae_df = train_autoencoder(labeled_df)
    save_autoencoder_artifacts(ae_scaler, ae_model, ae_threshold, MODELS_DIR)

    print("Training VAE...")
    vae_scaler, vae_model, vae_threshold, vae_df = train_vae(labeled_df)
    save_vae_artifacts(vae_scaler, vae_model, vae_threshold, MODELS_DIR)

    print("Combining predictions...")
    result_df = labeled_df.copy()
    result_df["iforest_score"] = if_df["iforest_score"]
    result_df["iforest_pred"] = if_df["iforest_pred"]
    result_df["ae_score"] = ae_df["ae_score"]
    result_df["ae_pred"] = ae_df["ae_pred"]
    result_df["vae_score"] = vae_df["vae_score"]
    result_df["vae_pred"] = vae_df["vae_pred"]

    predictions_path = PREDICTIONS_DIR / "model_predictions.csv"
    result_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to: {predictions_path}")

    print("Evaluating models...")
    comparison_table = build_comparison_table(result_df)
    comparison_path = TABLES_DIR / "model_comparison.csv"
    comparison_table.to_csv(comparison_path, index=False)
    print(f"Saved comparison table to: {comparison_path}")

    print("\n=== Model Comparison ===")
    print(comparison_table.to_string(index=False))

    print("Generating figures...")
    y_true = result_df["label"].astype(int)

    plot_label_distribution(result_df, FIGURES_DIR / "label_distribution.png")
    plot_metric_comparison(comparison_table, FIGURES_DIR / "model_comparison.png")

    plot_confusion(y_true, result_df["iforest_pred"], "Isolation Forest Confusion Matrix",
                   FIGURES_DIR / "iforest_confusion_matrix.png")
    plot_confusion(y_true, result_df["ae_pred"], "Autoencoder Confusion Matrix",
                   FIGURES_DIR / "ae_confusion_matrix.png")
    plot_confusion(y_true, result_df["vae_pred"], "VAE Confusion Matrix",
                   FIGURES_DIR / "vae_confusion_matrix.png")

    plot_score_histogram(result_df, "iforest_score", "Isolation Forest Score Distribution",
                         FIGURES_DIR / "iforest_score_distribution.png")
    plot_score_histogram(result_df, "ae_score", "Autoencoder Reconstruction Error",
                         FIGURES_DIR / "ae_score_distribution.png")
    plot_score_histogram(result_df, "vae_score", "VAE Reconstruction Error",
                         FIGURES_DIR / "vae_score_distribution.png")

    print("Done.")


if __name__ == "__main__":
    main()
