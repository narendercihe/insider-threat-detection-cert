from __future__ import annotations
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_binary_model(y_true, y_pred, model_name: str) -> dict:
    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ConfusionMatrix": confusion_matrix(y_true, y_pred),
    }


def build_comparison_table(result_df: pd.DataFrame) -> pd.DataFrame:
    y_true = result_df["label"].astype(int)

    results = [
        evaluate_binary_model(y_true, result_df["iforest_pred"].astype(int), "Isolation Forest"),
        evaluate_binary_model(y_true, result_df["ae_pred"].astype(int), "Autoencoder"),
        evaluate_binary_model(y_true, result_df["vae_pred"].astype(int), "VAE"),
    ]

    table = pd.DataFrame(results)
    return table[["Model", "Accuracy", "Precision", "Recall", "F1-score"]]
