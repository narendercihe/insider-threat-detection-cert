import os
import json
import warnings
import importlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
AE_ARTIFACTS_DIR = ARTIFACTS_DIR / "autoencoder"
VAE_ARTIFACTS_DIR = ARTIFACTS_DIR / "vae"

for d in [
    PROCESSED_DIR,
    FIGURES_DIR,
    PREDICTIONS_DIR,
    TABLES_DIR,
    MODELS_DIR,
    AE_ARTIFACTS_DIR,
    VAE_ARTIFACTS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(message, flush=True)


def import_module(module_name: str):
    return importlib.import_module(module_name)


def resolve_callable(module_name: str, candidate_names: list[str]):
    module = import_module(module_name)
    for name in candidate_names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise AttributeError(
        f"None of these functions were found in {module_name}: {candidate_names}"
    )


def resolve_callable_fuzzy(
    module_name: str,
    candidate_names: list[str],
    include_keywords: list[str],
    exclude_keywords: list[str] | None = None,
):
    module = import_module(module_name)

    for name in candidate_names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn

    exclude_keywords = exclude_keywords or []
    callables = []

    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if not callable(obj):
            continue

        lname = name.lower()
        if all(k.lower() in lname for k in include_keywords) and not any(
            bad.lower() in lname for bad in exclude_keywords
        ):
            callables.append((name, obj))

    if len(callables) == 1:
        log(f"Auto-detected function in {module_name}: {callables[0][0]}")
        return callables[0][1]

    if len(callables) > 1:
        preferred = sorted(callables, key=lambda x: len(x[0]))[0]
        log(f"Auto-detected function in {module_name}: {preferred[0]}")
        return preferred[1]

    available = [name for name in dir(module) if callable(getattr(module, name, None)) and not name.startswith("_")]
    raise AttributeError(
        f"Could not resolve function in {module_name}. "
        f"Tried explicit names: {candidate_names}. "
        f"Available callables: {available}"
    )


def ensure_label_column(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(
            f"Expected '{label_col}' column in labeled dataframe, but it was not found."
        )
    return df


def get_numeric_feature_columns(df: pd.DataFrame, label_col: str = "label") -> list[str]:
    exclude = {
        label_col,
        "user",
        "date",
        "session_id",
        "pc",
        "start_time",
        "end_time",
        "timestamp",
    }
    return [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def normalize_model_output(result, model_name: str) -> dict:
    out = {
        "scores": None,
        "y_pred": None,
        "threshold": None,
        "model": None,
    }

    if isinstance(result, dict):
        out["scores"] = result.get("scores")
        out["y_pred"] = result.get("y_pred", result.get("predictions"))
        out["threshold"] = result.get("threshold")
        out["model"] = result.get("model")
        return out

    if isinstance(result, (tuple, list)):
        if len(result) == 4:
            out["scores"], out["y_pred"], out["threshold"], out["model"] = result
            return out
        if len(result) == 3:
            a, b, c = result
            arr_a = np.asarray(a)
            if arr_a.ndim == 1 and set(np.unique(arr_a)).issubset({0, 1}):
                out["y_pred"], out["scores"], out["model"] = a, b, c
            else:
                out["scores"], out["y_pred"], out["threshold"] = a, b, c
            return out
        if len(result) == 2:
            out["y_pred"], out["scores"] = result
            return out

    raise ValueError(f"Unsupported output format from {model_name}: {type(result)}")


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
    plt.xticks(tick_marks, ["Normal", "Anomaly"])
    plt.yticks(tick_marks, ["Normal", "Anomaly"])
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
    scores,
    model_name: str,
    save_path: Path,
) -> None:
    temp = df.copy()
    temp["true_label"] = y_true
    temp[f"{model_name.lower()}_pred"] = y_pred
    if scores is not None:
        temp[f"{model_name.lower()}_score"] = np.asarray(scores)
    temp.to_csv(save_path, index=False)


def main():
    log("Loading datasets...")

    load_all_data = resolve_callable_fuzzy(
        "src.data_loader",
        ["load_all_data", "load_data", "load_raw_data"],
        include_keywords=["load"],
    )

    datasets = load_all_data(str(RAW_DIR))
    if datasets is None:
        raise ValueError("load_all_data() returned None. Please check src/data_loader.py")

    log("Preprocessing datasets...")

    try:
        preprocess_fn = resolve_callable_fuzzy(
            "src.preprocess",
            ["preprocess_all_data", "preprocess_data", "preprocess_datasets"],
            include_keywords=["preprocess"],
        )
        datasets = preprocess_fn(datasets)
    except Exception:
        pass

    log("Building features...")

    build_features = resolve_callable_fuzzy(
        "src.features",
        [
            "build_features",
            "build_feature_table",
            "build_daily_features",
            "engineer_features",
            "create_feature_table",
            "extract_features",
            "make_features",
            "generate_features",
            "create_features",
            "prepare_features",
            "build_user_features",
            "build_dataset_features",
        ],
        include_keywords=["feature"],
        exclude_keywords=["plot", "save", "load", "visual", "chart"],
    )

    feature_df = build_features(datasets)

    if not isinstance(feature_df, pd.DataFrame):
        raise ValueError("Feature builder did not return a pandas DataFrame.")

    if feature_df.empty:
        raise ValueError("Feature dataframe is empty.")

    feature_table_path = PROCESSED_DIR / "final_feature_table.csv"
    feature_df.to_csv(feature_table_path, index=False)
    log(f"Saved feature table to: {feature_table_path}")

    log("Building proxy labels...")

    label_fn = resolve_callable_fuzzy(
        "src.labels",
        [
            "build_proxy_labels",
            "create_proxy_labels",
            "generate_proxy_labels",
            "assign_proxy_labels",
            "apply_proxy_labels",
            "make_labels",
            "build_labels",
            "generate_labels",
            "create_labels",
        ],
        include_keywords=["label"],
        exclude_keywords=["plot", "save", "load"],
    )

    labeled_df = label_fn(feature_df)

    if not isinstance(labeled_df, pd.DataFrame):
        raise ValueError("Label builder did not return a pandas DataFrame.")

    labeled_df = ensure_label_column(labeled_df, label_col="label")

    labeled_feature_table_path = PROCESSED_DIR / "final_feature_table_with_labels.csv"
    labeled_df.to_csv(labeled_feature_table_path, index=False)
    log(f"Saved labeled feature table to: {labeled_feature_table_path}")

    feature_cols = get_numeric_feature_columns(labeled_df, label_col="label")
    if not feature_cols:
        raise ValueError("No numeric feature columns found for modeling.")

    y_true = labeled_df["label"].astype(int).values
    results_summary = []

    log("Training Isolation Forest...")

    train_iforest = resolve_callable_fuzzy(
        "src.baseline_iforest",
        [
            "train_isolation_forest",
            "run_isolation_forest",
            "train_iforest",
            "fit_isolation_forest",
            "build_isolation_forest",
        ],
        include_keywords=["forest"],
        exclude_keywords=["plot", "save", "load"],
    )

    iforest_result = train_iforest(labeled_df, feature_cols=feature_cols, label_col="label")
    iforest_out = normalize_model_output(iforest_result, "Isolation Forest")
    iforest_y_pred = np.asarray(iforest_out["y_pred"]).astype(int)

    iforest_metrics = evaluate_predictions(y_true, iforest_y_pred)
    results_summary.append(
        {
            "model": "Isolation Forest",
            "accuracy": iforest_metrics["accuracy"],
            "precision": iforest_metrics["precision"],
            "recall": iforest_metrics["recall"],
            "f1": iforest_metrics["f1"],
            "threshold": iforest_out["threshold"],
        }
    )

    save_predictions_file(
        labeled_df,
        y_true,
        iforest_y_pred,
        iforest_out["scores"],
        "iforest",
        PREDICTIONS_DIR / "iforest_predictions.csv",
    )

    save_confusion_matrix_plot(
        np.array(iforest_metrics["confusion_matrix"]),
        "Isolation Forest Confusion Matrix",
        FIGURES_DIR / "iforest_confusion_matrix.png",
    )

    if iforest_out["model"] is not None:
        joblib.dump(iforest_out["model"], MODELS_DIR / "isolation_forest.joblib")

    with open(TABLES_DIR / "iforest_metrics.json", "w") as f:
        json.dump(iforest_metrics, f, indent=2)

    log("Training Autoencoder...")

    train_ae = resolve_callable_fuzzy(
        "src.autoencoder_model",
        [
            "train_autoencoder",
            "run_autoencoder",
            "fit_autoencoder",
            "train_ae",
            "build_autoencoder",
        ],
        include_keywords=["autoencoder"],
        exclude_keywords=["plot", "save", "load"],
    )

    ae_result = train_ae(
        labeled_df,
        feature_cols=feature_cols,
        label_col="label",
        model_dir=str(AE_ARTIFACTS_DIR),
    )
    ae_out = normalize_model_output(ae_result, "Autoencoder")
    ae_y_pred = np.asarray(ae_out["y_pred"]).astype(int)

    ae_metrics = evaluate_predictions(y_true, ae_y_pred)
    results_summary.append(
        {
            "model": "Autoencoder",
            "accuracy": ae_metrics["accuracy"],
            "precision": ae_metrics["precision"],
            "recall": ae_metrics["recall"],
            "f1": ae_metrics["f1"],
            "threshold": ae_out["threshold"],
        }
    )

    save_predictions_file(
        labeled_df,
        y_true,
        ae_y_pred,
        ae_out["scores"],
        "autoencoder",
        PREDICTIONS_DIR / "autoencoder_predictions.csv",
    )

    save_confusion_matrix_plot(
        np.array(ae_metrics["confusion_matrix"]),
        "Autoencoder Confusion Matrix",
        FIGURES_DIR / "autoencoder_confusion_matrix.png",
    )

    with open(TABLES_DIR / "autoencoder_metrics.json", "w") as f:
        json.dump(ae_metrics, f, indent=2)

    log("Training VAE...")

    train_vae = resolve_callable("src.vae_model", ["train_vae"])

    vae_result = train_vae(
        labeled_df,
        feature_cols=feature_cols,
        label_col="label",
        model_dir=str(VAE_ARTIFACTS_DIR),
    )
    vae_out = normalize_model_output(vae_result, "VAE")
    vae_y_pred = np.asarray(vae_out["y_pred"]).astype(int)

    vae_metrics = evaluate_predictions(y_true, vae_y_pred)
    results_summary.append(
        {
            "model": "VAE",
            "accuracy": vae_metrics["accuracy"],
            "precision": vae_metrics["precision"],
            "recall": vae_metrics["recall"],
            "f1": vae_metrics["f1"],
            "threshold": vae_out["threshold"],
        }
    )

    save_predictions_file(
        labeled_df,
        y_true,
        vae_y_pred,
        vae_out["scores"],
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

    comparison_df = pd.DataFrame(results_summary)
    comparison_df = comparison_df[
        ["model", "accuracy", "precision", "recall", "f1", "threshold"]
    ]

    comparison_csv_path = TABLES_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)

    try:
        comparison_xlsx_path = TABLES_DIR / "model_comparison.xlsx"
        comparison_df.to_excel(comparison_xlsx_path, index=False)
    except Exception:
        comparison_xlsx_path = None

    save_metric_comparison_plot(
        comparison_df,
        FIGURES_DIR / "model_comparison.png",
    )

    log("\nDone.")
    log(f"Feature table: {feature_table_path}")
    log(f"Labeled feature table: {labeled_feature_table_path}")
    log(f"Comparison CSV: {comparison_csv_path}")
    if comparison_xlsx_path:
        log(f"Comparison Excel: {comparison_xlsx_path}")
    log(f"Figures folder: {FIGURES_DIR}")
    log(f"Predictions folder: {PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()
