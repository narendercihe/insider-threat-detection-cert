from pathlib import Path
import pandas as pd

from src.data_loader import load_all_data  # Changed to load_all_data
from src.preprocess import preprocess_logon_data, preprocess_psychometric_data
from src.features import build_daily_user_features, merge_psychometric_features
from src.model import train_isolation_forest, save_model
from src.evaluate import (
    save_top_anomalies,
    plot_hour_distribution,
    plot_top_suspicious_users,
    print_summary,
)


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"
MODEL_DIR = BASE_DIR / "outputs" / "models"


def ensure_dirs() -> None:
    for folder in [PROCESSED_DIR, FIGURES_DIR, PREDICTIONS_DIR, MODEL_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs()

    # Load all datasets using load_all_data
    print("Loading data...")
    data = load_all_data(RAW_DIR)

    logon_df = data.get("logon")
    psychometric_df = data.get("psychometric")

    use_psychometric = psychometric_df is not None

    if use_psychometric:
        print("psychometric.csv found and loaded.")
    else:
        print("psychometric.csv not found. Running project without psychometric data.")

    print("Preprocessing data...")
    logon_df = preprocess_logon_data(logon_df)

    if use_psychometric:
        psychometric_df = preprocess_psychometric_data(psychometric_df)

    print("Building features...")
    daily_features = build_daily_user_features(logon_df)

    if use_psychometric and psychometric_df is not None:
        final_features = merge_psychometric_features(daily_features, psychometric_df)
    else:
        final_features = daily_features.copy()

    processed_path = PROCESSED_DIR / "daily_user_features.csv"
    final_features.to_csv(processed_path, index=False)
    print(f"Saved processed features to: {processed_path}")

    print("Training model...")
    model, results = train_isolation_forest(final_features)

    results_path = PREDICTIONS_DIR / "anomaly_results.csv"
    results.to_csv(results_path, index=False)
    print(f"Saved anomaly results to: {results_path}")

    top_path = PREDICTIONS_DIR / "top_anomalies.csv"
    top_df = save_top_anomalies(results, top_path, top_n=25)
    print(f"Saved top anomalies to: {top_path}")

    model_path = MODEL_DIR / "isolation_forest.joblib"
    save_model(model, model_path)
    print(f"Saved model to: {model_path}")

    print("Generating plots...")
    plot_hour_distribution(logon_df, FIGURES_DIR / "hour_distribution.png")
    plot_top_suspicious_users(results, FIGURES_DIR / "top_suspicious_users.png")

    print_summary(results)

    print("\nTop 10 most suspicious user-day records:")
    display_cols = [
        "user",
        "day",
        "total_events",
        "after_hours_events",
        "unique_pcs",
        "after_hours_ratio",
        "anomaly_score",
    ]

    available_cols = [col for col in display_cols if col in top_df.columns]
    print(top_df[available_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
