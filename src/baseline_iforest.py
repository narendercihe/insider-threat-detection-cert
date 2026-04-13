from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "total_events",
    "total_logons",
    "total_logoffs",
    "unique_pcs",
    "after_hours_events",
    "weekend_events",
    "first_hour",
    "last_hour",
    "activity_span",
    "after_hours_ratio",
    "weekend_ratio",
    "device_events",
    "after_hours_device_events",
    "weekend_device_events",
    "unique_device_pcs",
    "connect_like_events",
    "after_hours_device_ratio",
    "weekend_device_ratio",
    "user_avg_total_events",
    "user_avg_after_hours",
    "user_avg_unique_pcs",
    "user_avg_device_events",
    "deviation_total_events",
    "deviation_after_hours",
    "deviation_unique_pcs",
    "deviation_device_events",
    "O",
    "C",
    "E",
    "A",
    "N",
]


def train_iforest(df: pd.DataFrame, contamination: float = 0.10, random_state: int = 42):
    X = df[FEATURE_COLUMNS].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_scaled)

    scores = model.decision_function(X_scaled)
    preds = model.predict(X_scaled)  # -1 anomaly, 1 normal

    result_df = df.copy()
    result_df["iforest_score"] = scores
    result_df["iforest_pred"] = (preds == -1).astype(int)

    return scaler, model, result_df


def save_iforest_artifacts(scaler, model, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_dir / "iforest_scaler.joblib")
    joblib.dump(model, output_dir / "iforest_model.joblib")
