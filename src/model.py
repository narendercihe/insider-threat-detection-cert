from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest


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
    "O",
    "C",
    "E",
    "A",
    "N",
]


def train_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42,
) -> tuple[IsolationForest, pd.DataFrame]:
    """
    Train Isolation Forest and return predictions.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[FEATURE_COLUMNS].copy()

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)

    results = df.copy()
    results["anomaly_label"] = model.predict(X)   # -1 anomaly, 1 normal
    results["anomaly_score"] = model.decision_function(X)
    results["is_anomaly"] = results["anomaly_label"].eq(-1)

    return model, results


def save_model(model: IsolationForest, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
