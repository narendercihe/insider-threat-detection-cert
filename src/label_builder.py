from __future__ import annotations
import pandas as pd


def build_pseudo_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create strong rule-based pseudo labels.
    label = 1 means suspicious.
    """

    data = df.copy()

    # Adaptive thresholds
    ah_threshold = max(2, int(data["after_hours_events"].quantile(0.90)))
    device_threshold = max(1, int(data["device_events"].quantile(0.85)))
    pc_threshold = max(2, int(data["unique_pcs"].quantile(0.90)))
    dev_total_threshold = max(2, int(data["deviation_total_events"].quantile(0.90)))
    dev_ah_threshold = max(1, int(data["deviation_after_hours"].quantile(0.90)))

    conditions = pd.DataFrame({
        "high_after_hours": data["after_hours_events"] >= ah_threshold,
        "high_after_hours_ratio": data["after_hours_ratio"] >= 0.40,
        "high_device_usage": data["device_events"] >= device_threshold,
        "after_hours_device_usage": data["after_hours_device_events"] >= 1,
        "many_pcs": data["unique_pcs"] >= pc_threshold,
        "weekend_plus_device": (data["weekend_events"] > 0) & (data["device_events"] > 0),
        "deviation_total": data["deviation_total_events"] >= dev_total_threshold,
        "deviation_after_hours": data["deviation_after_hours"] >= dev_ah_threshold,
    })

    data["rule_score"] = conditions.sum(axis=1)
    data["label"] = (data["rule_score"] >= 3).astype(int)

    return data
