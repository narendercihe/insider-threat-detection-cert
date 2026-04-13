import numpy as np
import pandas as pd


def _safe_numeric_columns(df, exclude=None):
    if exclude is None:
        exclude = []

    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def build_proxy_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create proxy anomaly labels from engineered features.

    Label rule:
    - Start with label = 0
    - Mark row as anomaly (1) if one or more suspicious feature thresholds are crossed
    - If expected columns are missing, fall back to row-wise anomaly score from numeric columns
    """
    if not isinstance(feature_df, pd.DataFrame):
        raise ValueError("build_proxy_labels expects a pandas DataFrame.")

    if feature_df.empty:
        raise ValueError("feature_df is empty.")

    df = feature_df.copy()
    df["label"] = 0

    suspicious_conditions = []

    # Common insider-threat style engineered features
    if "after_hours_logins" in df.columns:
        suspicious_conditions.append(df["after_hours_logins"] >= 3)

    if "weekend_logins" in df.columns:
        suspicious_conditions.append(df["weekend_logins"] >= 2)

    if "failed_logins" in df.columns:
        suspicious_conditions.append(df["failed_logins"] >= 5)

    if "unique_pcs" in df.columns:
        suspicious_conditions.append(df["unique_pcs"] >= 3)

    if "total_logins" in df.columns:
        suspicious_conditions.append(df["total_logins"] >= 20)

    if "usb_connects" in df.columns:
        suspicious_conditions.append(df["usb_connects"] >= 3)

    if "file_access_count" in df.columns:
        suspicious_conditions.append(df["file_access_count"] >= 25)

    if "http_count" in df.columns:
        suspicious_conditions.append(df["http_count"] >= 50)

    if "email_count" in df.columns:
        suspicious_conditions.append(df["email_count"] >= 30)

    # Apply rule-based proxy labels if any known suspicious columns exist
    if suspicious_conditions:
        combined = suspicious_conditions[0]
        for cond in suspicious_conditions[1:]:
            combined = combined | cond
        df.loc[combined, "label"] = 1
    else:
        # Fallback: use z-score style anomaly score over numeric columns
        numeric_cols = _safe_numeric_columns(
            df,
            exclude=["label"]
        )

        if not numeric_cols:
            raise ValueError("No numeric columns found to build proxy labels.")

        X = df[numeric_cols].fillna(0).astype(float)

        means = X.mean()
        stds = X.std(ddof=0).replace(0, 1)

        z = ((X - means) / stds).abs()
        anomaly_score = z.sum(axis=1)

        threshold = anomaly_score.quantile(0.95)
        df["label"] = (anomaly_score >= threshold).astype(int)

    return df


# Compatibility aliases for main.py fuzzy resolver
def create_proxy_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    return build_proxy_labels(feature_df)


def generate_proxy_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    return build_proxy_labels(feature_df)


def assign_proxy_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    return build_proxy_labels(feature_df)


def apply_proxy_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    return build_proxy_labels(feature_df)


def build_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    return build_proxy_labels(feature_df)


def create_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    return build_proxy_labels(feature_df)


def generate_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    return build_proxy_labels(feature_df)


def make_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    return build_proxy_labels(feature_df)
