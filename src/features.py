import pandas as pd


def build_daily_user_features(logon_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw logon events into user-day features.
    """
    grouped = logon_df.groupby(["user", "day"], as_index=False).agg(
        total_events=("id", "count"),
        total_logons=("is_logon", "sum"),
        total_logoffs=("is_logoff", "sum"),
        unique_pcs=("pc", "nunique"),
        after_hours_events=("is_after_hours", "sum"),
        weekend_events=("is_weekend", "sum"),
        first_hour=("hour", "min"),
        last_hour=("hour", "max"),
    )

    grouped["activity_span"] = grouped["last_hour"] - grouped["first_hour"]
    grouped["after_hours_ratio"] = grouped["after_hours_events"] / grouped["total_events"]
    grouped["weekend_ratio"] = grouped["weekend_events"] / grouped["total_events"]

    return grouped


def merge_psychometric_features(
    features_df: pd.DataFrame, psychometric_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge user-day behavior features with psychometric traits.
    """
    merged = features_df.merge(
        psychometric_df[["user_id", "O", "C", "E", "A", "N"]],
        left_on="user",
        right_on="user_id",
        how="left",
    )

    merged = merged.drop(columns=["user_id"], errors="ignore")

    # Fill missing psychometric values with median
    for col in ["O", "C", "E", "A", "N"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged[col].median())

    return merged
