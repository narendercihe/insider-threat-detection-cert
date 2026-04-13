from __future__ import annotations
import numpy as np
import pandas as pd


def build_logon_features(logon_df: pd.DataFrame) -> pd.DataFrame:
    features = (
        logon_df.groupby(["user", "day"], as_index=False)
        .agg(
            total_events=("activity", "count"),
            total_logons=("is_logon", "sum"),
            total_logoffs=("is_logoff", "sum"),
            unique_pcs=("pc", "nunique"),
            after_hours_events=("is_after_hours", "sum"),
            weekend_events=("is_weekend", "sum"),
            first_hour=("hour", "min"),
            last_hour=("hour", "max"),
        )
    )

    features["activity_span"] = features["last_hour"] - features["first_hour"]
    features["after_hours_ratio"] = features["after_hours_events"] / features["total_events"].clip(lower=1)
    features["weekend_ratio"] = features["weekend_events"] / features["total_events"].clip(lower=1)

    return features


def build_device_features(device_df: pd.DataFrame) -> pd.DataFrame:
    features = (
        device_df.groupby(["user", "day"], as_index=False)
        .agg(
            device_events=("activity", "count"),
            after_hours_device_events=("is_after_hours", "sum"),
            weekend_device_events=("is_weekend", "sum"),
            unique_device_pcs=("pc", "nunique"),
            connect_like_events=("is_connect_like", "sum"),
        )
    )

    features["after_hours_device_ratio"] = (
        features["after_hours_device_events"] / features["device_events"].clip(lower=1)
    )
    features["weekend_device_ratio"] = (
        features["weekend_device_events"] / features["device_events"].clip(lower=1)
    )

    return features


def add_behavior_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values(["user", "day"]).reset_index(drop=True)

    user_means = data.groupby("user").agg(
        user_avg_total_events=("total_events", "mean"),
        user_avg_after_hours=("after_hours_events", "mean"),
        user_avg_unique_pcs=("unique_pcs", "mean"),
        user_avg_device_events=("device_events", "mean"),
    ).reset_index()

    data = data.merge(user_means, on="user", how="left")

    data["deviation_total_events"] = data["total_events"] - data["user_avg_total_events"]
    data["deviation_after_hours"] = data["after_hours_events"] - data["user_avg_after_hours"]
    data["deviation_unique_pcs"] = data["unique_pcs"] - data["user_avg_unique_pcs"]
    data["deviation_device_events"] = data["device_events"] - data["user_avg_device_events"]

    return data


def build_final_feature_table(
    logon_features: pd.DataFrame,
    device_features: pd.DataFrame,
    psychometric_df: pd.DataFrame,
    users_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = logon_features.merge(device_features, on=["user", "day"], how="left")
    df = df.merge(psychometric_df, on="user", how="left")

    if users_df is not None and "user" in users_df.columns:
        user_cols = [c for c in users_df.columns if c != "user"]
        keep_cols = ["user"] + user_cols[:3]
        df = df.merge(users_df[keep_cols].drop_duplicates("user"), on="user", how="left")

    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    for col in ["O", "C", "E", "A", "N"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df = add_behavior_deviation_features(df)

    # Safe fill after engineered features
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)

    return df
