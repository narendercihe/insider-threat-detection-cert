import pandas as pd


def preprocess_logon_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich raw logon events.
    """
    data = df.copy()

    data["date"] = pd.to_datetime(data["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    data = data.dropna(subset=["date", "user", "pc", "activity"])

    data["activity"] = data["activity"].astype(str).str.strip()
    data["user"] = data["user"].astype(str).str.strip()
    data["pc"] = data["pc"].astype(str).str.strip()

    data["hour"] = data["date"].dt.hour
    data["day"] = data["date"].dt.date
    data["weekday"] = data["date"].dt.weekday
    data["is_weekend"] = data["weekday"] >= 5

    # Working hours: 06:00 to 18:59
    data["is_work_hour"] = data["hour"].between(6, 18)
    data["is_after_hours"] = ~data["is_work_hour"]

    data["is_logon"] = data["activity"].str.lower().eq("logon")
    data["is_logoff"] = data["activity"].str.lower().eq("logoff")

    return data


def preprocess_psychometric_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize psychometric dataset.
    """
    data = df.copy()
    data["user_id"] = data["user_id"].astype(str).str.strip()

    numeric_cols = ["O", "C", "E", "A", "N"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data
