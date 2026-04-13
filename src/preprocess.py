from __future__ import annotations
import pandas as pd


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]

    for original in df.columns:
        low = original.lower()
        for candidate in candidates:
            if candidate.lower() in low:
                return original

    raise ValueError(f"Could not find any matching columns from: {candidates}")


def preprocess_logon(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    date_col = _find_column(data, ["date", "datetime", "timestamp"])
    user_col = _find_column(data, ["user", "user_id"])
    pc_col = _find_column(data, ["pc", "computer", "machine"])
    activity_col = _find_column(data, ["activity", "action"])

    data = data.rename(
        columns={
            date_col: "date",
            user_col: "user",
            pc_col: "pc",
            activity_col: "activity",
        }
    )

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "user", "pc", "activity"]).copy()

    data["user"] = data["user"].astype(str).str.strip()
    data["pc"] = data["pc"].astype(str).str.strip()
    data["activity"] = data["activity"].astype(str).str.strip().str.lower()

    data["hour"] = data["date"].dt.hour
    data["day"] = data["date"].dt.date
    data["weekday"] = data["date"].dt.weekday
    data["is_weekend"] = data["weekday"] >= 5
    data["is_after_hours"] = ~data["hour"].between(6, 18)

    data["is_logon"] = data["activity"].eq("logon")
    data["is_logoff"] = data["activity"].eq("logoff")

    return data


def preprocess_device(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    date_col = _find_column(data, ["date", "datetime", "timestamp"])
    user_col = _find_column(data, ["user", "user_id"])
    pc_col = _find_column(data, ["pc", "computer", "machine"])
    activity_col = _find_column(data, ["activity", "action"])

    data = data.rename(
        columns={
            date_col: "date",
            user_col: "user",
            pc_col: "pc",
            activity_col: "activity",
        }
    )

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "user", "pc", "activity"]).copy()

    data["user"] = data["user"].astype(str).str.strip()
    data["pc"] = data["pc"].astype(str).str.strip()
    data["activity"] = data["activity"].astype(str).str.strip().str.lower()

    data["hour"] = data["date"].dt.hour
    data["day"] = data["date"].dt.date
    data["weekday"] = data["date"].dt.weekday
    data["is_weekend"] = data["weekday"] >= 5
    data["is_after_hours"] = ~data["hour"].between(6, 18)

    # Flexible detection for connect/insert style events
    data["is_connect_like"] = data["activity"].str.contains("connect|insert|plug", regex=True)

    return data


def preprocess_psychometric(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    rename_map = {}
    cols = {c.lower(): c for c in data.columns}

    if "user_id" in cols:
        rename_map[cols["user_id"]] = "user"
    elif "user" in cols:
        rename_map[cols["user"]] = "user"

    for trait in ["O", "C", "E", "A", "N"]:
        if trait in data.columns:
            rename_map[trait] = trait
        else:
            for col in data.columns:
                if col.lower() == trait.lower():
                    rename_map[col] = trait

    data = data.rename(columns=rename_map)

    required = ["user", "O", "C", "E", "A", "N"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"psychometric.csv missing columns: {missing}")

    data["user"] = data["user"].astype(str).str.strip()
    for col in ["O", "C", "E", "A", "N"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data[["user", "O", "C", "E", "A", "N"]]


def preprocess_users(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None

    data = df.copy()

    try:
        user_col = _find_column(data, ["user", "user_id"])
        data = data.rename(columns={user_col: "user"})
        data["user"] = data["user"].astype(str).str.strip()
        return data
    except Exception:
        return None
