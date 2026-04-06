from pathlib import Path
import pandas as pd


def load_logon_data(path: str | Path) -> pd.DataFrame:
    """
    Load CERT logon data.
    Expected columns:
    id, date, user, pc, activity
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"logon file not found: {path}")

    df = pd.read_csv(path)

    required_cols = {"id", "date", "user", "pc", "activity"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"logon.csv is missing columns: {missing}")

    return df


def load_psychometric_data(path: str | Path) -> pd.DataFrame:
    """
    Load psychometric data.
    Expected columns:
    employee_name, user_id, O, C, E, A, N
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"psychometric file not found: {path}")

    df = pd.read_csv(path)

    required_cols = {"employee_name", "user_id", "O", "C", "E", "A", "N"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"psychometric.csv is missing columns: {missing}")

    return df
