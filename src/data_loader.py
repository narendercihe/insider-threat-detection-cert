from pathlib import Path
from typing import Optional
import pandas as pd


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_all_data(raw_dir: str | Path) -> dict[str, Optional[pd.DataFrame]]:
    raw_dir = Path(raw_dir)

    data = {
        "logon": _safe_read_csv(raw_dir / "logon.csv"),
        "device": _safe_read_csv(raw_dir / "device.csv"),
        "psychometric": _safe_read_csv(raw_dir / "psychometric.csv"),
        "users": _safe_read_csv(raw_dir / "users.csv"),
        "file": _safe_read_csv(raw_dir / "file.csv"),
        "http": _safe_read_csv(raw_dir / "http.csv"),
        "email": _safe_read_csv(raw_dir / "email.csv"),  # Optional
        "ldap": _safe_read_csv(raw_dir / "ldap.csv"),
        "answers": _safe_read_csv(raw_dir / "answers.csv"),
    }

    # Raise errors if any of the required files are missing
    if data["logon"] is None:
        raise FileNotFoundError(f"logon.csv not found in {raw_dir}")

    if data["device"] is None:
        raise FileNotFoundError(f"device.csv not found in {raw_dir}")

    if data["psychometric"] is None:
        raise FileNotFoundError(f"psychometric.csv not found in {raw_dir}")

    # Add checks for the new files
    if data["file"] is None:
        raise FileNotFoundError(f"file.csv not found in {raw_dir}")

    if data["http"] is None:
        raise FileNotFoundError(f"http.csv not found in {raw_dir}")

    if data["ldap"] is None:
        raise FileNotFoundError(f"ldap.csv not found in {raw_dir}")

    if data["answers"] is None:
        raise FileNotFoundError(f"answers.csv not found in {raw_dir}")

    return data
