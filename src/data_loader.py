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
    }

    if data["logon"] is None:
        raise FileNotFoundError(f"logon.csv not found in {raw_dir}")

    if data["device"] is None:
        raise FileNotFoundError(f"device.csv not found in {raw_dir}")

    if data["psychometric"] is None:
        raise FileNotFoundError(f"psychometric.csv not found in {raw_dir}")

    return data
