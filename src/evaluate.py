from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def save_top_anomalies(results_df: pd.DataFrame, output_path: str | Path, top_n: int = 25) -> pd.DataFrame:
    """
    Save the most suspicious user-day records.
    Lower anomaly_score = more suspicious.
    """
    top_df = (
        results_df.sort_values(by="anomaly_score", ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(output_path, index=False)

    return top_df


def plot_hour_distribution(logon_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(logon_df["hour"], bins=24)
    plt.title("Logon Activity by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Event Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_top_suspicious_users(results_df: pd.DataFrame, output_path: str | Path, top_n: int = 10) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suspicious = results_df[results_df["is_anomaly"]].copy()
    if suspicious.empty:
        return

    counts = suspicious["user"].value_counts().head(top_n)

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title("Top Suspicious Users")
    plt.xlabel("User")
    plt.ylabel("Number of Anomalous Days")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_summary(results_df: pd.DataFrame) -> None:
    total_rows = len(results_df)
    anomaly_rows = int(results_df["is_anomaly"].sum())
    normal_rows = total_rows - anomaly_rows

    print("\n=== Model Summary ===")
    print(f"Total user-day records: {total_rows}")
    print(f"Normal records: {normal_rows}")
    print(f"Anomalous records: {anomaly_rows}")
