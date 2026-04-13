from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_label_distribution(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = df["label"].value_counts().sort_index()
    labels = ["Normal (0)", "Suspicious (1)"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels[:len(counts)], counts.values)
    plt.title("Pseudo Label Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_metric_comparison(table_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    x = range(len(table_df))
    width = 0.18

    plt.figure(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        plt.bar([p + i * width for p in x], table_df[metric], width=width, label=metric)

    plt.xticks([p + 1.5 * width for p in x], table_df["Model"])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion(y_true, y_pred, title: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Suspicious"])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_score_histogram(df: pd.DataFrame, score_col: str, title: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.hist(df[score_col], bins=40)
    plt.title(title)
    plt.xlabel(score_col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
