from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from load_data import load_data
from project_utils import ensure_output_directories


def main() -> None:
    print("=== Stage 1: Load and inspect dataset ===")
    data = load_data()
    ensure_output_directories()
    output_dir = Path(__file__).resolve().parent / "outputs"

    print(f"\nShape: {data.shape}")
    print("\nData types:")
    print(data.dtypes.to_string())
    print("\nMissing values by column:")
    print(data.isna().sum().to_string())
    print(f"\nDuplicate rows: {data.duplicated().sum():,}")
    class_counts = data["Class"].value_counts().sort_index()
    fraud_percentage = 100 * class_counts.get(1, 0) / len(data)
    print("\nClass counts (0 = legitimate, 1 = fraud):")
    print(class_counts.to_string())
    print(f"Fraud percentage: {fraud_percentage:.4f}%")
    print(
        "Why imbalance matters: with very few fraud examples, a model can appear accurate "
        "by predicting legitimate for nearly every transaction. "
        "Precision-recall metrics and class-aware training are more informative."
    )

    print("\n=== Stage 2: Save exploratory plots ===")
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 5))
    sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, color="steelblue")
    plt.xlabel("Class (0 = legitimate, 1 = fraud)")
    plt.ylabel("Transactions")
    plt.title("Transaction Class Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    amount_values = np.log1p(data["Amount"].to_numpy())
    time_values = data["Time"].to_numpy()
    amount_bins = np.histogram_bin_edges(amount_values, bins=50)
    time_bins = np.histogram_bin_edges(time_values, bins=50)
    for label, name, color in [(0, "Legitimate", "steelblue"), (1, "Fraud", "darkorange")]:
        subset = data.loc[data["Class"] == label]
        amount_counts, _ = np.histogram(np.log1p(subset["Amount"]), bins=amount_bins)
        time_counts, _ = np.histogram(subset["Time"], bins=time_bins)
        axes[0].bar(
            amount_bins[:-1],
            amount_counts,
            width=np.diff(amount_bins),
            align="edge",
            alpha=0.65,
            label=name,
            color=color,
        )
        axes[1].bar(
            time_bins[:-1],
            time_counts,
            width=np.diff(time_bins),
            align="edge",
            alpha=0.65,
            label=name,
            color=color,
        )
    axes[0].set_xlabel("log(1 + Amount)")
    axes[0].set_ylabel("Transactions")
    axes[0].set_title("Transaction Amount by Class")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Transactions")
    axes[1].set_title("Transaction Time by Class")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "amount_time_by_class.png", dpi=150)
    plt.close(fig)

    feature_columns = [column for column in data.columns if column != "Class"]
    plt.figure(figsize=(16, 13))
    sns.heatmap(
        data[feature_columns].corr(),
        cmap="coolwarm",
        center=0,
        xticklabels=True,
        yticklabels=True,
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved plots in {output_dir}")


if __name__ == "__main__":
    main()
