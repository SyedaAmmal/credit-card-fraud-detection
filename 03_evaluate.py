import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from load_data import load_data
from project_utils import ensure_output_directories, make_train_test_split, prepare_data


def main() -> None:
    root = Path(__file__).resolve().parent
    model_path = root / "models" / "best_model.joblib"
    threshold_path = root / "models" / "decision_threshold.json"
    if not model_path.exists() or not threshold_path.exists():
        raise FileNotFoundError("Model artifacts are missing. Run 02_train.py before evaluation.")

    print("=== Stage 1: Load saved model and recreate untouched test split ===")
    data = load_data()
    features, target = prepare_data(data)
    _, X_test, _, y_test = make_train_test_split(features, target)
    model = joblib.load(model_path)
    with threshold_path.open(encoding="utf-8") as file:
        threshold = float(json.load(file)["threshold"])
    print(f"Evaluating {len(X_test):,} held-out rows at threshold {threshold:.6f}.")

    print("\n=== Stage 2: Evaluate test predictions ===")
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )
    metrics_text = (
        f"Decision threshold: {threshold:.6f}\n"
        f"Precision: {precision:.6f}\n"
        f"Recall: {recall:.6f}\n"
        f"F1: {f1:.6f}\n"
        f"ROC-AUC: {roc_auc:.6f}\n"
        f"PR-AUC (average precision): {pr_auc:.6f}\n"
        f"True positives: {tp:,}\n"
        f"False positives: {fp:,}\n"
        f"False negatives: {fn:,}\n"
        f"True negatives: {tn:,}\n\n"
        f"Classification report:\n{report}"
    )
    print(metrics_text)

    print("\n=== Stage 3: Save evaluation report and plots ===")
    ensure_output_directories()
    output_dir = root / "outputs"
    (output_dir / "evaluation_metrics.txt").write_text(metrics_text, encoding="utf-8")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        [[tn, fp], [fn, tp]],
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=["Predicted legitimate", "Predicted fraud"],
        yticklabels=["Actual legitimate", "Actual fraud"],
    )
    plt.title("Confusion Matrix at Tuned Threshold")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()

    pr_precision, pr_recall, _ = precision_recall_curve(y_test, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(pr_recall, pr_precision, label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close()

    threshold_precision, threshold_recall, thresholds = precision_recall_curve(y_test, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, threshold_precision[:-1], label="Precision")
    plt.plot(thresholds, threshold_recall[:-1], label="Recall")
    plt.axvline(threshold, color="gray", linestyle="--", label=f"Selected = {threshold:.3f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title("Precision and Recall vs. Decision Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_vs_threshold.png", dpi=150)
    plt.close()

    print(
        "\nTrade-off: lowering the threshold usually catches more fraudulent transactions "
        f"(higher recall; {fn:,} frauds were missed here), but also flags more legitimate "
        f"transactions for review ({fp:,} false alarms). Raising it tends to reduce false "
        "alarms while allowing more frauds to pass undetected."
    )
    print(f"Saved report and plots in {output_dir}")


if __name__ == "__main__":
    main()
