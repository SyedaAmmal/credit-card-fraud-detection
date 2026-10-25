import json
from pathlib import Path

import joblib
import pandas as pd

from load_data import load_data
from project_utils import make_train_test_split, prepare_data


def main() -> None:
    root = Path(__file__).resolve().parent
    model_path = root / "models" / "best_model.joblib"
    threshold_path = root / "models" / "decision_threshold.json"
    if not model_path.exists() or not threshold_path.exists():
        raise FileNotFoundError("Model artifacts are missing. Run 02_train.py before the demo.")

    print("=== Credit Card Fraud Detection: 10-Transaction Demo ===")
    data = load_data()
    features, target = prepare_data(data)
    _, X_test, _, y_test = make_train_test_split(features, target)
    fraud_indices = y_test[y_test == 1].sample(n=5, random_state=42).index
    legitimate_indices = y_test[y_test == 0].sample(n=5, random_state=42).index
    selected_indices = fraud_indices.append(legitimate_indices)
    X_sample = X_test.loc[selected_indices]
    y_sample = y_test.loc[selected_indices]

    model = joblib.load(model_path)
    with threshold_path.open(encoding="utf-8") as file:
        threshold = float(json.load(file)["threshold"])
    probabilities = model.predict_proba(X_sample)[:, 1]
    predictions = probabilities >= threshold

    rows = []
    for actual, probability, predicted in zip(y_sample, probabilities, predictions):
        actual_name = "FRAUD" if actual == 1 else "LEGIT"
        predicted_name = "FRAUD" if predicted else "LEGIT"
        flag = "FRAUD" if predicted else "LEGIT"
        rows.append(
            {
                "Actual": actual_name,
                "Fraud probability": f"{probability:.4f}",
                "Predicted": predicted_name,
                "Flag": flag,
            }
        )
    table = pd.DataFrame(rows, index=range(1, len(rows) + 1))
    table.index.name = "Transaction"
    print(f"Decision threshold: {threshold:.6f}\n")
    print(table.to_string())


if __name__ == "__main__":
    main()
