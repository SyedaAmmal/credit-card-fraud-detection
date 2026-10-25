from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


DATA_PATH = Path(__file__).resolve().parent / "data" / "creditcard.csv"
REQUIRED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]


def _validate_data(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    data = data.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS[:-1]:
        data[column] = pd.to_numeric(data[column], errors="raise").astype(np.float32)

    class_values = pd.to_numeric(data["Class"], errors="raise")
    labels = set(class_values.dropna().unique())
    if (
        class_values.isna().any()
        or not class_values.isin([0, 1]).all()
        or labels != {0, 1}
    ):
        raise ValueError("Class must contain both binary labels: 0 (legitimate) and 1 (fraud).")
    data["Class"] = class_values.astype(np.int8)
    return data


def load_data() -> pd.DataFrame:
    """Load the local transaction file, fetching it from OpenML if necessary."""
    if DATA_PATH.exists():
        print(f"Loading dataset from {DATA_PATH}")
        column_dtypes = {column: np.float32 for column in REQUIRED_COLUMNS[:-1]}
        column_dtypes["Class"] = np.int8
        return _validate_data(pd.read_csv(DATA_PATH, dtype=column_dtypes))

    print(f"Dataset not found at {DATA_PATH}. Trying scikit-learn OpenML fallback...")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        fetched = fetch_openml(name="creditcard", version=1, as_frame=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not load data from OpenML. Check your internet connection and "
            "confirm the OpenML dataset is available."
        ) from exc

    frame = fetched.data.copy()
    if "Class" not in frame.columns and fetched.target is not None:
        frame["Class"] = fetched.target
    data = _validate_data(frame)
    data.to_csv(DATA_PATH, index=False)
    print(f"Saved OpenML dataset to {DATA_PATH}")
    return data


if __name__ == "__main__":
    print("=== Loading credit card transaction data ===")
    dataset = load_data()
    print(f"Loaded {len(dataset):,} rows and {len(dataset.columns)} columns.")
