from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Class"


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Remove duplicate rows and separate predictors from the binary target."""
    duplicate_count = int(data.duplicated().sum())
    if duplicate_count:
        print(f"Dropping {duplicate_count:,} duplicate transaction rows.")
    data = data.drop_duplicates().reset_index(drop=True)

    features = data.drop(columns=[TARGET_COLUMN]).astype(np.float32)
    target = data[TARGET_COLUMN].astype(np.int8)
    return features, target


def make_train_test_split(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def ensure_output_directories() -> None:
    root = Path(__file__).resolve().parent
    (root / "outputs").mkdir(exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
