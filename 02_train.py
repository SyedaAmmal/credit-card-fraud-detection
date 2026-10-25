import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

from load_data import load_data
from project_utils import RANDOM_STATE, ensure_output_directories, make_train_test_split, prepare_data


N_SPLITS = 5
FAST_CV_ROWS = 50_000
IMBALANCE_STRATEGIES = ("class_weight", "smote", "undersampling")


def build_estimator(model_name: str, strategy: str, y_fit: pd.Series):
    use_class_weight = strategy == "class_weight"
    if model_name == "Logistic Regression":
        return LogisticRegression(
            class_weight="balanced" if use_class_weight else None,
            max_iter=1000,
            solver="lbfgs",
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced" if use_class_weight else None,
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
    if model_name == "XGBoost":
        from xgboost import XGBClassifier

        negative_count = int((y_fit == 0).sum())
        positive_count = int((y_fit == 1).sum())
        scale_pos_weight = negative_count / positive_count if use_class_weight else 1.0
        return XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            n_jobs=1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )
    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(model_name: str, strategy: str, y_fit: pd.Series) -> Pipeline:
    sampler = {
        "class_weight": "passthrough",
        "smote": SMOTE(random_state=RANDOM_STATE),
        "undersampling": RandomUnderSampler(random_state=RANDOM_STATE),
    }[strategy]
    scale_amount_time = ColumnTransformer(
        [("amount_time", RobustScaler(), ["Amount", "Time"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("scale", scale_amount_time),
            ("resample", sampler),
            ("classifier", build_estimator(model_name, strategy, y_fit)),
        ]
    )


def available_models() -> list[str]:
    models = ["Logistic Regression", "Random Forest"]
    if importlib.util.find_spec("xgboost") is not None:
        models.append("XGBoost")
        print("XGBoost detected; it will be included in the comparison.")
    else:
        print("XGBoost not installed; continuing with Logistic Regression and Random Forest.")
    return models


def fast_cv_sample(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Sample the CV data while retaining every fraud row in the training split."""
    if len(X_train) <= FAST_CV_ROWS:
        return X_train, y_train

    fraud_indices = y_train.index[y_train == 1]
    legitimate_indices = y_train.index[y_train == 0]
    legitimate_sample_size = min(
        len(legitimate_indices), max(0, FAST_CV_ROWS - len(fraud_indices))
    )
    sampled_legitimate_indices = legitimate_indices.to_series().sample(
        n=legitimate_sample_size, random_state=RANDOM_STATE
    ).index
    sampled_indices = np.concatenate((fraud_indices, sampled_legitimate_indices))
    np.random.default_rng(RANDOM_STATE).shuffle(sampled_indices)
    return X_train.loc[sampled_indices], y_train.loc[sampled_indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare fraud detection models.")
    cv_mode = parser.add_mutually_exclusive_group()
    cv_mode.add_argument(
        "--fast",
        dest="cv_mode",
        action="store_const",
        const="fast",
        help=f"Use a stratified CV sample of about {FAST_CV_ROWS:,} training rows (default).",
    )
    cv_mode.add_argument(
        "--full",
        dest="cv_mode",
        action="store_const",
        const="full",
        help="Run cross-validation on the complete training split.",
    )
    parser.set_defaults(cv_mode="fast")
    args = parser.parse_args()

    print("=== Stage 1: Load, de-duplicate, and split data ===")
    data = load_data()
    features, target = prepare_data(data)
    X_train, _, y_train, _ = make_train_test_split(features, target)
    held_out_rows = int(len(target) * 0.2)
    del data, features, target
    print(
        f"Training rows: {len(X_train):,}; validation/test rows held out: "
        f"{held_out_rows:,}. The test set is not used for CV or threshold selection."
    )

    print("\n=== Stage 2: Compare models and imbalance strategies (5-fold CV) ===")
    model_names = available_models()
    if args.cv_mode == "fast":
        X_cv, y_cv = fast_cv_sample(X_train, y_train)
        print(
            f"Fast mode: cross-validation uses {len(X_cv):,} training rows, "
            f"including all {int((y_train == 1).sum()):,} fraud rows. "
            "The final fit still uses the full training split."
        )
    else:
        X_cv, y_cv = X_train, y_train
        print(f"Full mode: cross-validation uses all {len(X_cv):,} training rows.")

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []
    total_folds = len(model_names) * len(IMBALANCE_STRATEGIES) * N_SPLITS
    completed_folds = 0
    cv_started = perf_counter()

    for model_name in model_names:
        for strategy in IMBALANCE_STRATEGIES:
            oof_probabilities = np.zeros(len(y_cv), dtype=np.float32)
            fold_scores = []
            for fold_number, (train_indices, validation_indices) in enumerate(
                folds.split(X_cv, y_cv), start=1
            ):
                fold_started = perf_counter()
                fold_X_train = X_cv.iloc[train_indices]
                fold_y_train = y_cv.iloc[train_indices]
                fold_X_validation = X_cv.iloc[validation_indices]
                fold_y_validation = y_cv.iloc[validation_indices]
                pipeline = build_pipeline(model_name, strategy, fold_y_train)
                pipeline.fit(fold_X_train, fold_y_train)
                probabilities = pipeline.predict_proba(fold_X_validation)[:, 1]
                oof_probabilities[validation_indices] = probabilities
                fold_score = average_precision_score(fold_y_validation, probabilities)
                fold_scores.append(fold_score)
                completed_folds += 1
                fold_elapsed = perf_counter() - fold_started
                total_elapsed = perf_counter() - cv_started
                print(
                    f"[{completed_folds}/{total_folds}] {model_name} | {strategy} | "
                    f"fold {fold_number}/{N_SPLITS}: PR-AUC={fold_score:.5f} | "
                    f"fold elapsed={fold_elapsed:.1f}s | total elapsed={total_elapsed:.1f}s",
                    flush=True,
                )

            results.append(
                {
                    "model": model_name,
                    "strategy": strategy,
                    "mean_pr_auc": float(np.mean(fold_scores)),
                    "std_pr_auc": float(np.std(fold_scores)),
                    "oof_probabilities": oof_probabilities,
                }
            )

    results.sort(key=lambda result: result["mean_pr_auc"], reverse=True)
    table = pd.DataFrame(
        [
            {
                "Model": result["model"],
                "Imbalance strategy": result["strategy"],
                "Mean CV PR-AUC": result["mean_pr_auc"],
                "Std CV PR-AUC": result["std_pr_auc"],
            }
            for result in results
        ]
    )
    print("\nCross-validation results (sorted by mean PR-AUC):")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    best = results[0]
    precision, recall, thresholds = precision_recall_curve(y_cv, best["oof_probabilities"])
    f1_scores = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + np.finfo(float).eps
    )
    best_threshold = float(thresholds[int(np.argmax(f1_scores))])
    print(
        f"\nBest combination: {best['model']} + {best['strategy']} "
        f"(mean CV PR-AUC={best['mean_pr_auc']:.5f})."
    )
    print(f"Selected decision threshold from out-of-fold training predictions: {best_threshold:.6f}")

    print("\n=== Stage 3: Fit selected pipeline and save artifacts ===")
    best_pipeline = build_pipeline(best["model"], best["strategy"], y_train)
    best_pipeline.fit(X_train, y_train)
    ensure_output_directories()
    root = Path(__file__).resolve().parent
    model_path = root / "models" / "best_model.joblib"
    import joblib

    joblib.dump(best_pipeline, model_path)
    threshold_path = root / "models" / "decision_threshold.json"
    with threshold_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "threshold": best_threshold,
                "model": best["model"],
                "strategy": best["strategy"],
                "mean_cv_pr_auc": best["mean_pr_auc"],
            },
            file,
            indent=2,
        )
    print(f"Saved selected pipeline to {model_path}")
    print(f"Saved out-of-fold decision threshold to {threshold_path}")


if __name__ == "__main__":
    main()
