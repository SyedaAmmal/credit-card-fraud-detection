import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from load_data import load_data
from project_utils import make_train_test_split, prepare_data


ROOT = Path(__file__).resolve().parent
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
PLOT_FILES = {
    "ROC curve": "roc_curve.png",
    "Precision-recall curve": "precision_recall_curve.png",
    "Precision and recall vs. threshold": "precision_recall_vs_threshold.png",
}
CSV_CHUNK_SIZE = 10_000


st.set_page_config(
    page_title="Fraud Detection | Model Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the saved model and held-out test data...")
def load_demo_assets():
    model_path = ROOT / "models" / "best_model.joblib"
    threshold_path = ROOT / "models" / "decision_threshold.json"
    if not model_path.exists() or not threshold_path.exists():
        raise FileNotFoundError(
            "Saved model artifacts are missing. Expected models/best_model.joblib "
            "and models/decision_threshold.json."
        )

    model = joblib.load(model_path)
    with threshold_path.open(encoding="utf-8") as file:
        saved_threshold = float(json.load(file)["threshold"])

    data = load_data()
    features, target = prepare_data(data)
    _, X_test, _, y_test = make_train_test_split(features, target)
    probabilities = model.predict_proba(X_test)[:, 1].astype(np.float32)
    del data, features, target
    return model, saved_threshold, X_test, y_test, probabilities


def get_global_importances(model) -> pd.DataFrame | None:
    classifier = model.named_steps.get("classifier")
    importances = getattr(classifier, "feature_importances_", None)
    if importances is None:
        return None

    scale = model.named_steps.get("scale")
    if scale is not None:
        feature_names = scale.get_feature_names_out()
    else:
        feature_names = FEATURE_COLUMNS
    return (
        pd.DataFrame({"Feature": feature_names, "Global importance": importances})
        .sort_values("Global importance", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )


def show_transaction(
    row_index: int,
    features: pd.DataFrame,
    labels: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    importances: pd.DataFrame | None,
) -> None:
    row = features.iloc[row_index]
    probability = float(probabilities[row_index])
    actual = int(labels.iloc[row_index])
    predicted = int(probability >= threshold)
    correct = predicted == actual

    amount_col, time_col = st.columns(2)
    amount_col.metric("Transaction amount", f"${row['Amount']:,.2f}")
    time_col.metric("Transaction time", f"{row['Time']:,.0f} seconds")

    probability_col, verdict_col = st.columns([2, 1])
    probability_col.metric("Fraud probability", f"{probability:.1%}")
    probability_col.markdown(
        f'<div class="probability-gauge" role="img" '
        f'aria-label="Fraud probability {probability:.1%}; decision threshold {threshold:.1%}">'
        f'<div class="gauge-track">'
        f'<div class="gauge-fill" style="width:{probability:.2%};'
        f'background:{"#b42318" if probability >= threshold else "#087443"}"></div>'
        f'<div class="gauge-threshold" style="left:{threshold:.2%}"></div>'
        f'<div class="gauge-label" style="left:clamp(3.5rem, {threshold:.2%}, '
        f'calc(100% - 3.5rem))">Threshold {threshold:.1%}</div>'
        f'</div><div class="gauge-endpoints"><span>0%</span><span>100%</span></div></div>',
        unsafe_allow_html=True,
    )

    verdict = "FRAUD" if predicted else "LEGIT"
    verdict_color = "#b42318" if predicted else "#087443"
    verdict_col.markdown(
        f'<div class="verdict-badge" style="color:{verdict_color};'
        f'border-color:{verdict_color};">{verdict}</div>',
        unsafe_allow_html=True,
    )
    verdict_col.markdown(
        f'<div class="truth-label">Actual label: <b>{"FRAUD" if actual else "LEGIT"}</b></div>',
        unsafe_allow_html=True,
    )
    verdict_col.markdown(
        f'<div class="truth-label" style="color:{"#087443" if correct else "#b42318"}">'
        f'<b>{"CORRECT" if correct else "WRONG"}</b></div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Top feature importances")
    if importances is not None:
        st.dataframe(importances, hide_index=True, width="stretch")


def show_model_performance(
    labels: pd.Series, probabilities: np.ndarray, threshold: float, saved_threshold: float
) -> None:
    predictions = probabilities >= threshold
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    roc_auc = roc_auc_score(labels, probabilities)
    pr_auc = average_precision_score(labels, probabilities)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    metric_columns = st.columns(5)
    for column, title, value in zip(
        metric_columns,
        ("Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"),
        (precision, recall, f1, roc_auc, pr_auc),
    ):
        column.metric(title, f"{value:.3f}")

    st.subheader("Confusion matrix")
    matrix = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual legitimate", "Actual fraud"],
        columns=["Predicted legitimate", "Predicted fraud"],
    )
    st.dataframe(matrix.style.format("{:,}"), width="stretch")
    count_columns = st.columns(4)
    for column, label, value in zip(
        count_columns, ("TP", "FP", "FN", "TN"), (tp, fp, fn, tn)
    ):
        column.metric(label, f"{value:,}")

    st.subheader("Plots")
    plot_columns = st.columns(3)
    for column, (title, filename) in zip(plot_columns, PLOT_FILES.items()):
        plot_path = ROOT / "outputs" / filename
        with column:
            st.markdown(f"**{title}**")
            if plot_path.exists():
                st.image(str(plot_path), width="stretch")
    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds):
        fig, axis = plt.subplots(figsize=(8, 3.5))
        axis.plot(thresholds, precision_curve[:-1], label="Precision", color="#2563eb")
        axis.plot(thresholds, recall_curve[:-1], label="Recall", color="#d97706")
        axis.axvline(threshold, color="#111827", linestyle="--", label=f"Current: {threshold:.3f}")
        axis.axvline(
            saved_threshold,
            color="#64748b",
            linestyle=":",
            label=f"Saved: {saved_threshold:.3f}",
        )
        axis.set(xlabel="Decision threshold", ylabel="Score", ylim=(0, 1))
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
        fig.tight_layout()
        st.subheader("Live precision/recall trade-off")
        st.pyplot(fig, width="stretch")
        plt.close(fig)


def score_uploaded_csv(uploaded_file, model, threshold: float) -> pd.DataFrame:
    uploaded_file.seek(0)
    header = pd.read_csv(uploaded_file, nrows=0)
    missing = [column for column in FEATURE_COLUMNS if column not in header.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    uploaded_file.seek(0)
    results = []
    row_offset = 0
    for chunk in pd.read_csv(uploaded_file, chunksize=CSV_CHUNK_SIZE):
        features = chunk.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        features = features.astype(np.float32)
        values = features.to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("CSV features must contain only finite numeric values.")

        probabilities = model.predict_proba(features)[:, 1]
        result = chunk.loc[:, ["Time", "Amount"]].copy()
        result.insert(0, "CSV row", np.arange(row_offset + 1, row_offset + len(chunk) + 1))
        result["Fraud probability"] = probabilities
        result["Prediction"] = np.where(probabilities >= threshold, "FRAUD", "LEGIT")
        results.append(result)
        row_offset += len(chunk)

    if not results:
        raise ValueError("The uploaded CSV contains no data rows.")
    return pd.concat(results, ignore_index=True)


def make_held_out_sample_csv(features: pd.DataFrame, labels: pd.Series) -> bytes:
    rng = np.random.default_rng(42)
    sample_indices = np.concatenate(
        [
            rng.choice(np.flatnonzero(labels.to_numpy() == 1), size=3, replace=False),
            rng.choice(np.flatnonzero(labels.to_numpy() == 0), size=17, replace=False),
        ]
    )
    rng.shuffle(sample_indices)
    return features.iloc[sample_indices][FEATURE_COLUMNS].to_csv(index=False).encode("utf-8")


def render_app() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2.0rem; padding-bottom: 2rem; max-width: 1400px;}
        .hero {padding: 1.2rem 1.5rem; border-radius: 16px;
               background: linear-gradient(115deg, #0f172a, #1e3a5f);
               color: white; margin-bottom: 1rem;}
        .hero h1 {margin: 0 0 .35rem 0; font-size: 2.2rem;}
        .tab-footer {margin-top: 2rem; padding-top: .65rem; border-top: 1px solid
                     var(--secondary-background-color, #e2e8f0); color: #7c8491;
                     font-size: .78rem;}
        .probability-gauge {margin: 2.5rem .2rem .4rem; color: var(--text-color, currentColor);}
        .gauge-track {height: 1rem; position: relative; border-radius: 999px;
                      background: var(--secondary-background-color, #e2e8f0);}
        .gauge-fill {height: 100%; border-radius: 999px;}
        .gauge-threshold {position: absolute; top: -.35rem; bottom: -.35rem;
                          border-left: 3px solid var(--text-color, currentColor);
                          box-shadow: 0 0 0 1px var(--background-color, white);}
        .gauge-label {position: absolute; bottom: 1.35rem; transform: translateX(-50%);
                      white-space: nowrap; font-size: .78rem; font-weight: 650;
                      color: var(--text-color, currentColor);}
        .gauge-endpoints {display: flex; justify-content: space-between;
                          font-size: .75rem; margin-top: .25rem;
                          color: var(--text-color, currentColor);}
        .verdict-badge {margin-top: 1.5rem; border: 3px solid; border-radius: 16px;
                        text-align: center; font-size: 2rem; font-weight: 800;
                        padding: .85rem .5rem; background: white;}
        .truth-label {text-align: center; margin-top: .65rem; font-size: 1.05rem;}
        </style>
        <div class="hero">
            <h1>🛡️ Credit Card Fraud Detection</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model, saved_threshold, X_test, y_test, probabilities = load_demo_assets()
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

    if not 0 <= saved_threshold <= 1:
        st.error("The saved decision threshold must be between 0 and 1.")
        st.stop()

    threshold_key = "decision_threshold_slider"
    if threshold_key not in st.session_state:
        st.session_state[threshold_key] = saved_threshold
    if st.sidebar.button("Reset to saved threshold", width="stretch"):
        st.session_state[threshold_key] = saved_threshold
    threshold = st.sidebar.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        step=0.001,
        format="%.3f",
        key=threshold_key,
    )
    if np.isclose(threshold, saved_threshold, rtol=0.0, atol=1e-9):
        st.sidebar.success("Using saved threshold")
    else:
        st.sidebar.warning("Exploring: not the reported threshold")

    tabs = st.tabs(["Check a transaction", "Model performance", "Score a CSV"])
    with tabs[0]:
        st.subheader("Check a transaction")
        legitimate_indices = np.flatnonzero(y_test.to_numpy() == 0)
        fraud_indices = np.flatnonzero(y_test.to_numpy() == 1)
        random_column, fraud_column, mix_column = st.columns(3)
        if random_column.button("Random legitimate", width="stretch"):
            st.session_state["test_row_index"] = int(np.random.choice(legitimate_indices))
            st.session_state["case_button_message"] = ""
        if fraud_column.button("Random fraud", width="stretch"):
            st.session_state["test_row_index"] = int(np.random.choice(fraud_indices))
            st.session_state["case_button_message"] = ""
        if mix_column.button("Random transaction (realistic mix)", width="stretch"):
            st.session_state["test_row_index"] = int(np.random.randint(len(X_test)))
            st.session_state["case_button_message"] = ""

        predictions = probabilities >= threshold
        missed_fraud_indices = np.flatnonzero(
            (y_test.to_numpy() == 1) & ~predictions
        )
        false_alarm_indices = np.flatnonzero(
            (y_test.to_numpy() == 0) & predictions
        )
        missed_col, alarm_col = st.columns(2)
        if missed_col.button(
            "Random missed fraud (false negative)", width="stretch"
        ):
            if len(missed_fraud_indices):
                st.session_state["test_row_index"] = int(np.random.choice(missed_fraud_indices))
                st.session_state["case_button_message"] = ""
            else:
                st.session_state["case_button_message"] = (
                    "No missed frauds at the current threshold. Try lowering the threshold."
                )
        if alarm_col.button(
            "Random false alarm (false positive)", width="stretch"
        ):
            if len(false_alarm_indices):
                st.session_state["test_row_index"] = int(np.random.choice(false_alarm_indices))
                st.session_state["case_button_message"] = ""
            else:
                st.session_state["case_button_message"] = (
                    "No false alarms at the current threshold. Try raising the threshold."
                )

        if st.session_state.get("case_button_message"):
            st.info(st.session_state["case_button_message"])
        if "test_row_index" not in st.session_state:
            st.session_state["test_row_index"] = 0
        row_index = st.number_input(
            "Test row number",
            min_value=0,
            max_value=len(X_test) - 1,
            step=1,
            key="test_row_index",
        )
        show_transaction(
            int(row_index),
            X_test,
            y_test,
            probabilities,
            threshold,
            get_global_importances(model),
        )
        st.markdown(
            '<div class="tab-footer">Educational demo on a public, anonymised dataset. '
            'Not for real financial decisions.</div>',
            unsafe_allow_html=True,
        )

    with tabs[1]:
        st.subheader("Test-set performance")
        show_model_performance(y_test, probabilities, threshold, saved_threshold)
        st.markdown(
            '<div class="tab-footer">Educational demo on a public, anonymised dataset. '
            'Not for real financial decisions.</div>',
            unsafe_allow_html=True,
        )

    with tabs[2]:
        st.subheader("Score a CSV")
        sample_csv = make_held_out_sample_csv(X_test, y_test)
        st.download_button(
            "Download sample CSV",
            data=sample_csv,
            file_name="fraud_demo_sample.csv",
            mime="text/csv",
        )
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                with st.spinner("Scoring uploaded rows in memory..."):
                    results = score_uploaded_csv(uploaded_file, model, threshold)
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(f"Could not score this CSV: {exc}")
            else:
                st.success(f"Scored {len(results):,} rows at threshold {threshold:.3f}.")
                st.dataframe(results, hide_index=True, width="stretch")
                st.download_button(
                    "Download scored results",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_scores.csv",
                    mime="text/csv",
                )

        st.markdown(
            '<div class="tab-footer">Educational demo on a public, anonymised dataset. '
            'Not for real financial decisions.</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render_app()
