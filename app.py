import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from dashboard_core import get_classification_report, get_training_history, predict_class, load_model_and_scaler
from path_utils import CHARTS

st.set_page_config(page_title="Wine Quality Classifier", layout="wide")
st.title("Wine Quality Multi-class Predictor")
st.caption("Raw wine features -> saved scaler -> ANN logits -> softmax probabilities")

# --- HEALTH CHECK ---
model, scaler = load_model_and_scaler()
with st.sidebar:
    st.header("Wine Chemistry Inputs")
    if model and scaler:
        st.success("✅ System Ready: Model & Scaler Loaded")
    else:
        st.error("❌ System Error: Missing Model/Scaler Artifacts")
        st.info("Ensure `models/` and `data/processed/` folders are populated.")

features = {
    "fixed acidity": st.sidebar.slider("fixed acidity", 4.0, 16.0, 8.0, 0.1),
    "volatile acidity": st.sidebar.slider("volatile acidity", 0.1, 1.6, 0.5, 0.01),
    "citric acid": st.sidebar.slider("citric acid", 0.0, 1.0, 0.3, 0.01),
    "residual sugar": st.sidebar.slider("residual sugar", 0.8, 16.0, 2.5, 0.1),
    "chlorides": st.sidebar.slider("chlorides", 0.01, 0.7, 0.08, 0.01),
    "free sulfur dioxide": st.sidebar.slider("free sulfur dioxide", 1, 80, 15),
    "total sulfur dioxide": st.sidebar.slider("total sulfur dioxide", 6, 300, 46),
    "density": st.sidebar.slider("density", 0.990, 1.005, 0.996, 0.0001),
    "pH": st.sidebar.slider("pH", 2.7, 4.1, 3.3, 0.01),
    "sulphates": st.sidebar.slider("sulphates", 0.3, 2.0, 0.62, 0.01),
    "alcohol": st.sidebar.slider("alcohol", 8.0, 15.5, 10.2, 0.1),
}

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if st.button("Predict Quality", disabled=(model is None)):
    try:
        pred, probs = predict_class(features)
        labels = ["Low (0)", "Medium (1)", "High (2)"]
        st.session_state.prediction_history.append(probs)

        st.subheader("Live Prediction Result")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("Predicted Class", labels[pred])
            st.metric("Confidence", f"{max(probs):.3f}")
        with c2:
            st.write("Class probabilities")
            prob_df = pd.DataFrame({"class": labels, "probability": probs})
            st.bar_chart(prob_df.set_index("class"))

        hist_df = pd.DataFrame(
            st.session_state.prediction_history,
            columns=["Low (0)", "Medium (1)", "High (2)"],
        )
        st.line_chart(hist_df)
        st.caption("Live prediction history (updates on each Predict click).")
    except Exception as e:
        st.error(f"Prediction Error: {e}")

curves_tab, cm_tab, f1_tab, arch_tab = st.tabs(
    ["Training Curves", "Confusion Matrix", "Per-class F1", "Model & Metrics Guide"]
)

with curves_tab:
    hist = get_training_history()
    if hist:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        x = np.arange(1, len(hist["train_loss"]) + 1)
        ax[0].plot(x, hist["train_loss"], label="train")
        ax[0].plot(x, hist["val_loss"], label="val")
        ax[0].set_title("Loss")
        ax[0].legend()
        ax[1].plot(x, hist["val_acc"], label="val_acc")
        ax[1].set_title("Validation Accuracy")
        ax[1].legend()
        st.pyplot(fig)
    else:
        st.info("Run notebook 03 to generate training history.")

with cm_tab:
    cm_path = CHARTS / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Multi-class Confusion Matrix")
    else:
        st.info("Run notebook 03 to generate confusion matrix chart.")

with f1_tab:
    report = get_classification_report()
    if report and "classification_report" in report:
        rows = []
        for k in ["0", "1", "2"]:
            if k in report["classification_report"]:
                rows.append({"class": k, "f1-score": report["classification_report"][k]["f1-score"]})
        if rows:
            df = pd.DataFrame(rows)
            st.bar_chart(df.set_index("class"))
    else:
        st.info("Run notebook 03 to generate classification report.")

with arch_tab:
    st.code(
        "Input(11) -> Linear(128) -> BatchNorm1d -> ReLU -> Dropout(0.4)\n"
        "          -> Linear(64)  -> BatchNorm1d -> ReLU -> Dropout(0.3)\n"
        "          -> Linear(3)   -> logits (no softmax in model)",
        language="text",
    )

    report = get_classification_report()
    if report:
        metric_cols = st.columns(3)
        if "accuracy" in report:
            metric_cols[0].metric("Accuracy", f"{report['accuracy']:.4f}")
        if "macro_f1" in report:
            metric_cols[1].metric("Macro F1", f"{report['macro_f1']:.4f}")
        if "weighted_f1" in report:
            metric_cols[2].metric("Weighted F1", f"{report['weighted_f1']:.4f}")

    st.markdown("### Metric Meaning")
    left, right = st.columns(2)
    with left:
        st.success(
            "**Accuracy**\n\n"
            "Overall correct predictions. Good when class distribution is balanced."
        )
        st.info(
            "**Macro F1**\n\n"
            "Treats all classes equally. Best when minority-class performance matters."
        )
    with right:
        st.success(
            "**Weighted F1**\n\n"
            "Accounts for class frequencies. Useful for overall practical performance."
        )
        st.info(
            "**Confusion Matrix**\n\n"
            "Shows where classes are mixed up (rows=true, columns=predicted)."
        )
