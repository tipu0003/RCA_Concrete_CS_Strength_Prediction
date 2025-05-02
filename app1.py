#!/usr/bin/env python
# streamlit run app1.py

import numpy as np
import numpy.random._pickle as _np_pickle
# ─── Monkey‐patch to accept class BitGenerator names ─────────
_orig = _np_pickle.__bit_generator_ctor
def _fixed_ctor(bit_generator_name):
    # if we're handed the class rather than its string name, grab __name__
    if isinstance(bit_generator_name, type):
        bit_generator_name = bit_generator_name.__name__
    return _orig(bit_generator_name)
_np_pickle.__bit_generator_ctor = _fixed_ctor
# ─────────────────────────────────────────────────────────────

# import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import cloudpickle

# ───────────────────── paths & constants ─────────────────────
MODEL_DIR = Path("saved_models")
IMP_DIR   = Path("feature_importance")
DATA_PATH = Path("dataset.xlsx")

OUTPUT_LABEL = "Predicted Compressive Strength"
OUTPUT_UNIT  = "MPa"

# ─────────────────── load feature names & units ───────────────────
data = pd.read_excel(DATA_PATH)
feature_names = list(data.columns[:-1])

UNITS = {
    "w/c"        : "-",
    "Cement"     : "kg/m³",
    "Curing Days": "days",
    "SP (%)"     : "%",
    "NCA (%)"    : "%",
    "RCA (%)"    : "%",
    "FA"         : "kg/m³"
}
def make_label(col):
    unit = UNITS.get(col, "")
    return f"{col} [{unit}]" if unit else col

display_labels = [make_label(c) for c in feature_names]

# ───────────────────── load scaler, models, metrics ───────────────────
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

# load every .pkl except the scaler
model_paths = sorted(p for p in MODEL_DIR.glob("*.pkl") if p.name != "scaler.pkl")
models = {p.stem: cloudpickle.load(open(p, "rb"))
          for p in MODEL_DIR.glob("*.pkl")
          if p.name != "scaler.pkl"}

metrics_df = (
    pd.read_excel(MODEL_DIR / "test_metrics.xlsx")
      .set_index("Model")
)

# ───────────────────── Streamlit UI setup ─────────────────────
st.set_page_config(
    page_title="Model Zoo – Interactive Predictor",
    page_icon="✨",
    layout="centered"
)

st.title("📈 Interactive Regression Predictor")
st.write(
    "Enter feature values, pick a model (or all), then **Predict**.  "
    "The app scales inputs, returns predictions, shows test‐set metrics, "
    "and plots feature importance."
)

model_choice = st.sidebar.selectbox(
    "🔧 Choose model:",
    ["All Models"] + list(models.keys())
)

# ─────────────────────── input form ───────────────────────
with st.form("input_form"):
    cols = st.columns(2)
    user_vals = []
    for idx, label in enumerate(display_labels):
        with cols[idx % 2]:
            val = st.number_input(label, format="%.5f")
            user_vals.append(val)
    submitted = st.form_submit_button("Predict")

# ────────────────── plot helper ──────────────────
def plot_importance(model_name):
    path = IMP_DIR / f"{model_name}_imp.csv"
    if not path.exists():
        st.warning("No importance data for this model.")
        return
    imp = pd.read_csv(path).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(6, max(2, imp.shape[0]*0.4)))
    ax.barh(imp["feature"], imp["importance"], edgecolor="k")
    ax.set_xlabel("Permutation Importance (mean R² drop)")
    ax.set_title(f"{model_name} Feature Importance")
    st.pyplot(fig)
    plt.close(fig)

# ─────────────────── on form submission ───────────────────
if submitted:
    X_scaled = scaler.transform(np.array(user_vals).reshape(1, -1))

    if model_choice == "All Models":
        preds = {name: mdl.predict(X_scaled)[0] for name, mdl in models.items()}
        preds_df = pd.DataFrame.from_dict(
            preds, orient="index", columns=[f"Pred ({OUTPUT_UNIT})"]
        )
        show_cols = ["R2", "RMSE", "MAE", "MAPE"]
        preds_df = preds_df.join(metrics_df[show_cols])

        st.subheader("🏷 Predictions & Test Metrics (All Models)")
        st.dataframe(preds_df.style.format({
            f"Pred ({OUTPUT_UNIT})": "{:.4f}",
            "R2": "{:.3f}", "RMSE": "{:.3f}",
            "MAE": "{:.3f}", "MAPE": "{:.2%}"
        }))

        st.markdown("---")
        picked = st.selectbox(
            "🔍 See feature importance for:",
            list(models.keys())
        )
        plot_importance(picked)

    else:
        mdl = models[model_choice]
        pred = mdl.predict(X_scaled)[0]

        st.subheader(f"🏷 Prediction – **{model_choice}**")
        st.metric(OUTPUT_LABEL, f"{pred:,.4f} {OUTPUT_UNIT}")

        row = metrics_df.loc[model_choice]
        st.markdown(
            f"**Test‐set metrics:**  \n"
            f"• R² = `{row.R2:.3f}`  \n"
            f"• RMSE = `{row.RMSE:.3f}`  \n"
            f"• MAE = `{row.MAE:.3f}`  \n"
            f"• MAPE = `{row.MAPE:.2%}`"
        )

        st.markdown("---")
        plot_importance(model_choice)

st.caption("© 2025 Dr. Rupesh Kumar Tipu — Streamlit")
