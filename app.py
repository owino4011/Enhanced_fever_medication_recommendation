# app.py  
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils import get_logger

LOGGER = get_logger("streamlit", "logs/streamlit_output.log")

MODEL_PATH = Path("models/fever_model.pkl")
RAW_DIR = Path("data/raw")

st.set_page_config(page_title="Enhanced Fever Medication Recommendation System", layout="centered")

# ===============================
# Custom Banner / Header
# ===============================
st.markdown(
    """
    <div style="background-color:#4CAF50;padding:12px;border-radius:10px;margin-bottom:20px">
        <h1 style="color:white;text-align:center;">💊 Enhanced Fever Medication Recommendation System</h1>
        <p style="color:white;text-align:center;">🧑‍⚕️ Enter patient details and get tailored medication suggestions 🌡️</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("Model artifact not found. Please run the pipeline (python main.py) to train the model first.")
    st.stop()

# Load model pipeline
pipeline = joblib.load(MODEL_PATH)

# Load raw csv to derive UI ordering and choices
raw_csvs = list(RAW_DIR.glob("**/*.csv"))
if not raw_csvs:
    st.error("Raw CSV not found in data/raw/. Please extract your dataset first.")
    st.stop()
raw_df = pd.read_csv(raw_csvs[0], low_memory=False)

# Ensure order of inputs is same as raw columns (excluding target & dropped columns)
TARGET = "Recommended_Medication"
DROP = ["Previous_Medication"]
input_cols = [c for c in raw_df.columns if c not in DROP and c != TARGET]

# ===============================
# Sidebar for Inputs (Grouped)
# ===============================
st.sidebar.header("📝 Patient Information")

# We'll use stable widget keys and store defaults so Clear can restore them reliably
defaults = {}
input_data = {}

# Group inputs: numeric (clinical) vs categorical (demographics)
with st.sidebar.expander("🌡️ Clinical Measurements", expanded=True):
    for col in [c for c in input_cols if np.issubdtype(raw_df[c].dtype, np.number)]:
        dtype = raw_df[col].dtype
        widget_key = f"input_{col}"

        col_min = float(raw_df[col].min())
        col_max = float(raw_df[col].max())
        default = float(raw_df[col].median())
        if default <= col_min:
            default = min(col_min + 1.0, col_max)

        defaults[widget_key] = default
        if widget_key not in st.session_state:
            st.session_state[widget_key] = default

        val = st.number_input(
            label=f"{col} 🌡️" if "Temp" in col else col,
            min_value=col_min, max_value=col_max,
            step=1.0,
            key=widget_key
        )
        input_data[col] = val

with st.sidebar.expander("🧍 Patient Demographics", expanded=True):
    for col in [c for c in input_cols if not np.issubdtype(raw_df[c].dtype, np.number)]:
        widget_key = f"input_{col}"
        uniques = pd.Series(raw_df[col].dropna().unique()).astype(str).tolist()
        if not uniques:
            uniques = ["Unknown"]

        defaults[widget_key] = uniques[0]
        if widget_key not in st.session_state or str(st.session_state[widget_key]) not in uniques:
            st.session_state[widget_key] = uniques[0]

        try:
            idx = uniques.index(str(st.session_state[widget_key]))
        except ValueError:
            idx = 0
            st.session_state[widget_key] = uniques[0]

        val = st.selectbox(f"{col} 💊", options=uniques, index=idx, key=widget_key)
        input_data[col] = val

# Define reset callback
def _reset_inputs():
    for k, v in defaults.items():
        st.session_state[k] = v
    st.session_state["_inputs_reset_by_user"] = True

# ===============================
# Prediction + Download
# ===============================
if st.sidebar.button("🔍 Recommend medication"):
    try:
        X_input = pd.DataFrame([input_data])
        preds = pipeline.predict(X_input)
        probs = pipeline.predict_proba(X_input) if hasattr(pipeline, "predict_proba") else None
        pred = preds[0]
        conf = max(probs[0]) if probs is not None else None

        st.success(f"✅ Recommendation: **{pred}** {f'(confidence {conf:.2f})' if conf is not None else ''}")
        LOGGER.info(f"Input={input_data} | Prediction={pred} | Confidence={conf}")

        # Confidence bar chart (simple)
        if probs is not None:
            st.markdown("### 📊 Confidence Breakdown")
            fig, ax = plt.subplots()
            ax.bar(pipeline.classes_, probs[0], color=["#4CAF50", "#2196F3"])
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            for i, v in enumerate(probs[0]):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
            st.pyplot(fig)

        # Prepare downloadable CSV
        result_df = X_input.copy()
        result_df["Recommendation"] = pred
        if conf is not None:
            result_df["Confidence"] = conf

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Result as CSV",
            data=csv,
            file_name="medication_recommendation.csv",
            mime="text/csv",
        )

        # Clear Inputs button in sidebar (only shows after prediction)
        st.sidebar.button("🔄 Clear Inputs", on_click=_reset_inputs)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        LOGGER.error(f"Prediction failed: {e}")

# ===============================
# Helpful Footer
# ===============================
st.markdown(
    """
    <hr style="margin-top:40px;margin-bottom:10px;">
    <p style="text-align:center;color:gray;font-size:0.9em;">
    ℹ️ This system provides medication recommendations based on patient input. Always consult a qualified healthcare provider before taking any medication.
    </p>
    """,
    unsafe_allow_html=True,
)



