import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Alzheimer's Disease Risk Classifier",
    page_icon="🧠",
    layout="centered",
)

# ------------------------------------------------------------------
# Load model (cached so it only loads once per session)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model/model.pkl")
    features = joblib.load("model/features.pkl")
    return model, features

model, FEATURES = load_model()

FIELD_INFO = {
    "MemoryComplaints": {"label": "Memory Complaints", "type": "select"},
    "BehavioralProblems": {"label": "Behavioral Problems", "type": "select"},
    "SleepQuality": {"label": "Sleep Quality", "type": "slider", "min": 4.0, "max": 10.0, "default": 7.0,
                      "help": "0 = poor sleep, 10 = excellent sleep"},
    "CholesterolHDL": {"label": "Cholesterol HDL (mg/dL)", "type": "slider", "min": 20.0, "max": 100.0, "default": 55.0,
                        "help": "'Good' cholesterol level"},
    "MMSE": {"label": "MMSE Score", "type": "slider", "min": 0.0, "max": 30.0, "default": 24.0,
             "help": "Mini-Mental State Exam — lower score indicates more cognitive impairment"},
    "FunctionalAssessment": {"label": "Functional Assessment", "type": "slider", "min": 0.0, "max": 10.0, "default": 6.0,
                              "help": "0 = severe impairment, 10 = fully independent"},
    "ADL": {"label": "ADL (Activities of Daily Living)", "type": "slider", "min": 0.0, "max": 10.0, "default": 6.0,
            "help": "0 = needs full assistance, 10 = fully independent"},
}

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🧠 Alzheimer's Disease Risk Classifier")
st.caption("Project P685 — Random Forest model on 7 clinically-selected features")

st.info(
    "This is a student project demo, not a certified diagnostic tool. "
    "It must not be used for real medical decisions. Always consult a "
    "qualified healthcare professional.",
    icon="⚠️",
)

# ------------------------------------------------------------------
# Sidebar inputs
# ------------------------------------------------------------------
st.sidebar.header("Patient Inputs")

inputs = {}
for feat in FEATURES:
    meta = FIELD_INFO[feat]
    if meta["type"] == "select":
        choice = st.sidebar.radio(meta["label"], options=["No", "Yes"], horizontal=True, key=feat)
        inputs[feat] = 1 if choice == "Yes" else 0
    else:
        inputs[feat] = st.sidebar.slider(
            meta["label"],
            min_value=meta["min"],
            max_value=meta["max"],
            value=meta["default"],
            help=meta.get("help"),
            key=feat,
        )

predict_btn = st.sidebar.button("🔍 Predict", use_container_width=True, type="primary")

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if predict_btn:
    X = pd.DataFrame([[inputs[f] for f in FEATURES]], columns=FEATURES)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    st.subheader("Result")

    col1, col2 = st.columns(2)
    with col1:
        if pred == 1:
            st.error("**Likely Alzheimer's Indicators Present**")
        else:
            st.success("**Low Risk / No Strong Indicators**")
    with col2:
        st.metric("Model Confidence", f"{prob * 100:.1f}%")

    st.progress(min(max(prob, 0.0), 1.0))

    with st.expander("See input summary"):
        st.dataframe(X.T.rename(columns={0: "Value"}), use_container_width=True)

    with st.expander("Why these 7 features?"):
        st.write(
            "These features were selected after combining four different "
            "methods — Chi-Square test, Independent T-Test, Mutual "
            "Information, and Random Forest importance — through a voting "
            "system. Only features that were consistently flagged as "
            "important across at least 3 of the 4 methods were kept, which "
            "reduces noise and avoids overfitting to any single statistical "
            "test."
        )

    # Feature importance chart from the trained model itself
    with st.expander("Model feature importance"):
        importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        st.bar_chart(importances)

else:
    st.write("Set the patient's values in the sidebar, then click **Predict**.")

st.divider()
st.caption("Built with Streamlit · Random Forest (200 trees) · Project P685")
