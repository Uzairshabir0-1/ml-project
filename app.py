import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load bundle
bundle = joblib.load("model_bundle.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
columns = bundle["columns"]  # The 37 X columns

st.set_page_config(page_title="CTG Fetal Health Predictor", page_icon="🫀", layout="wide")
st.title("🫀 Fetal Health Predictor (CTG)")
st.markdown("Enter the cardiotocography values below to predict fetal health **(NSP)**.")

# NSP labels
nsp_labels = {1: "✅ Normal", 2: "⚠️ Suspect", 3: "🚨 Pathological"}

# Default values
defaults = {
    "b": 0.0, "e": 0.0, "LBE": 120.0, "LB": 120.0,
    "AC": 0.0, "FM": 0.0, "UC": 0.0, "DR": 0.0,
    "ASTV": 50.0, "MSTV": 1.0, "ALTV": 0.0, "MLTV": 5.0,
    "DL": 0.0, "DS": 0.0, "DP": 0.0,
    "Width": 60.0, "Min": 100.0, "Max": 160.0,
    "Nmax": 3.0, "Nzeros": 0.0, "Mode": 120.0,
    "Mean": 120.0, "Median": 120.0, "Variance": 10.0, "Tendency": 0.0,
    "A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0,
    "AD": 0.0, "DE": 0.0, "LD": 0.0, "FS": 0.0, "SUSP": 0.0, "CLASS": 0.0,
}

st.subheader("📋 Input Features")

# 3-column layout for inputs
cols = st.columns(3)
user_input = {}

for i, feat in enumerate(columns):
    with cols[i % 3]:
        user_input[feat] = st.number_input(
            label=feat,
            value=float(defaults.get(feat, 0.0)),
            format="%.2f",
            key=feat
        )

st.divider()

if st.button("🔍 Predict Fetal Health (NSP)", use_container_width=True):
    input_df = pd.DataFrame([user_input])[columns]  # Ensure correct column order

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.subheader("🧾 Prediction Result")
    label = nsp_labels.get(int(prediction), f"Class {prediction}")
    st.markdown(f"### {label}")

    # Confidence scores
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        st.subheader("📊 Confidence Scores")
        proba_df = pd.DataFrame({
            "Class": ["1 - Normal", "2 - Suspect", "3 - Pathological"],
            "Confidence": [f"{p*100:.1f}%" for p in proba],
            "Bar": proba
        })
        for _, row in proba_df.iterrows():
            st.write(f"**{row['Class']}** — {row['Confidence']}")
            st.progress(float(row["Bar"]))