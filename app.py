# SAFE FINAL VERSION FOR RENDER — FULL FEATURES KEPT

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import base64
import os
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

from ml_engine import (
    train_model,
    predict_one,
    condition_from_wear_and_pvalue,
    MODEL_NAMES,
    normalize_columns,
    prepare_features,
    GOOD_THRESHOLD,
)

# ---------------- SAFE BACKGROUND ----------------
def set_background(image_file="bg.jpg"):
    if not os.path.exists(image_file):
        return
    try:
        with open(image_file, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image:
                    linear-gradient(rgba(255,255,255,0.75), rgba(255,255,255,0.75)),
                    url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except:
        pass

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="IoT Equipment Wear Prediction Dashboard",
    page_icon="📦",
    layout="wide"
)

set_background("bg.jpg")

# ---------------- SESSION ----------------
for k, v in {
    "model_trained": False,
    "train_result": None,
    "uploaded_df": None,
    "show_metrics": False,
    "all_train_results": None,
    "best_model_name": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- TITLE ----------------
st.title("IoT Equipment Wear Prediction System")
st.caption("Predictive Maintenance using IoT Sensor Data")

# ---------------- UPLOAD ----------------
uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded:
    file_bytes = uploaded.getvalue()
    df = pd.read_csv(BytesIO(file_bytes))
    st.session_state.uploaded_df = df
    st.success(f"Loaded {len(df)} rows")
else:
    st.warning("Upload dataset first")

# ---------------- MODEL SELECTION ----------------
model_choice = st.radio(
    "Select model (or train all):",
    [
        "Random Forest",
        "Decision Tree",
        "SVR",
        "XGBoost",
        "Logistic Regression",
        "ANN",
        "GRU",
        "All Models",
    ],
    horizontal=True,
)

# ---------------- TRAIN ----------------
if st.button("Train Model"):
    if st.session_state.uploaded_df is None:
        st.error("Upload dataset first")
    else:
        with st.spinner("Training..."):
            if model_choice == "All Models":
                results = {}
                for m in [
                    "Random Forest",
                    "Decision Tree",
                    "SVR",
                    "XGBoost",
                    "Logistic Regression",
                ]:
                    r = train_model(st.session_state.uploaded_df, m)
                    if r:
                        results[m] = r

                best = max(results, key=lambda x: results[x]["metrics"]["Accuracy"])
                st.session_state.train_result = results[best]
                st.session_state.all_train_results = results
                st.success(f"Best model auto selected: {best}")
            else:
                st.session_state.train_result = train_model(
                    st.session_state.uploaded_df, model_choice
                )

            st.session_state.model_trained = True

# ---------------- PREDICT ----------------
if st.session_state.model_trained:

    st.subheader("Prediction Input")

    c1, c2, c3 = st.columns(3)

    with c1:
        speed = st.number_input("Speed", 0.0, 200.0, 33.0)
        pressure = st.number_input("Pressure", 0.0, 5000.0, 1200.0)

    with c2:
        temp = st.number_input("Temperature", 0.0, 200.0, 60.0)
        lat = st.number_input("Latitude", value=13.08)

    with c3:
        lon = st.number_input("Longitude", value=80.29)

    payload = {
        "Speed(kmph)": speed,
        "Pressure(psi)": pressure,
        "Temperature(°C)": temp,
        "Latitude": lat,
        "Longitude": lon,
    }

    if st.button("Predict Wear"):
        result = predict_one(st.session_state.train_result, payload)
        p = st.session_state.train_result["p_value"]
        cond, rel = condition_from_wear_and_pvalue(result, p)

        st.metric("Predicted Wear (mm)", round(result, 4))
        st.write("p-value:", p)
        st.write("Condition:", cond)
