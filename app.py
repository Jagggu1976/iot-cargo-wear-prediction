"""
IoT-Based Equipment Wear Prediction and Performance Analysis System for Cargo Logistics
Homepage: Title, subtitle, dataset upload, 9-model radio, sample table, manual input, Get Details,
prediction output (Good/Bad + 3-month forecast table & line graph), Performance Metrics (ALL models, graphs).
"""
import os
import sys
import requests
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000")

st.set_page_config(page_title="Wear Prediction", page_icon="📦", layout="wide")

def api_get(path, **kwargs):
    try:
        r = requests.get(f"{BACKEND_URL}{path}", timeout=60, **kwargs)
        return r.json() if r.headers.get("content-type", "").startswith("application/json") else None
    except Exception as e:
        st.error(f"Backend unreachable: {e}")
        return None

def api_post(path, json=None, files=None):
    try:
        if files:
            r = requests.post(f"{BACKEND_URL}{path}", files=files, timeout=180)
        else:
            r = requests.post(f"{BACKEND_URL}{path}", json=json, timeout=180)
        return r.json() if r.headers.get("content-type", "").startswith("application/json") else None
    except Exception as e:
        st.error(f"Backend error: {e}")
        return None

def main():
    # ---------- 1.1 Top Section: Center-aligned Project Title ----------
    st.markdown(
        "<h1 style='text-align: center; font-size: 1.75rem;'>IoT-Based Equipment Wear Prediction and Performance Analysis System for Cargo Logistics</h1>",
        unsafe_allow_html=True,
    )
    # ---------- 1.2 Subtitle ----------
    st.markdown(
        "<p style='text-align: center; color: #555; font-size: 1.05rem;'>Predictive Maintenance using IoT Sensor Data</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ---------- 1.2 Middle Section ----------
    # A. Dataset Upload: Button "Select Dataset", CSV only, user must upload
    st.subheader("A. Dataset Upload")
    uploaded_file = st.file_uploader(
        "**Select Dataset** (CSV only)",
        type=["csv"],
        help="Upload your IoT cargo dataset. Columns: Timestamp, DeviceID, SensorID, Speed(kmph), Pressure(psi), Temperature(°C), Latitude, Longitude, Wear(mm), Status, Obs_Obj, Collision, Type",
    )
    dataset_df = None
    if uploaded_file is not None:
        dataset_df = pd.read_csv(uploaded_file)
        st.success(f"Dataset loaded: **{uploaded_file.name}** — {len(dataset_df)} rows")
        if st.button("Upload to database"):
            r = requests.post(
                f"{BACKEND_URL}/api/data/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                timeout=120,
            )
            if r.ok and r.json().get("success"):
                st.success("Uploaded to database.")
            else:
                st.error(r.json().get("error", "Upload failed") if r.ok else "Request failed")
    else:
        data = api_get("/api/data/list?limit=100")
        if data and data.get("success") and data.get("data"):
            dataset_df = pd.DataFrame(data["data"])
            st.info("Showing data from database. Upload a CSV above to use your own dataset.")

    # B. Model Selection: Radio (single selection) — only fast/reliable models for live training
    st.subheader("B. Model Selection")
    model_choice = st.radio(
        "Select model (single selection):",
        options=[
            "Random Forest",
            "Decision Tree",
            "SVR",
            "XGBoost",
            "Logistic Regression",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )
    if st.button("Train models (main models; selected model used for predictions)"):
        try:
            if uploaded_file is not None:
                resp = requests.post(
                    f"{BACKEND_URL}/api/train",
                    params={"model": model_choice},
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                    timeout=300,
                )
            else:
                # By default do not request full DL models (ANN/CNN/GRU) — faster and safe for live use
                resp = requests.post(f"{BACKEND_URL}/api/train", params={"model": model_choice}, timeout=300)
            j = resp.json() if resp.ok else None
            if j and j.get("success"):
                st.success("All models trained. Selected model: **" + model_choice + "**")
                # store metrics and trained flag in session_state to avoid retraining
                st.session_state["trained"] = True
                st.session_state["selected_model"] = model_choice
                st.session_state["metrics"] = j.get("all_metrics", {})
                if j.get("all_metrics"):
                    with st.expander("View all metrics"):
                        st.json(j["all_metrics"])
            elif j:
                st.error(j.get("error", "Training failed"))
            else:
                st.error("Request failed.")
        except Exception as e:
            st.error(str(e))

    # C. Sample Dataset Preview: First 100 rows, scrollable table, all columns
    st.subheader("C. Sample Dataset Preview (first 100 rows)")
    display_cols = [
        "Timestamp", "DeviceID", "Speed(kmph)", "Pressure(psi)", "Temperature(°C)",
        "Latitude", "Longitude", "Wear(mm)", "Status", "Obs_Obj", "Collision", "Type",
    ]
    if dataset_df is not None:
        sample = dataset_df.head(100)
        # Normalize column names for display (backend may use different names)
        rename = {
            "Speed(kmph)": "Speed(kmph)", "Pressure(psi)": "Pressure(psi)",
            "Temperature(°C)": "Temperature(°C)", "Wear(mm)": "Wear(mm)",
        }
        cols_to_show = [c for c in display_cols if c in sample.columns] or list(sample.columns[:12])
        st.dataframe(sample[cols_to_show] if cols_to_show else sample, use_container_width=True, height=350)
    else:
        data = api_get("/api/data/list?limit=100")
        if data and data.get("success") and data.get("data"):
            df = pd.DataFrame(data["data"])
            col_map = {"speed_kmph": "Speed(kmph)", "pressure_psi": "Pressure(psi)", "temperature_c": "Temperature(°C)",
                       "wear_mm": "Wear(mm)", "device_id": "DeviceID", "timestamp": "Timestamp",
                       "latitude": "Latitude", "longitude": "Longitude", "status": "Status",
                       "obs_obj": "Obs_Obj", "collision": "Collision", "type_name": "Type"}
            df = df.rename(columns=col_map)
            st.dataframe(df.head(100), use_container_width=True, height=350)
        else:
            st.info("Upload a CSV to see the sample dataset.")

    st.divider()

    # ---------- Offline-trained Deep Learning Models (Comparative Analysis) ----------
    st.subheader("Offline-trained Deep Learning Models (Comparative Analysis)")
    st.markdown("These models (ANN, CNN, GRU) are heavy — metrics are loaded from offline training results.")
    if st.button("Load advanced/offline metrics"):
        off = api_get("/api/metrics_offline")
        if not off or not off.get("success"):
            st.info(off.get("error") if off else "No offline metrics available. Train offline and save advanced_metrics.json to backend/ or data/.")
        else:
            adv = off.get("offline_metrics", {})
            st.success("Loaded offline advanced metrics")
            # store offline metrics in session
            st.session_state["offline_metrics"] = adv
            # fetch live ML metrics for comparison
            live = api_get("/api/metrics") or {}
            live_metrics = live.get("all_metrics", {})
            # Build combined dataframe: ML live models + offline DL models
            rows = []
            for name, m in (live_metrics.items() if isinstance(live_metrics, dict) else []):
                rows.append({"Model": name, "Accuracy": m.get("Accuracy"), "RMSE": m.get("RMSE"), "R2": m.get("R2")})
            for name, m in adv.items():
                rows.append({"Model": name, "Accuracy": m.get("Accuracy"), "RMSE": m.get("RMSE"), "R2": m.get("R2")})
            if rows:
                df_comb = pd.DataFrame(rows).set_index("Model")
                st.dataframe(df_comb, use_container_width=True)
                import matplotlib.pyplot as plt
                import seaborn as sns
                sns.set(style="whitegrid")
                # Accuracy
                if "Accuracy" in df_comb.columns:
                    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
                    df_comb["Accuracy"].plot(kind="bar", ax=ax, color=sns.color_palette("muted", n_colors=len(df_comb)))
                    ax.set_ylabel("Accuracy"); ax.set_title("Accuracy — ML vs DL (offline)")
                    st.pyplot(fig); plt.close()
                # RMSE
                if "RMSE" in df_comb.columns:
                    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
                    df_comb["RMSE"].plot(kind="bar", ax=ax, color=sns.color_palette("muted", n_colors=len(df_comb)))
                    ax.set_ylabel("RMSE"); ax.set_title("RMSE — ML vs DL (offline)")
                    st.pyplot(fig); plt.close()
                # R²
                if "R2" in df_comb.columns:
                    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
                    df_comb["R2"].plot(kind="bar", ax=ax, color=sns.color_palette("muted", n_colors=len(df_comb)))
                    ax.set_ylabel("R²"); ax.set_title("R² — ML vs DL (offline)")
                    st.pyplot(fig); plt.close()

    # ---------- 2. Manual Input Section ----------
    st.subheader("2. Manual Input (enter all features except Wear)")
    col1, col2, col3 = st.columns(3)
    with col1:
        timestamp_str = st.text_input("Timestamp (current or selected date)", value="", placeholder="e.g. 01-05-2025 08:00")
        device_id = st.text_input("DeviceID", value="TT125")
        sensor_id = st.text_input("SensorID", value="SID014")
        speed = st.number_input("Speed (kmph)", value=33.0, min_value=0.0)
        pressure = st.number_input("Pressure (psi)", value=1200.0)
    with col2:
        temp = st.number_input("Temperature (°C)", value=60.0)
        lat = st.number_input("Latitude", value=13.08, format="%.4f")
        lon = st.number_input("Longitude", value=80.29, format="%.4f")
        status = st.selectbox("Status", ["Active", "Inactive"])
    with col3:
        obs_obj = st.selectbox("Obs_Obj", ["Vehicle", "Person", "Signal"])
        collision = st.selectbox("Collision", ["No", "Yes"])
        type_name = st.selectbox("Type", ["Sudden", "Planned"])
    # Wear(mm) must NOT be entered by the user — no input for Wear

    payload = {
        "DeviceID": device_id,
        "SensorID": sensor_id,
        "Speed(kmph)": speed,
        "Pressure(psi)": pressure,
        "Temperature(°C)": temp,
        "Latitude": lat,
        "Longitude": lon,
        "Status": status,
        "Obs_Obj": obs_obj,
        "Collision": collision,
        "Type": type_name,
    }
    if timestamp_str:
        payload["Timestamp"] = timestamp_str

    # ---------- 2.2 Action Button: Get Details ----------
    if st.button("**Get Details**"):
        resp = api_post("/api/details", json=payload)
        if resp and resp.get("success"):
            pred_wear = resp.get("predicted_wear_mm", 0)
            cond = resp.get("status", "—")
            forecast = resp.get("forecast_3months", [])

            # store last prediction in session state
            st.session_state["last_prediction"] = pred_wear
            st.session_state["last_prediction_status"] = cond
            st.session_state["last_forecast"] = forecast

            # 3.1 Wear Prediction (Immediate)
            st.subheader("3. Prediction Output")
            st.markdown("**3.1 Wear Prediction (Immediate)**")
            st.metric("Predicted Wear (mm)", round(pred_wear, 4))
            st.markdown(f"**Equipment Condition:** {'Good — Wear within safe threshold' if cond == 'Good' else 'Bad — Wear exceeds threshold'}")

            # 3.2 & 3.3 Wear Forecast (Next 3 Months) — table
            st.markdown("**3.2–3.3 Wear Forecast (Next 3 Months)** — Time-series/trend-based prediction (Month 1, 2, 3).")
            if forecast:
                tbl = pd.DataFrame(forecast)
                st.dataframe(tbl, use_container_width=True, hide_index=True)
                # 3.4 Forecast Graph: Line — X: Now, +1, +2, +3 Months; Y: Wear (mm); Green = Good, Red = Bad
                times = [t["future_time"] for t in forecast]
                wears = [t["predicted_wear_mm"] for t in forecast]
                conditions = [t["condition"] for t in forecast]
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
                ax.plot(times, wears, "o-", color="gray", linewidth=2, markersize=8)
                for i, (t, w, c) in enumerate(zip(times, wears, conditions)):
                    ax.scatter([t], [w], color="green" if c == "Good" else "red", s=90, zorder=5, edgecolors="black")
                ax.axhline(y=1.5, color="orange", linestyle="--", label="Threshold (1.5 mm)")
                ax.set_xlabel("Time")
                ax.set_ylabel("Wear (mm)")
                ax.set_title("3-Month Wear Forecast (Green = Good, Red = Bad)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No forecast available.")
        else:
            st.error(resp.get("error", "Train a model first, then click Get Details.") if resp else "Backend unreachable.")

    st.divider()

    # ---------- 4. Performance Metrics Section ----------
    st.subheader("4. Performance Metrics (ALL models)")
    if st.button("**Performance Metrics**"):
        resp = api_get("/api/metrics")
        if resp and resp.get("success"):
            # cache metrics in session state to avoid unnecessary re-fetch
            st.session_state["metrics"] = resp.get("all_metrics", {})
            st.session_state["cv_results"] = resp.get("cv_results", {})
            st.session_state["classification_metrics"] = resp.get("classification_metrics", {})
            st.session_state["roc_data"] = resp.get("roc_data", {})
            st.session_state["feature_importance"] = resp.get("feature_importance", {})
            st.session_state["actual_vs_predicted"] = resp.get("actual_vs_predicted", {})
            all_metrics = resp.get("all_metrics", {})
            if not all_metrics:
                st.info("Train models first to see metrics.")
            else:
                # 4.2 Show metrics for ALL models (table)
                rows = []
                for name, m in all_metrics.items():
                    rows.append({
                        "Model": name,
                        "MAE": m.get("MAE"),
                        "RMSE": m.get("RMSE"),
                        "R²": m.get("R2"),
                        "MAPE": m.get("MAPE"),
                        "Median AE": m.get("Median_AE"),
                        "Accuracy": m.get("Accuracy"),
                        "p-value": m.get("p_value"),
                    })
                df_m = pd.DataFrame(rows)
                st.dataframe(df_m, use_container_width=True, hide_index=True)
                # Bar charts: MAE, RMSE, R², MAPE comparison — improved visuals
                st.markdown("**Bar charts: MAE, RMSE, R², MAPE comparison**")
                import matplotlib.pyplot as plt
                import seaborn as sns
                sns.set(style="whitegrid")
                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
                    df_plot = df_m.set_index("Model")[["MAE", "RMSE"]]
                    df_plot.plot(kind="bar", ax=ax, color=sns.color_palette("muted", n_colors=len(df_plot)))
                    ax.set_ylabel("Error")
                    ax.set_title("MAE and RMSE by Model")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close()
                with c2:
                    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
                    r2col = "R²" if "R²" in df_m.columns else "R2"
                    df_plot2 = df_m.set_index("Model")[[r2col, "MAPE"]]
                    df_plot2.plot(kind="bar", ax=ax, color=sns.color_palette("muted", n_colors=len(df_plot2)))
                    ax.set_title("R² and MAPE by Model")
                    ax.set_ylabel(r2col)
                    st.pyplot(fig)
                    plt.close()
                # Actual vs Predicted (line chart)
                avp = resp.get("actual_vs_predicted", {})
                if avp and avp.get("actual") and avp.get("predicted"):
                    st.markdown("**Actual vs Predicted Wear (line chart)**")
                    avp_df = pd.DataFrame({"Actual": avp["actual"], "Predicted": avp["predicted"]})
                    st.line_chart(avp_df)
                # K-Fold CV
                cv = resp.get("cv_results", {})
                if cv:
                    st.markdown("**K-Fold Cross Validation (mean ± std)**")
                    cv_rows = [{"Model": k, "CV MAE mean": v.get("MAE_mean"), "CV MAE std": v.get("MAE_std"), "CV R² mean": v.get("R2_mean"), "CV R² std": v.get("R2_std")} for k, v in cv.items()]
                    st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)
                    cv_df = pd.DataFrame(cv_rows).set_index("Model")
                    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
                    cv_df[["CV MAE mean", "CV R² mean"]].plot(kind="bar", ax=ax, color=sns.color_palette("pastel", n_colors=len(cv_df)))
                    ax.set_title("K-Fold CV: MAE mean and R² mean")
                    ax.set_ylabel("Value")
                    st.pyplot(fig)
                    plt.close()
                # Classification: ROC & AUC, Confusion Matrix
                clf = resp.get("classification_metrics", {})
                roc = resp.get("roc_data", {})
                if clf:
                    st.markdown("**Classification (Logistic Regression): Accuracy, Precision, Recall, F1, AUC**")
                    st.write(f"Accuracy: {clf.get('accuracy')}, Precision: {clf.get('precision')}, Recall: {clf.get('recall')}, F1: {clf.get('f1')}, AUC: {clf.get('auc')}")
                    if clf.get("confusion_matrix"):
                        import seaborn as sns
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots()
                        sns.heatmap(clf["confusion_matrix"], annot=True, fmt="d", ax=ax, xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"])
                        ax.set_title("Confusion Matrix")
                        st.pyplot(fig)
                        plt.close()
                if roc and roc.get("fpr") and roc.get("tpr"):
                    st.markdown("**ROC Curve (AUC = " + str(round(roc.get("auc", 0), 3)) + ")**")
                    roc_df = pd.DataFrame({"FPR": roc["fpr"], "TPR": roc["tpr"]})
                    st.line_chart(roc_df)
                # Feature Importance (Root Cause Analysis)
                fi = resp.get("feature_importance", {})
                if fi:
                    st.markdown("**Feature Importance (Root Cause: Speed, Pressure, Temperature, Location)**")
                    for model_name, imp in fi.items():
                        fi_df = pd.DataFrame(list(imp.items()), columns=["Feature", "Importance"]).set_index("Feature")
                        fig, ax = plt.subplots(figsize=(8, 3), dpi=110)
                        fi_df.sort_values("Importance", ascending=True).plot(kind="barh", ax=ax, legend=False, color=sns.color_palette("deep", n_colors=len(fi_df)))
                        ax.set_title(f"Feature importance — {model_name}")
                        st.pyplot(fig)
                        plt.close()
        else:
            st.info("No metrics yet. Train models first.")

    with st.sidebar:
        st.header("Backend")
        health = api_get("/health")
        st.success("API connected") if health else st.warning("Start Flask: python backend/app.py")
        st.caption(f"BACKEND_URL = {BACKEND_URL}")
        st.markdown("All visualizations are dynamically generated at runtime; deep learning metrics are loaded from offline training results for comparative analysis.")

if __name__ == "__main__":
    main()
