# IoT-Based Equipment Wear Prediction and Classification System for Cargo Logistics

**Application type:** Streamlit web application — fully interactive, deployable on Streamlit Community Cloud.

**Dataset columns:** Timestamp, DeviceID, Speed (kmph), Pressure (psi), Temperature (°C), Latitude, Longitude, Wear (mm) → target, Status, Obs_Obj, Collision, Type.

---

## Run locally (no backend required)

```bash
cd c:\Users\bhara\OneDrive\Desktop\phd
pip install -r requirements-streamlit.txt
streamlit run app.py
```

Open: **http://localhost:8501**

---

## UI flow (very important)

1. **Step 1 — Dataset upload & model selection (first action)**  
   - **A.** Upload CSV (no dataset loaded by default; user must upload).  
   - **B.** Select model (radio, single choice): Random Forest, Decision Tree, SVR, XGBoost, Logistic Regression, ANN, CNN, GRU.  
   - **C.** Click **Train Model**.

2. **Step 2 — Training status**  
   - After training: “Model trained successfully”, training time, selected model name, dataset size.

3. **Step 3 — Input section (enabled only after training)**  
   - User enters all parameters **except Wear (mm)**.  
   - Click **Predict Wear**.

4. **Step 4 — Prediction output & classification**  
   - Predicted Wear (mm).  
   - p-value: **p < 0.05 → Prediction is Reliable**, **p ≥ 0.05 → Prediction is Unreliable**.  
   - Condition: **Good** (green) / **Bad** (red).

5. **Step 5 — Performance metrics**  
   - Click **View Performance Metrics**.  
   - Regression: MAE, RMSE, R² (0.5–0.99 = Good), MAPE, Median AE, p-value, Prediction Accuracy.  
   - Classification: Confusion Matrix, Accuracy, Precision, Recall, F1, ROC & AUC (Logistic Regression).  
   - K-Fold CV: mean score, standard deviation, visualization.  
   - Root cause: Feature importance (Speed, Pressure, Temperature, Latitude, Longitude).  
   - Graphs: Bar charts (MAE, RMSE, R², MAPE), Actual vs Predicted line chart, Confusion Matrix heatmap, Feature importance, ROC curve.

---

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Sign in with GitHub, click **New app**.
4. **Repository:** your repo. **Branch:** main. **Main file path:** `app.py`. **App URL:** e.g. `https://iot-wear-prediction.streamlit.app`.
5. **Advanced settings:**  
   - Python version: 3.10 or 3.11.  
   - Use `requirements-streamlit.txt` (rename to `requirements.txt` in repo for Cloud, or set “Requirements file” to `requirements-streamlit.txt` if the UI allows).
6. Deploy. The app runs without local setup: upload dataset, train model, predict wear & condition.

**Note:** For Cloud, use `requirements-streamlit.txt` as your `requirements.txt` (no Flask/MySQL). TensorFlow is optional (comment out in requirements for faster deploy; ANN/CNN/GRU will be skipped).

---

## Project layout

```
phd/
├── app.py                      # Main Streamlit app (use this for Cloud)
├── ml_engine.py                # Self-contained ML/DL engine (no Flask/MySQL)
├── requirements-streamlit.txt  # For Streamlit Cloud / local run
├── packages.txt                # System packages (Streamlit Cloud)
├── backend/                    # Optional Flask + MySQL (local only)
│   ├── app.py
│   ├── db.py
│   ├── full_analysis.py
│   └── model_service.py
├── frontend/
│   └── app.py                  # Alternative Streamlit UI (uses backend)
├── database/
│   └── schema.sql
├── data/
└── README.md
```

---

## Optional: Flask backend + frontend (local)

Backend and frontend are wired correctly: the frontend calls the backend for health, data upload/list, train, predict/details, and metrics.

**To run backend + frontend:**

1. **Backend (Flask)** — from project root:
   ```bash
   pip install -r requirements.txt
   # Create MySQL DB (see database/schema.sql), set .env if needed
   python backend/app.py
   ```
   Backend runs at **http://127.0.0.1:5000** (or `FLASK_PORT` from `.env`).

2. **Frontend (Streamlit)** — in a second terminal:
   ```bash
   streamlit run frontend/app.py
   ```
   Open **http://localhost:8501**. The sidebar shows "API connected" when the backend is reachable.

3. **Flow:** Upload CSV (or use DB data) → Train models → Get Details (prediction + 3‑month forecast) → Performance Metrics. All frontend buttons use the backend API; no backend means "Backend unreachable" in the UI.
