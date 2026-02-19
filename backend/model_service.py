"""
Wear prediction model service: load data, preprocess, train, predict.
Supports both CSV column naming (Speed(kmph), Wear(mm), etc.) and normalized names.
"""
import io
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Map user CSV columns to normalized names (with/without spaces, and underscore variants)
COLUMN_MAP = {
    "Speed(kmph)": "speed_kmph", "Speed (kmph)": "speed_kmph", "Speed_kmph": "speed_kmph",
    "Pressure(psi)": "pressure_psi", "Pressure (psi)": "pressure_psi", "Pressure_psi": "pressure_psi",
    "Temperature(°C)": "temperature_c", "Temperature (°C)": "temperature_c", "Temperature_C": "temperature_c",
    "Wear(mm)": "wear_mm", "Wear (mm)": "wear_mm", "Wear_mm": "wear_mm",
    "Obs_Obj": "obs_obj", "Collision": "collision", "Type": "type_name",
    "Status": "status", "DeviceID": "device_id", "SensorID": "sensor_id",
    "Timestamp": "timestamp", "Latitude": "latitude", "Longitude": "longitude",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard names and ensure numeric where needed."""
    df = df.copy()
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    # Already normalized names
    for c in ["speed_kmph", "pressure_psi", "temperature_c", "latitude", "longitude", "wear_mm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Encode collision: Yes/No -> 1/0, or numeric 0/1 as-is
    if "collision" in df.columns:
        col = df["collision"]
        if pd.api.types.is_numeric_dtype(col):
            df["collision_bin"] = (col.fillna(0) != 0).astype(int)
        else:
            df["collision_bin"] = (col.astype(str).str.lower() == "yes").astype(int)
    else:
        df["collision_bin"] = 0
    return df

def prepare_features(df: pd.DataFrame):
    """Build feature matrix and target. Returns (X, y, feature_cols, encoders)."""
    df = normalize_columns(df)
    df = df.dropna(subset=["wear_mm", "speed_kmph", "pressure_psi", "temperature_c"])
    if df.empty or len(df) < 20:
        return None, None, None, None

    le_type = LabelEncoder()
    le_obs = LabelEncoder()
    le_status = LabelEncoder()
    if "type_name" in df.columns:
        df["type_enc"] = le_type.fit_transform(df["type_name"].astype(str))
    else:
        df["type_enc"] = 0
    if "obs_obj" in df.columns:
        df["obs_enc"] = le_obs.fit_transform(df["obs_obj"].astype(str))
    else:
        df["obs_enc"] = 0
    if "status" in df.columns:
        df["status_enc"] = le_status.fit_transform(df["status"].astype(str))
    else:
        df["status_enc"] = 0
    if "device_id" in df.columns:
        df["device_enc"] = LabelEncoder().fit_transform(df["device_id"].astype(str))
    else:
        df["device_enc"] = 0

    feature_cols = ["speed_kmph", "pressure_psi", "temperature_c", "latitude", "longitude",
                    "obs_enc", "collision_bin", "type_enc", "device_enc", "status_enc"]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    X = df[feature_cols].astype(float)
    y = df["wear_mm"]
    encoders = {"type": le_type, "obs": le_obs, "status": le_status}
    return X, y, feature_cols, encoders

def _compute_metrics(y_test, pred):
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    mape_val = np.mean(np.abs((y_test - pred) / (y_test + 1e-8))) * 100
    median_ae = np.median(np.abs(y_test - pred))
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape_val, 2),
        "Median_AE": round(float(median_ae), 4),
    }

def train_and_metrics(X, y, test_size=0.2):
    """Train RandomForest, return model, scaler, and metrics dict."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=RANDOM_STATE)
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    metrics = _compute_metrics(y_test, pred)
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))
    return model, scaler, metrics, list(X.columns)

def train_all_models(df: pd.DataFrame):
    """Train Random Forest and XGBoost; return all_metrics dict and (models, scalers, encoders)."""
    X, y, feature_cols, encoders = prepare_features(df)
    if X is None or len(X) < 20:
        return None, None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    all_metrics = {}
    models = {}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=RANDOM_STATE)
    rf.fit(X_train_s, y_train)
    pred_rf = rf.predict(X_test_s)
    all_metrics["Random Forest"] = _compute_metrics(y_test, pred_rf)
    all_metrics["Random Forest"]["n_train"] = int(len(X_train))
    all_metrics["Random Forest"]["n_test"] = int(len(X_test))
    models["Random Forest"] = (rf, scaler, feature_cols)

    # Decision Tree
    dt = DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE)
    dt.fit(X_train_s, y_train)
    pred_dt = dt.predict(X_test_s)
    all_metrics["Decision Tree"] = _compute_metrics(y_test, pred_dt)
    all_metrics["Decision Tree"]["n_train"] = int(len(X_train))
    all_metrics["Decision Tree"]["n_test"] = int(len(X_test))
    models["Decision Tree"] = (dt, scaler, feature_cols)

    # SVR
    svr = SVR(kernel="rbf", C=10, gamma="scale")
    svr.fit(X_train_s, y_train)
    pred_svr = svr.predict(X_test_s)
    all_metrics["SVR"] = _compute_metrics(y_test, pred_svr)
    all_metrics["SVR"]["n_train"] = int(len(X_train))
    all_metrics["SVR"]["n_test"] = int(len(X_test))
    models["SVR"] = (svr, scaler, feature_cols)

    # Logistic Regression (classification: approximate wear via threshold)
    lr_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    # binary target: Good if wear < GOOD_THRESHOLD
    y_train_cls = (y_train < GOOD_THRESHOLD).astype(int)
    y_test_cls = (y_test < GOOD_THRESHOLD).astype(int)
    try:
        lr_clf.fit(X_train_s, y_train_cls)
        y_pred_cls = lr_clf.predict(X_test_s)
        # approximate regression-style predicted wear for metrics comparison
        pred_wear_lr = y_pred_cls.astype(float) * (GOOD_THRESHOLD - 0.1) + (1 - y_pred_cls.astype(float)) * (GOOD_THRESHOLD + 0.5)
        all_metrics["Logistic Regression"] = _compute_metrics(y_test, pred_wear_lr)
        all_metrics["Logistic Regression"]["Accuracy"] = float((y_test_cls == y_pred_cls).mean())
        all_metrics["Logistic Regression"]["n_train"] = int(len(X_train))
        all_metrics["Logistic Regression"]["n_test"] = int(len(X_test))
        models["Logistic Regression"] = (lr_clf, scaler, feature_cols)
    except Exception:
        # If logistic fails, skip
        pass

    # XGBoost
    if XGB_AVAILABLE:
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_STATE)
        xgb_model.fit(X_train_s, y_train)
        pred_xgb = xgb_model.predict(X_test_s)
        all_metrics["XGBoost"] = _compute_metrics(y_test, pred_xgb)
        all_metrics["XGBoost"]["n_train"] = int(len(X_train))
        all_metrics["XGBoost"]["n_test"] = int(len(X_test))
        models["XGBoost"] = (xgb_model, scaler, feature_cols)

    return all_metrics, (models, encoders)

def run_from_dataframe(df: pd.DataFrame):
    """Run full pipeline: prepare -> train -> metrics. Returns (metrics, model, scaler, feature_cols, encoders) or (None,...) on failure."""
    X, y, feature_cols, encoders = prepare_features(df)
    if X is None or len(X) < 20:
        return None, None, None, None, None
    model, scaler, metrics, _ = train_and_metrics(X, y)
    return metrics, model, scaler, feature_cols, encoders

def run_from_csv_bytes(csv_bytes: bytes, filename: str = "upload.csv"):
    """Load CSV from bytes (Streamlit upload), then run_from_dataframe."""
    df = pd.read_csv(io.BytesIO(csv_bytes))
    return run_from_dataframe(df)

# In-memory cache for last trained model (so Flask can serve predictions without retraining every time)
_cached_model = None
_cached_scaler = None
_cached_feature_cols = None
_cached_encoders = None

# Last all-models metrics for Performance metrics page
_last_all_metrics = {}

def set_last_all_metrics(m):
    global _last_all_metrics
    _last_all_metrics = m or {}

def get_last_all_metrics():
    return _last_all_metrics

def set_cached_model(model, scaler, feature_cols, encoders=None):
    global _cached_model, _cached_scaler, _cached_feature_cols, _cached_encoders
    _cached_model, _cached_scaler = model, scaler
    _cached_feature_cols, _cached_encoders = feature_cols, encoders or {}

def get_cached_model():
    return _cached_model, _cached_scaler, _cached_feature_cols, _cached_encoders or {}

GOOD_THRESHOLD = 1.5  # Wear (mm) below this = Good

def wear_to_status(wear_mm: float, good_threshold: float = None) -> str:
    """Return 'Good' or 'Bad' based on predicted wear (mm)."""
    th = good_threshold if good_threshold is not None else GOOD_THRESHOLD
    return "Good" if wear_mm < th else "Bad"

def forecast_3_months(wear_now: float, monthly_rate: float = 0.03) -> list:
    """
    Time-series/trend-based forecast: Month 1, 2, 3.
    monthly_rate: wear increase per month (mm). Default 0.03 from typical degradation.
    Returns list of dicts: [{future_time, predicted_wear_mm, condition}, ...]
    """
    th = GOOD_THRESHOLD
    out = []
    for m in [0, 1, 2, 3]:
        if m == 0:
            w = wear_now
            label = "Now"
        else:
            w = wear_now + monthly_rate * m
            label = f"+{m} Month" + ("s" if m > 1 else "")
        out.append({
            "future_time": label,
            "predicted_wear_mm": round(float(w), 4),
            "condition": "Good" if w < th else "Bad",
        })
    return out

def predict_single(row_dict, model, scaler, feature_cols, encoders):
    """Predict wear_mm for one row (dict with keys matching CSV/DB). Returns float."""
    df = pd.DataFrame([row_dict])
    df = normalize_columns(df)
    for enc_name, le in encoders.items():
        col = "type_name" if enc_name == "type" else "obs_obj" if enc_name == "obs" else "status"
        if col in df.columns:
            v = str(df[col].iloc[0])
            df[f"{enc_name}_enc"] = le.transform([v])[0] if v in le.classes_ else 0
        else:
            df[f"{enc_name}_enc"] = 0
    df["device_enc"] = 0
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    X = df[feature_cols].astype(float)
    X_s = scaler.transform(X)
    # Logistic Regression predicts class (0=Good, 1=Bad); map to approximate wear (mm)
    if type(model).__name__ == "LogisticRegression":
        cls = model.predict(X_s)[0]
        return float(GOOD_THRESHOLD - 0.2 if cls == 0 else GOOD_THRESHOLD + 0.3)
    return float(model.predict(X_s)[0])
