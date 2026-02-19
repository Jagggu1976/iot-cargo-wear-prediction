"""
Flask backend for IoT cargo wear prediction.
Endpoints: health, data (load CSV into MySQL, list), train, predict, runs.
"""
import os
import sys
import json
import io
import uuid

import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Add parent so we can import backend.db and backend.model_service
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db import get_connection, init_db
from datetime import datetime, timedelta
from backend.model_service import (
    run_from_dataframe,
    train_all_models,
    set_cached_model,
    get_cached_model,
    set_last_all_metrics,
    get_last_all_metrics,
    wear_to_status,
    forecast_3_months,
    normalize_columns,
    prepare_features,
    predict_single as model_predict_single,
)

# Full analysis (all 9 models, CV, classification, ROC, feature importance)
_full_analysis_result = None
_selected_model_name = "Random Forest"

def set_full_analysis_result(result, selected_model="Random Forest"):
    global _full_analysis_result, _selected_model_name
    _full_analysis_result = result
    _selected_model_name = selected_model

def get_full_analysis_result():
    return _full_analysis_result, _selected_model_name

def predict_with_stored_model(row_dict):
    """Predict wear using stored full-analysis model (sklearn or Keras)."""
    res, selected = get_full_analysis_result()
    if res is None or "all_models" not in res or selected not in res["all_models"]:
        # Fallback to cached model from train_all_models
        model, scaler, feature_cols, encoders = get_cached_model()
        if model is None:
            raise ValueError("No model trained. Train first.")
        return model_predict_single(row_dict, model, scaler, feature_cols, encoders)
    model, scaler, feature_cols = res["all_models"][selected]
    encoders = res.get("encoders", {})
    # Build feature vector
    from backend.model_service import normalize_columns
    import pandas as pd
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
    # Logistic Regression: classifier (0=Good, 1=Bad) -> map to approximate wear for forecast
    if selected == "Logistic Regression":
        cls = model.predict(X_s)[0]
        from backend.model_service import GOOD_THRESHOLD
        return float(GOOD_THRESHOLD - 0.2 if cls == 0 else GOOD_THRESHOLD + 0.3)
    # Keras models (ANN, CNN, GRU)
    if selected in ("ANN", "CNN", "GRU"):
        if selected == "ANN":
            pred = model.predict(X_s, verbose=0)
        else:
            seq = X_s.reshape(1, 1, X_s.shape[1])
            pred = model.predict(seq, verbose=0)
        return float(pred.ravel()[0])
    return float(model.predict(X_s)[0])

app = Flask(__name__)
CORS(app)

# Ensure DB tables exist on startup
@app.before_request
def _ensure_db():
    try:
        init_db()
    except Exception:
        pass

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "wear-prediction-api"})

@app.route("/api/init-db", methods=["POST"])
def api_init_db():
    try:
        init_db()
        return jsonify({"success": True, "message": "DB initialized"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def _read_upload():
    if "file" in request.files:
        f = request.files["file"]
        if f.filename and f.filename.endswith(".csv"):
            return pd.read_csv(f), f.filename
    if request.get_data():
        try:
            return pd.read_csv(io.BytesIO(request.get_data())), "pasted.csv"
        except Exception:
            pass
    return None, None

def _df_to_db_rows(df):
    """Convert normalized DataFrame to list of tuples for INSERT."""
    df = normalize_columns(df)
    cols = ["timestamp", "device_id", "sensor_id", "speed_kmph", "pressure_psi", "temperature_c",
            "latitude", "longitude", "wear_mm", "status", "obs_obj", "collision", "type_name"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df["sensor_id"] = df.get("sensor_id", "")
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r.get("timestamp", "")),
            str(r.get("device_id", "")),
            str(r.get("sensor_id", "")),
            float(r["speed_kmph"]) if pd.notna(r["speed_kmph"]) else None,
            float(r["pressure_psi"]) if pd.notna(r["pressure_psi"]) else None,
            float(r["temperature_c"]) if pd.notna(r["temperature_c"]) else None,
            float(r["latitude"]) if pd.notna(r["latitude"]) else None,
            float(r["longitude"]) if pd.notna(r["longitude"]) else None,
            float(r["wear_mm"]) if pd.notna(r["wear_mm"]) else None,
            str(r.get("status", "")),
            str(r.get("obs_obj", "")),
            str(r.get("collision", "")),
            str(r.get("type_name", "")),
        ))
    return rows

@app.route("/api/data/upload", methods=["POST"])
def api_data_upload():
    """Upload CSV: store in MySQL and optionally train model."""
    df, filename = _read_upload()
    if df is None:
        return jsonify({"success": False, "error": "No CSV file or invalid data"}), 400
    try:
        rows = _df_to_db_rows(df)
    except Exception as e:
        return jsonify({"success": False, "error": f"Bad CSV format: {e}"}), 400
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            for i in range(0, len(rows), 500):
                chunk = rows[i : i + 500]
                cur.executemany(
                    """INSERT INTO sensor_readings
                       (timestamp, device_id, sensor_id, speed_kmph, pressure_psi, temperature_c,
                        latitude, longitude, wear_mm, status, obs_obj, collision, type_name)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    chunk,
                )
            conn.commit()
            cur.execute("SELECT COUNT(*) FROM sensor_readings")
            total = cur.fetchone()[0]
            cur.close()
        return jsonify({
            "success": True,
            "rows_inserted": len(rows),
            "total_rows": total,
            "filename": filename or "upload.csv",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/data/list", methods=["GET"])
def api_data_list():
    """List sensor_readings with optional limit and offset."""
    limit = min(int(request.args.get("limit", 100)), 5000)
    offset = int(request.args.get("offset", 0))
    try:
        with get_connection() as conn:
            cur = conn.cursor(dictionary=True)
            cur.execute(
                "SELECT id, timestamp, device_id, sensor_id, speed_kmph, pressure_psi, temperature_c, "
                "latitude, longitude, wear_mm, status, obs_obj, collision, type_name FROM sensor_readings "
                "ORDER BY id DESC LIMIT %s OFFSET %s",
                (limit, offset),
            )
            rows = cur.fetchall()
            cur.execute("SELECT COUNT(*) AS c FROM sensor_readings")
            total = cur.fetchone()["c"]
            cur.close()
        # Convert decimals for JSON
        for r in rows:
            for k, v in r.items():
                if hasattr(v, "__float__") and not isinstance(v, (int, float)):
                    r[k] = float(v) if v is not None else None
        return jsonify({"success": True, "data": rows, "total": total})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/data/export", methods=["GET"])
def api_data_export():
    """Export sensor_readings as CSV."""
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT timestamp, device_id, sensor_id, speed_kmph, pressure_psi, temperature_c, "
                "latitude, longitude, wear_mm, status, obs_obj, collision, type_name FROM sensor_readings"
            )
            rows = cur.fetchall()
            cur.close()
        df = pd.DataFrame(
            rows,
            columns=[
                "timestamp", "device_id", "sensor_id", "speed_kmph", "pressure_psi", "temperature_c",
                "latitude", "longitude", "wear_mm", "status", "obs_obj", "collision", "type_name",
            ],
        )
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="sensor_export.csv")
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/train", methods=["POST"])
def api_train():
    """Train ALL models (RF, DT, SVR, XGBoost, KNN, Logistic Regression, ANN, CNN, GRU). ?model=... sets prediction model."""
    df = None
    source = "upload"
    if "file" in request.files:
        f = request.files["file"]
        if f.filename and f.filename.endswith(".csv"):
            df = pd.read_csv(f)
            source = f.filename
    if df is None:
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT timestamp, device_id, sensor_id, speed_kmph, pressure_psi, temperature_c, "
                    "latitude, longitude, wear_mm, status, obs_obj, collision, type_name FROM sensor_readings"
                )
                rows = cur.fetchall()
                cur.close()
            if not rows:
                return jsonify({"success": False, "error": "No data in database. Upload CSV first."}), 400
            df = pd.DataFrame(
                rows,
                columns=[
                    "timestamp", "device_id", "sensor_id", "speed_kmph", "pressure_psi", "temperature_c",
                    "latitude", "longitude", "wear_mm", "status", "obs_obj", "collision", "type_name",
                ],
            )
            source = "database"
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    selected = request.args.get("model", "Random Forest")
    err_msg = "Not enough valid rows or analysis failed."
    try:
        from backend.full_analysis import run_full_analysis
        # Full analysis (including ANN/CNN/GRU) is expensive — only run when explicitly requested
        run_full = request.args.get("full", "false").lower() in ("1", "true", "yes")
        result = run_full and run_full_analysis(df) or None
    except Exception as e:
        result = None
        err_msg = str(e)

    if result is None:
        # Fallback: train_all_models (RF + XGBoost only)
        all_metrics, models_enc = train_all_models(df)
        if all_metrics is None:
            res = run_from_dataframe(df)
            if res[0] is None:
                return jsonify({"success": False, "error": err_msg if result is None else "Not enough valid rows."}), 400
            metrics, model, scaler, feature_cols, encoders = res
            all_metrics = {"Random Forest": metrics}
            set_cached_model(model, scaler, feature_cols, encoders)
            set_last_all_metrics(all_metrics)
            set_full_analysis_result(None, "Random Forest")
        else:
            models_dict, encoders = models_enc
            set_last_all_metrics(all_metrics)
            sel = selected if selected in models_dict else list(models_dict.keys())[0]
            model, scaler, feature_cols = models_dict[sel]
            set_cached_model(model, scaler, feature_cols, encoders)
            set_full_analysis_result(None, sel)
        run_name = f"run_{uuid.uuid4().hex[:8]}"
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO prediction_runs (run_name, dataset_source, model_name, metrics_json) VALUES (%s,%s,%s,%s)",
                    (run_name, source, selected, json.dumps(all_metrics)),
                )
                conn.commit()
                cur.close()
        except Exception:
            pass
        return jsonify({"success": True, "all_metrics": all_metrics, "run_name": run_name, "selected_model": selected, "source": source})

    # Full analysis succeeded
    all_metrics = result["all_metrics"]
    set_last_all_metrics(all_metrics)
    sel = selected if selected in result["all_models"] else list(result["all_models"].keys())[0]
    set_full_analysis_result(result, sel)
    model, scaler, feature_cols = result["all_models"][sel]
    set_cached_model(model, scaler, feature_cols, result.get("encoders"))
    run_name = f"run_{uuid.uuid4().hex[:8]}"
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO prediction_runs (run_name, dataset_source, model_name, metrics_json) VALUES (%s,%s,%s,%s)",
                (run_name, source, sel, json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"} for k, v in all_metrics.items()})),
            )
            conn.commit()
            cur.close()
    except Exception:
        pass
    payload = {
        "success": True,
        "all_metrics": all_metrics,
        "run_name": run_name,
        "selected_model": sel,
        "source": source,
    }
    if "cv_results" in result:
        payload["cv_results"] = result["cv_results"]
    if "classification_metrics" in result:
        payload["classification_metrics"] = result["classification_metrics"]
    return jsonify(payload)

@app.route("/api/details", methods=["POST"])
def api_details():
    """Predict wear (immediate), Good/Bad, and 3-month forecast (Month 1, 2, 3) with conditions."""
    body = request.get_json() or {}
    try:
        pred = predict_with_stored_model(body)
        status = wear_to_status(pred)
        forecast = forecast_3_months(pred, monthly_rate=0.03)
        return jsonify({
            "success": True,
            "predicted_wear_mm": round(pred, 4),
            "status": status,
            "forecast_3months": forecast,
            "message": f"Equipment Condition: {status}. Wear forecast (time-series/trend-based) for +1, +2, +3 months.",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Return ALL models' metrics, CV, classification (ROC/AUC, Confusion Matrix), feature importance, actual vs predicted."""
    res, _ = get_full_analysis_result()
    all_metrics = get_last_all_metrics()
    if not all_metrics and (res is None or not res):
        return jsonify({"success": True, "all_metrics": {}, "message": "Train models first."})
    payload = {"success": True, "all_metrics": all_metrics or {}}
    if res:
        payload["cv_results"] = res.get("cv_results", {})
        payload["classification_metrics"] = res.get("classification_metrics", {})
        payload["roc_data"] = res.get("roc_data", {})
        payload["feature_importance"] = res.get("feature_importance", {})
        payload["actual_vs_predicted"] = res.get("actual_vs_predicted", {})
    return jsonify(payload)


@app.route("/api/metrics_offline", methods=["GET"])
def api_metrics_offline():
    """Load precomputed / offline metrics for advanced models (ANN/CNN/GRU) from disk.
    Look for backend/advanced_metrics.json or data/advanced_metrics.json.
    """
    candidates = [
        os.path.join(os.path.dirname(__file__), "advanced_metrics.json"),
        os.path.join(os.getcwd(), "data", "advanced_metrics.json"),
    ]
    metrics = None
    for p in candidates:
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as fh:
                    metrics = json.load(fh)
                    break
        except Exception:
            continue
    if not metrics:
        return jsonify({"success": False, "error": "No offline advanced metrics found. Place advanced_metrics.json in backend/ or data/ folder."})
    return jsonify({"success": True, "offline_metrics": metrics})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict wear_mm for one row (JSON body) or bulk (JSON array)."""
    model, scaler, feature_cols, encoders = get_cached_model()
    if model is None:
        return jsonify({"success": False, "error": "No model trained. Call /api/train first."}), 400

    body = request.get_json() or {}
    if isinstance(body, list):
        out = []
        for row in body:
            try:
                pred = model_predict_single(row, model, scaler, feature_cols, encoders)
                out.append({"predicted_wear_mm": pred})
            except Exception as e:
                out.append({"error": str(e)})
        return jsonify({"success": True, "predictions": out})
    else:
        try:
            pred = model_predict_single(body, model, scaler, feature_cols, encoders)
            return jsonify({"success": True, "predicted_wear_mm": pred})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/runs", methods=["GET"])
def api_runs():
    """List recent prediction runs."""
    limit = min(int(request.args.get("limit", 20)), 100)
    try:
        with get_connection() as conn:
            cur = conn.cursor(dictionary=True)
            cur.execute(
                "SELECT id, run_name, dataset_source, model_name, metrics_json, created_at FROM prediction_runs ORDER BY id DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
            cur.close()
        for r in rows:
            if r.get("metrics_json"):
                r["metrics"] = json.loads(r["metrics_json"])
            if r.get("created_at"):
                r["created_at"] = r["created_at"].isoformat()
        return jsonify({"success": True, "runs": rows})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

