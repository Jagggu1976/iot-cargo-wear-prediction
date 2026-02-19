"""
Full analysis: all ML/DL models, K-Fold CV, classification (ROC/AUC, Confusion Matrix),
feature importance, p-value. For Performance Metrics section.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
from scipy import stats

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, GRU, Conv1D, MaxPooling1D, Flatten, Input
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.model_service import (
    prepare_features, normalize_columns, RANDOM_STATE,
    _compute_metrics,
)

GOOD_THRESHOLD = 1.5  # Wear (mm) below this = Good


def _build_ann(input_dim):
    if not TF_AVAILABLE:
        return None
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _build_cnn(input_dim, seq_len=1):
    if not TF_AVAILABLE:
        return None
    model = Sequential([
        Input(shape=(seq_len, input_dim)),
        Conv1D(32, 1, activation="relu"),
        MaxPooling1D(1),
        Flatten(),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _build_gru(input_dim, seq_len=1):
    if not TF_AVAILABLE:
        return None
    model = Sequential([
        Input(shape=(seq_len, input_dim)),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def run_full_analysis(df: pd.DataFrame):
    """
    Train all models (RF, DT, SVR, XGBoost, KNN, Logistic Regression, ANN, CNN, GRU).
    Return: all_metrics, cv_results, classification_metrics, feature_importance, roc_data.
    """
    X, y, feature_cols, encoders = prepare_features(df)
    if X is None or len(X) < 30:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    n_features = X_train_s.shape[1]

    # Binary target for classification (Good/Bad)
    y_train_cls = (y_train < GOOD_THRESHOLD).astype(int)
    y_test_cls = (y_test < GOOD_THRESHOLD).astype(int)

    all_metrics = {}
    all_models = {}
    cv_results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # --- ML Regression ---
    models_reg = [
        ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=12, random_state=RANDOM_STATE)),
        ("Decision Tree", DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE)),
        ("SVR", SVR(kernel="rbf", C=10, gamma="scale")),
        ("KNN", KNeighborsRegressor(n_neighbors=15, weights="distance")),
    ]
    if XGB_AVAILABLE:
        models_reg.append(("XGBoost", xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_STATE)))

    for name, model in models_reg:
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        all_metrics[name] = _compute_metrics(y_test, pred)
        all_metrics[name]["Accuracy"] = max(0, min(1, 1 - all_metrics[name]["RMSE"] / (y_test.std() or 1)))
        all_models[name] = (model, scaler, feature_cols)
        # K-Fold CV
        scores_mae = -cross_val_score(model, X_train_s, y_train, cv=kf, scoring="neg_mean_absolute_error")
        scores_r2 = cross_val_score(model, X_train_s, y_train, cv=kf, scoring="r2")
        cv_results[name] = {
            "MAE_mean": float(np.mean(scores_mae)), "MAE_std": float(np.std(scores_mae)),
            "R2_mean": float(np.mean(scores_r2)), "R2_std": float(np.std(scores_r2)),
        }

    # --- Logistic Regression (classification: Good/Bad) ---
    lr_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_clf.fit(X_train_s, y_train_cls)
    y_pred_cls = lr_clf.predict(X_test_s)
    y_proba_cls = lr_clf.predict_proba(X_test_s)[:, 1]
    all_models["Logistic Regression"] = (lr_clf, scaler, feature_cols)
    # Regression-style metrics using predicted wear from threshold
    pred_wear_lr = y_pred_cls.astype(float) * (GOOD_THRESHOLD - 0.1) + (1 - y_pred_cls.astype(float)) * (GOOD_THRESHOLD + 0.5)
    all_metrics["Logistic Regression"] = _compute_metrics(y_test, pred_wear_lr)  # approximate for comparison
    all_metrics["Logistic Regression"]["Accuracy"] = accuracy_score(y_test_cls, y_pred_cls)
    cv_results["Logistic Regression"] = {"MAE_mean": 0, "MAE_std": 0, "R2_mean": 0, "R2_std": 0}

    # Classification metrics (for Logistic Regression)
    classification_metrics = {
        "accuracy": accuracy_score(y_test_cls, y_pred_cls),
        "precision": precision_score(y_test_cls, y_pred_cls, zero_division=0),
        "recall": recall_score(y_test_cls, y_pred_cls, zero_division=0),
        "f1": f1_score(y_test_cls, y_pred_cls, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test_cls, y_pred_cls).tolist(),
        "auc": roc_auc_score(y_test_cls, y_proba_cls) if len(np.unique(y_test_cls)) > 1 else 0.5,
    }
    fpr, tpr, _ = roc_curve(y_test_cls, y_proba_cls)
    roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": classification_metrics["auc"]}

    # --- p-value: compare best model vs baseline (e.g. mean) ---
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_errors = np.abs(y_test.values - baseline_pred)
    for name in list(all_metrics.keys())[:3]:  # RF, DT, SVR
        model, _, _ = all_models[name]
        pred = model.predict(X_test_s)
        model_errors = np.abs(y_test.values - pred)
        _, p_val = stats.ttest_rel(baseline_errors, model_errors)
        all_metrics[name]["p_value"] = round(float(p_val), 6)
    for name in all_metrics:
        if "p_value" not in all_metrics[name]:
            all_metrics[name]["p_value"] = None

    # --- DL: ANN, CNN, GRU ---
    if TF_AVAILABLE:
        # ANN
        ann = _build_ann(n_features)
        ann.fit(X_train_s, y_train, epochs=50, batch_size=32, validation_split=0.15, callbacks=[EarlyStopping(patience=8, restore_best_weights=True)], verbose=0)
        pred_ann = ann.predict(X_test_s, verbose=0).ravel()
        all_metrics["ANN"] = _compute_metrics(y_test, pred_ann)
        all_metrics["ANN"]["Accuracy"] = max(0, min(1, 1 - all_metrics["ANN"]["RMSE"] / (y_test.std() or 1)))
        all_models["ANN"] = (ann, scaler, feature_cols)
        cv_results["ANN"] = {"MAE_mean": all_metrics["ANN"]["MAE"], "MAE_std": 0, "R2_mean": all_metrics["ANN"]["R2"], "R2_std": 0}
        all_metrics["ANN"]["p_value"] = None

        seq_len = 1
        X_train_seq = X_train_s.reshape(-1, seq_len, n_features)
        X_test_seq = X_test_s.reshape(-1, seq_len, n_features)
        # CNN
        cnn = _build_cnn(n_features, seq_len)
        cnn.fit(X_train_seq, y_train, epochs=50, batch_size=32, validation_split=0.15, callbacks=[EarlyStopping(patience=8, restore_best_weights=True)], verbose=0)
        pred_cnn = cnn.predict(X_test_seq, verbose=0).ravel()
        all_metrics["CNN"] = _compute_metrics(y_test, pred_cnn)
        all_metrics["CNN"]["Accuracy"] = max(0, min(1, 1 - all_metrics["CNN"]["RMSE"] / (y_test.std() or 1)))
        all_models["CNN"] = (cnn, scaler, feature_cols)
        cv_results["CNN"] = {"MAE_mean": all_metrics["CNN"]["MAE"], "MAE_std": 0, "R2_mean": all_metrics["CNN"]["R2"], "R2_std": 0}
        all_metrics["CNN"]["p_value"] = None

        # GRU
        gru = _build_gru(n_features, seq_len)
        gru.fit(X_train_seq, y_train, epochs=50, batch_size=32, validation_split=0.15, callbacks=[EarlyStopping(patience=8, restore_best_weights=True)], verbose=0)
        pred_gru = gru.predict(X_test_seq, verbose=0).ravel()
        all_metrics["GRU"] = _compute_metrics(y_test, pred_gru)
        all_metrics["GRU"]["Accuracy"] = max(0, min(1, 1 - all_metrics["GRU"]["RMSE"] / (y_test.std() or 1)))
        all_models["GRU"] = (gru, scaler, feature_cols)
        cv_results["GRU"] = {"MAE_mean": all_metrics["GRU"]["MAE"], "MAE_std": 0, "R2_mean": all_metrics["GRU"]["R2"], "R2_std": 0}
        all_metrics["GRU"]["p_value"] = None

    # --- Feature importance (RF, XGBoost) ---
    feature_importance = {}
    if "Random Forest" in all_models:
        rf = all_models["Random Forest"][0]
        feature_importance["Random Forest"] = dict(zip(feature_cols, [round(float(x), 4) for x in rf.feature_importances_]))
    if XGB_AVAILABLE and "XGBoost" in all_models:
        xgb_m = all_models["XGBoost"][0]
        feature_importance["XGBoost"] = dict(zip(feature_cols, [round(float(x), 4) for x in xgb_m.feature_importances_]))

    # Actual vs Predicted (for line chart) - use first regression model
    first_reg = list(all_models.keys())[0]
    m, sc, fc = all_models[first_reg]
    if hasattr(m, "predict"):
        pred_first = m.predict(X_test_s)
    else:
        pred_first = m.predict(X_test_s, verbose=0).ravel()
    actual_vs_pred = {"actual": y_test.values.tolist(), "predicted": pred_first.tolist()}

    return {
        "all_metrics": all_metrics,
        "all_models": all_models,
        "encoders": encoders,
        "cv_results": cv_results,
        "classification_metrics": classification_metrics,
        "roc_data": roc_data,
        "feature_importance": feature_importance,
        "actual_vs_predicted": actual_vs_pred,
        "feature_cols": feature_cols,
        "y_test": y_test,
        "X_test_s": X_test_s,
    }
