"""
IoT Cargo Wear Prediction & Status Classification
- Wear (mm) regression: ML (LR, RF, SVR, DT, KNN, XGBoost) + DL (ANN, GRU)
- Status classification: Logistic Regression, AUC, ROC, Confusion Matrix
- Metrics: MAE, RMSE, R², MAPE, Median AE, CV, p-value, factor importance
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, classification_report
)
from scipy import stats
from sklearn.base import clone
import xgboost as xgb

# Deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, GRU, Input, Reshape
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

os.makedirs("outputs", exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
if TF_AVAILABLE:
    tf.random.set_seed(RANDOM_STATE)

# ---------- Metrics ----------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def median_absolute_error_custom(y_true, y_pred):
    return np.median(np.abs(np.array(y_true) - np.array(y_pred)))

def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "Median_AE": median_absolute_error_custom(y_true, y_pred),
    }

# ---------- Load & Preprocess ----------
def load_data(path="data/iot_cargo_dataset.csv"):
    df = pd.read_csv(path)
    # Column names match schema (Speed_kmph etc. in CSV)
    return df

def preprocess_regression(df):
    df = df.copy()
    le_type = LabelEncoder()
    df["Type_enc"] = le_type.fit_transform(df["Type"])
    df["DeviceID_enc"] = LabelEncoder().fit_transform(df["DeviceID"].astype(str))
    feature_cols = ["Speed_kmph", "Pressure_psi", "Temperature_C", "Latitude", "Longitude",
                    "Obs_Obj", "Collision", "Type_enc", "DeviceID_enc"]
    X = df[feature_cols]
    y = df["Wear_mm"]
    return X, y, feature_cols

def preprocess_classification(df):
    df = df.copy()
    le_type = LabelEncoder()
    df["Type_enc"] = le_type.fit_transform(df["Type"])
    df["DeviceID_enc"] = LabelEncoder().fit_transform(df["DeviceID"].astype(str))
    feature_cols = ["Speed_kmph", "Pressure_psi", "Temperature_C", "Latitude", "Longitude",
                    "Obs_Obj", "Collision", "Type_enc", "DeviceID_enc"]
    X = df[feature_cols]
    y = df["Status"]
    return X, y, feature_cols

# ---------- Cross-validation ----------
def run_cv_regression(X, y, model, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    mae_scores, rmse_scores, r2_scores = [], [], []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        model_clone = clone_model(model)
        model_clone.fit(X_train_s, y_train)
        pred = model_clone.predict(X_val_s)
        mae_scores.append(mean_absolute_error(y_val, pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, pred)))
        r2_scores.append(r2_score(y_val, pred))
    return {"MAE_mean": np.mean(mae_scores), "MAE_std": np.std(mae_scores),
            "RMSE_mean": np.mean(rmse_scores), "RMSE_std": np.std(rmse_scores),
            "R2_mean": np.mean(r2_scores), "R2_std": np.std(r2_scores)}

def clone_model(estimator):
    return clone(estimator)

# ---------- ML Regression Models ----------
def get_ml_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=12),
        "KNN": KNeighborsRegressor(n_neighbors=15, weights="distance"),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_STATE),
        "SVR": SVR(kernel="rbf", C=10, gamma="scale"),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_STATE),
    }

# ---------- DL Models ----------
def build_ann(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def build_gru(input_dim, seq_len=1):
    # Use each sample as a trivial sequence of 1 step for compatibility with tabular data
    model = Sequential([
        Input(shape=(seq_len, input_dim)),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# ---------- Main ----------
def main():
    print("Loading data...")
    df = load_data()
    X_reg, y_reg, feature_cols = preprocess_regression(df)
    X_clf, y_clf, _ = preprocess_classification(df)

    scaler_reg = StandardScaler()
    X_reg_scaled = scaler_reg.fit_transform(X_reg)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg_scaled, y_reg, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler_clf = StandardScaler()
    X_clf_scaled = scaler_clf.fit_transform(X_clf)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf_scaled, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
    )

    # ----- Regression: ML models -----
    results_reg = []
    models_trained = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {name: {"MAE": [], "RMSE": [], "R2": []} for name in get_ml_models()}

    for name, model in get_ml_models().items():
        model.fit(X_train_r, y_train_r)
        pred = model.predict(X_test_r)
        metrics = regression_metrics(y_test_r, pred)
        metrics["Model"] = name
        metrics["Accuracy"] = max(0, 1 - metrics["RMSE"] / y_test_r.std())  # proxy
        results_reg.append(metrics)
        models_trained[name] = model
        # Cross-validation
        for train_idx, val_idx in kf.split(X_reg_scaled):
            m = clone_model(model)
            m.fit(X_reg_scaled[train_idx], y_reg.iloc[train_idx])
            p = m.predict(X_reg_scaled[val_idx])
            cv_results[name]["MAE"].append(mean_absolute_error(y_reg.iloc[val_idx], p))
            cv_results[name]["RMSE"].append(np.sqrt(mean_squared_error(y_reg.iloc[val_idx], p)))
            cv_results[name]["R2"].append(r2_score(y_reg.iloc[val_idx], p))

    # ----- Regression: DL (ANN, GRU) -----
    if TF_AVAILABLE:
        # ANN
        ann = build_ann(X_train_r.shape[1])
        ann.fit(X_train_r, y_train_r, epochs=80, batch_size=32, validation_split=0.15,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=0)
        pred_ann = ann.predict(X_test_r, verbose=0).ravel()
        results_reg.append({**regression_metrics(y_test_r, pred_ann), "Model": "ANN", "Accuracy": max(0, 1 - np.sqrt(mean_squared_error(y_test_r, pred_ann)) / y_test_r.std())})
        models_trained["ANN"] = ann
        # GRU (reshape to sequences)
        seq_len = 1
        X_train_gru = X_train_r.reshape(X_train_r.shape[0], seq_len, X_train_r.shape[1])
        X_test_gru = X_test_r.reshape(X_test_r.shape[0], seq_len, X_test_r.shape[1])
        gru = build_gru(X_train_r.shape[1], seq_len)
        gru.fit(X_train_gru, y_train_r, epochs=80, batch_size=32, validation_split=0.15,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=0)
        pred_gru = gru.predict(X_test_gru, verbose=0).ravel()
        results_reg.append({**regression_metrics(y_test_r, pred_gru), "Model": "GRU", "Accuracy": max(0, 1 - np.sqrt(mean_squared_error(y_test_r, pred_gru)) / y_test_r.std())})
        models_trained["GRU"] = gru
        for nm in ["ANN", "GRU"]:
            cv_results[nm] = {"MAE": [], "RMSE": [], "R2": []}
            for train_idx, val_idx in kf.split(X_reg_scaled):
                Xtr, Xva = X_reg_scaled[train_idx], X_reg_scaled[val_idx]
                ytr, yva = y_reg.iloc[train_idx], y_reg.iloc[val_idx]
                if nm == "GRU":
                    Xtr = Xtr.reshape(-1, seq_len, X_reg_scaled.shape[1])
                    Xva = Xva.reshape(-1, seq_len, X_reg_scaled.shape[1])
                    m = build_gru(X_reg_scaled.shape[1], seq_len)
                    m.fit(Xtr, ytr, epochs=30, batch_size=32, verbose=0)
                    p = m.predict(Xva, verbose=0).ravel()
                else:
                    m = build_ann(X_reg_scaled.shape[1])
                    m.fit(Xtr, ytr, epochs=30, batch_size=32, verbose=0)
                    p = m.predict(Xva, verbose=0).ravel()
                cv_results[nm]["MAE"].append(mean_absolute_error(yva, p))
                cv_results[nm]["RMSE"].append(np.sqrt(mean_squared_error(yva, p)))
                cv_results[nm]["R2"].append(r2_score(yva, p))

    # ----- Statistical comparison (p-value): pairwise paired t-tests -----
    preds_test = {}
    for name in models_trained:
        m = models_trained[name]
        if name in ["ANN"]:
            preds_test[name] = m.predict(X_test_r, verbose=0).ravel()
        elif name == "GRU":
            preds_test[name] = m.predict(X_test_r.reshape(X_test_r.shape[0], 1, X_test_r.shape[1]), verbose=0).ravel()
        else:
            preds_test[name] = m.predict(X_test_r)
    names = list(preds_test.keys())
    p_value_lines = ["Paired t-test on absolute errors (same test set):\n"]
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            e1 = np.abs(y_test_r.values - preds_test[n1])
            e2 = np.abs(y_test_r.values - preds_test[n2])
            t_stat, p_val = stats.ttest_rel(e1, e2)
            p_value_lines.append(f"  {n1} vs {n2}: t={t_stat:.4f}, p={p_val:.6f}\n")
    with open("outputs/p_value_comparison.txt", "w") as f:
        f.writelines(p_value_lines)

    # ----- Regression results table (Prediction accuracy = R2 for regression) -----
    df_results = pd.DataFrame(results_reg)
    df_results["Prediction_accuracy"] = df_results["R2"]  # R2 as primary accuracy metric for regression
    df_results = df_results[["Model", "Prediction_accuracy", "MAE", "RMSE", "R2", "MAPE", "Median_AE", "Accuracy"]]
    df_results.to_csv("outputs/wear_prediction_forecast_accuracy_table.csv", index=False)
    print("\n--- Wear Prediction Forecast Accuracy Table ---")
    print(df_results.to_string(index=False))

    # ----- Cross-validation summary -----
    cv_summary = []
    for name, v in cv_results.items():
        cv_summary.append({
            "Model": name,
            "CV_MAE_mean": np.mean(v["MAE"]), "CV_MAE_std": np.std(v["MAE"]),
            "CV_RMSE_mean": np.mean(v["RMSE"]), "CV_RMSE_std": np.std(v["RMSE"]),
            "CV_R2_mean": np.mean(v["R2"]), "CV_R2_std": np.std(v["R2"]),
        })
    pd.DataFrame(cv_summary).to_csv("outputs/cross_validation_summary.csv", index=False)

    # ----- Factor importance (RF & XGBoost) -----
    rf = models_trained["Random Forest"]
    imp_rf = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    xgb_model = models_trained["XGBoost"]
    imp_xgb = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    imp_rf.to_csv("outputs/factor_importance_RandomForest.csv")
    imp_xgb.to_csv("outputs/factor_importance_XGBoost.csv")

    # ----- Status classification: Logistic Regression -----
    le_status = LabelEncoder()
    y_train_c_enc = le_status.fit_transform(y_train_c)
    y_test_c_enc = le_status.transform(y_test_c)
    n_classes = len(le_status.classes_)
    lr_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, multi_class="multinomial")
    lr_clf.fit(X_train_c, y_train_c_enc)
    y_pred_c = lr_clf.predict(X_test_c)
    y_proba_c = lr_clf.predict_proba(X_test_c)
    acc = accuracy_score(y_test_c_enc, y_pred_c)
    cm = confusion_matrix(y_test_c_enc, y_pred_c)
    print("\n--- Status Classification (Logistic Regression) ---")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print(classification_report(y_test_c_enc, y_pred_c, target_names=le_status.classes_))
    # AUC-ROC (one-vs-rest)
    auc_ovr = roc_auc_score(y_test_c_enc, y_proba_c, multi_class="ovr", average="weighted")
    print("AUC-ROC (weighted ovr):", auc_ovr)
    np.savetxt("outputs/confusion_matrix_status.csv", cm, delimiter=",")

    # ----- Plots -----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # 1. Bar: MAE comparison
    ax = axes[0, 0]
    df_results.sort_values("MAE").plot(x="Model", y="MAE", kind="barh", ax=ax, legend=False)
    ax.set_title("Wear Prediction: MAE by Model")
    ax.set_xlabel("MAE")
    plt.setp(ax.get_xticklabels(), rotation=0)
    # 2. Bar: R2 comparison
    ax = axes[0, 1]
    df_results.sort_values("R2").plot(x="Model", y="R2", kind="barh", ax=ax, legend=False, color="green", alpha=0.7)
    ax.set_title("Wear Prediction: R² by Model")
    ax.set_xlabel("R²")
    plt.setp(ax.get_xticklabels(), rotation=0)
    # 3. Confusion matrix heatmap
    ax = axes[1, 0]
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, xticklabels=le_status.classes_, yticklabels=le_status.classes_)
    ax.set_title("Status Classification: Confusion Matrix")
    # 4. ROC curves (one-vs-rest for each class)
    ax = axes[1, 1]
    for i, cls in enumerate(le_status.classes_):
        bin_true = (y_test_c_enc == i).astype(int)
        bin_proba = y_proba_c[:, i]
        fpr, tpr, _ = roc_curve(bin_true, bin_proba)
        ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc_score(bin_true, bin_proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curves (Status)"); ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/comparative_analysis_metrics.png", dpi=150)
    plt.close()

    # KNN vs RF vs XGBoost comparison (separate scales for error vs R2)
    subset = df_results[df_results["Model"].isin(["KNN", "Random Forest", "XGBoost"])].reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(subset))
    width = 0.35
    ax1.bar(x - width/2, subset["MAE"], width, label="MAE", color="steelblue")
    ax1.bar(x + width/2, subset["RMSE"], width, label="RMSE", color="coral")
    ax1.set_xticks(x); ax1.set_xticklabels(subset["Model"])
    ax1.set_ylabel("Error (mm)"); ax1.set_title("MAE & RMSE")
    ax1.legend()
    ax2.bar(subset["Model"], subset["R2"], color="seagreen", alpha=0.8)
    ax2.set_ylabel("R²"); ax2.set_title("R² Score"); ax2.tick_params(axis="x", rotation=15)
    fig.suptitle("KNN vs Random Forest vs XGBoost (Wear Prediction)", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/knn_rf_xgboost_comparison.png", dpi=150)
    plt.close()

    # ML vs DL comparative (all models)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    order = df_results.sort_values("R2", ascending=False)["Model"].tolist()
    df_plot = df_results.set_index("Model").reindex(order)
    axes[0].barh(df_plot.index, df_plot["MAE"], color="steelblue", alpha=0.8)
    axes[0].set_xlabel("MAE (mm)"); axes[0].set_title("Wear Prediction: MAE Comparison (ML & DL)")
    axes[1].barh(df_plot.index, df_plot["R2"], color="seagreen", alpha=0.8)
    axes[1].set_xlabel("R²"); axes[1].set_title("Wear Prediction: R² Comparison (ML & DL)")
    plt.tight_layout()
    plt.savefig("outputs/ml_dl_comparative_graphical.png", dpi=150)
    plt.close()

    # Factor importance
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    imp_rf.sort_values().plot(kind="barh", ax=axes[0]); axes[0].set_title("Factors Influencing Wear (Random Forest)")
    imp_xgb.sort_values().plot(kind="barh", ax=axes[1]); axes[1].set_title("Factors Influencing Wear (XGBoost)")
    plt.tight_layout()
    plt.savefig("outputs/factor_importance_wear.png", dpi=150)
    plt.close()

    # Actual vs Predicted (best model by R2)
    best_name = df_results.loc[df_results["R2"].idxmax(), "Model"]
    if best_name in models_trained and best_name not in ["ANN", "GRU"]:
        pred_best = models_trained[best_name].predict(X_test_r)
    elif best_name == "ANN" and TF_AVAILABLE:
        pred_best = models_trained["ANN"].predict(X_test_r, verbose=0).ravel()
    elif best_name == "GRU" and TF_AVAILABLE:
        pred_best = models_trained["GRU"].predict(X_test_r.reshape(X_test_r.shape[0], 1, X_test_r.shape[1]), verbose=0).ravel()
    else:
        pred_best = models_trained["Random Forest"].predict(X_test_r)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_r, pred_best, alpha=0.5)
    plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], "r--")
    plt.xlabel("Actual Wear (mm)"); plt.ylabel("Predicted Wear (mm)")
    plt.title(f"Actual vs Predicted Wear ({best_name})")
    plt.savefig("outputs/actual_vs_predicted_wear.png", dpi=150)
    plt.close()

    print("\nAll outputs saved in outputs/")
    print("Done.")

if __name__ == "__main__":
    main()
