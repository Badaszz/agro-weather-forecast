import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# === CONFIG ===
FEATURES = [
    "temperature_2m_max", "temperature_2m_min",
    "windspeed_10m_max", "et0_fao_evapotranspiration",
    "precip_lag_1", "precip_lag_3", "precip_lag_7",
    "precip_roll7_mean", "precip_roll7_std",
    "precip_roll14_mean", "precip_roll14_std",
    "day_of_year", "month", "week",
    "is_dry_season", "is_peak_rain",
    "is_wet_season", "is_aug_break",
]
TARGET = "precip_next_day"

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("agro-rainfall-forecasting")


# === HELPERS ===
def load_splits(path: str = "data/features.csv"):
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    n = len(df)
    train = df.iloc[:int(n * 0.80)]
    val   = df.iloc[int(n * 0.80):int(n * 0.90)]
    test  = df.iloc[int(n * 0.90):]
    return train, val, test


def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def log_season_metrics(y_true, y_pred, index):
    df_eval = pd.DataFrame({
        "true": np.array(y_true),
        "pred": np.array(y_pred),
        "month": index.month
    })
    seasons = {
        "peak_rain":  df_eval["month"].isin([6, 9, 10]),
        "dry_season": df_eval["month"].isin([11, 12, 1, 2, 3]),
        "aug_break":  df_eval["month"] == 8,
    }
    for label, mask in seasons.items():
        subset = df_eval[mask]
        if len(subset) > 0:
            mae = mean_absolute_error(subset["true"], subset["pred"])
            mlflow.log_metric(f"MAE_{label}", round(mae, 4))


def save_scaler(scaler, path="data/scaler.pkl"):
    os.makedirs("data", exist_ok=True)
    joblib.dump(scaler, path)
    print(f"💾 Scaler saved to {path}")


# === MODEL 1: XGBOOST ===
def train_xgboost(train, val, test, scaler):
    print("\n🟢 Training XGBoost Regressor...")

    X_train = scaler.transform(train[FEATURES])
    X_val   = scaler.transform(val[FEATURES])
    X_test  = scaler.transform(test[FEATURES])
    y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]

    param_grid = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.01},
    ]

    best_val_mae = float("inf")
    best_params  = None

    for params in param_grid:
        with mlflow.start_run(run_name=f"xgboost-n{params['n_estimators']}-d{params['max_depth']}-lr{params['learning_rate']}"):
            model = XGBRegressor(**params, random_state=42, verbosity=0)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)

            val_preds  = model.predict(X_val)
            test_preds = model.predict(X_test)

            val_metrics  = compute_metrics(y_val, val_preds)
            test_metrics = compute_metrics(y_test, test_preds)

            mlflow.log_param("model", "XGBoost")
            mlflow.log_param("location", "Lagos, Nigeria")
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            log_season_metrics(y_test, test_preds, test.index)

            print(f"  XGB {params} → Val MAE: {val_metrics['MAE']} | R²: {val_metrics['R2']}")

            if val_metrics["MAE"] < best_val_mae:
                best_val_mae = val_metrics["MAE"]
                best_params  = params
                mlflow.sklearn.log_model(model, "xgb_model",
                                         registered_model_name="agro-rainfall-forecaster")

    print(f"\n🏆 Best XGBoost → {best_params} | Val MAE: {best_val_mae}")


# === MODEL 2: RANDOM FOREST ===
def train_random_forest(train, val, test, scaler):
    print("\n🟡 Training Random Forest Regressor...")

    X_train = scaler.transform(train[FEATURES])
    X_val   = scaler.transform(val[FEATURES])
    X_test  = scaler.transform(test[FEATURES])
    y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]

    param_grid = [
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 300, "max_depth": None},  # fully grown trees
    ]

    best_val_mae = float("inf")
    best_params  = None

    for params in param_grid:
        with mlflow.start_run(run_name=f"rf-n{params['n_estimators']}-d{params['max_depth']}"):
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            val_preds  = model.predict(X_val)
            test_preds = model.predict(X_test)

            val_metrics  = compute_metrics(y_val, val_preds)
            test_metrics = compute_metrics(y_test, test_preds)

            mlflow.log_param("model", "RandomForest")
            mlflow.log_param("location", "Lagos, Nigeria")
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            log_season_metrics(y_test, test_preds, test.index)

            print(f"  RF {params} → Val MAE: {val_metrics['MAE']} | R²: {val_metrics['R2']}")

            if val_metrics["MAE"] < best_val_mae:
                best_val_mae = val_metrics["MAE"]
                best_params  = params
                mlflow.sklearn.log_model(model, "rf_model",
                                         registered_model_name="agro-rainfall-forecaster")

    print(f"\n🏆 Best Random Forest → {best_params} | Val MAE: {best_val_mae}")


# === MODEL 3: RIDGE (BEST FROM TUNING) ===
def train_best_ridge(train, val, test, scaler):
    print("\n🔵 Training Best Ridge (α=100) for comparison...")

    X_train = scaler.transform(train[FEATURES])
    X_val   = scaler.transform(val[FEATURES])
    X_test  = scaler.transform(test[FEATURES])
    y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]

    with mlflow.start_run(run_name="ridge-best-alpha100"):
        model = Ridge(alpha=100.0)
        model.fit(X_train, y_train)

        val_preds  = model.predict(X_val)
        test_preds = model.predict(X_test)

        val_metrics  = compute_metrics(y_val, val_preds)
        test_metrics = compute_metrics(y_test, test_preds)

        mlflow.log_param("model", "Ridge")
        mlflow.log_param("alpha", 100.0)
        mlflow.log_param("location", "Lagos, Nigeria")
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        log_season_metrics(y_test, test_preds, test.index)

        mlflow.sklearn.log_model(model, "ridge_model",
                                  registered_model_name="agro-rainfall-forecaster")

        print(f"✅ Ridge α=100 → Val MAE: {val_metrics['MAE']} | R²: {val_metrics['R2']}")


# === MAIN ===
if __name__ == "__main__":
    train, val, test = load_splits()

    # Fit and save scaler
    scaler = StandardScaler()
    scaler.fit(train[FEATURES])
    save_scaler(scaler)

    train_xgboost(train, val, test, scaler)
    train_random_forest(train, val, test, scaler)
    train_best_ridge(train, val, test, scaler)

    print("\nAll runs logged! View at http://localhost:5000")