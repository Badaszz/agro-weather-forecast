import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
BEST_ALPHA = 100.0


def load_data(path: str = "data/features.csv"):
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    n = len(df)
    train_val = df.iloc[:int(n * 0.90)]  # train + val combined
    test      = df.iloc[int(n * 0.90):]  # held-out test stays untouched
    return train_val, test


def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


if __name__ == "__main__":
    print("Loading data...")
    train_val, test = load_data()
    print(f"   Training on {len(train_val)} rows (train + val)")
    print(f"   Evaluating on {len(test)} rows (held-out test)")

    X_train = train_val[FEATURES]
    y_train = train_val[TARGET]
    X_test  = test[FEATURES]
    y_test  = test[TARGET]

    # Fit scaler on train+val
    print("\nFitting scaler...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train final model
    print(f"Training final Ridge model (α={BEST_ALPHA})...")
    model = Ridge(alpha=BEST_ALPHA)
    model.fit(X_train_sc, y_train)

    # Evaluate on held-out test set
    test_preds  = model.predict(X_test_sc)
    metrics     = compute_metrics(y_test, test_preds)
    print(f"\nFinal Model Test Performance:")
    print(f"   MAE  : {metrics['MAE']}")
    print(f"   RMSE : {metrics['RMSE']}")
    print(f"   R²   : {metrics['R2']}")

    # Save model and scaler
    os.makedirs("data", exist_ok=True)
    joblib.dump(model,  "data/best_model.pkl")
    joblib.dump(scaler, "data/scaler.pkl")
    print("\nSaved:")
    print("   → data/best_model.pkl")
    print("   → data/scaler.pkl")
    print("\nFinal model ready for deployment!")