import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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
SEQUENCE_LEN = 14  # days of history for LSTM

# === CONFIG ===
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


def log_season_metrics(y_true, y_pred, index, run):
    """Log MAE broken down by season for deeper insight."""
    df_eval = pd.DataFrame({
        "true": y_true.values,
        "pred": y_pred,
        "month": index.month
    })
    peak_mask = df_eval["month"].isin([6, 9, 10])
    dry_mask  = df_eval["month"].isin([11, 12, 1, 2, 3])
    aug_mask  = df_eval["month"] == 8

    for label, mask in [("peak_rain", peak_mask), ("dry_season", dry_mask), ("aug_break", aug_mask)]:
        subset = df_eval[mask]
        if len(subset) > 0:
            mae = mean_absolute_error(subset["true"], subset["pred"])
            mlflow.log_metric(f"MAE_{label}", round(mae, 4))


# === MODEL 1: RIDGE REGRESSION ===
def train_ridge(train, val, test):
    print("\nTraining Baseline — Ridge Regression (Hyperparameter Tuning)...")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val,   y_val   = val[FEATURES],   val[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    alphas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0]
    best_val_mae = float("inf")
    best_scaler  = scaler
    best_alpha   = None

    for alpha in alphas:
        with mlflow.start_run(run_name=f"ridge-alpha-{alpha}"):
            model = Ridge(alpha=alpha)
            model.fit(X_train_sc, y_train)

            val_preds  = model.predict(X_val_sc)
            test_preds = model.predict(X_test_sc)

            val_metrics  = compute_metrics(y_val, val_preds)
            test_metrics = compute_metrics(y_test, test_preds)

            # Log params
            mlflow.log_param("model", "Ridge")
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("location", "Lagos, Nigeria")
            mlflow.log_param("features_count", len(FEATURES))

            # Log metrics
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            log_season_metrics(y_test, test_preds, test.index, mlflow)

            print(f"  α={alpha:<8} → Val MAE: {val_metrics['MAE']} | RMSE: {val_metrics['RMSE']} | R²: {val_metrics['R2']}")

            # Track best model
            if val_metrics["MAE"] < best_val_mae:
                best_val_mae = val_metrics["MAE"]
                best_alpha   = alpha
                best_model   = model
                mlflow.sklearn.log_model(model, "ridge_model",
                                         registered_model_name="agro-rainfall-forecaster")

    print(f"\nBest Ridge → α={best_alpha} | Val MAE: {best_val_mae}")
    return best_scaler


# === MODEL 2: LSTM ===
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()


def make_sequences(df, features, target, seq_len):
    X, y = [], []
    data = df[features].values
    tgt  = df[target].values
    for i in range(seq_len, len(df)):
        X.append(data[i - seq_len:i])
        y.append(tgt[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_lstm(train, val, test, scaler):
    print("\nTraining LSTM...")

    # Scale using same scaler from Ridge
    all_data = pd.concat([train, val, test])
    all_data[FEATURES] = scaler.transform(all_data[FEATURES])
    train_sc = all_data.iloc[:len(train)]
    val_sc   = all_data.iloc[len(train):len(train)+len(val)]
    test_sc  = all_data.iloc[len(train)+len(val):]

    X_train, y_train = make_sequences(train_sc, FEATURES, TARGET, SEQUENCE_LEN)
    X_val,   y_val   = make_sequences(val_sc,   FEATURES, TARGET, SEQUENCE_LEN)
    X_test,  y_test  = make_sequences(test_sc,  FEATURES, TARGET, SEQUENCE_LEN)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train),
                                             torch.tensor(y_train)),
                              batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMForecaster(input_size=len(FEATURES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    EPOCHS = 30

    with mlflow.start_run(run_name="lstm-v1"):
        mlflow.log_param("model", "LSTM")
        mlflow.log_param("hidden_size", 64)
        mlflow.log_param("num_layers", 2)
        mlflow.log_param("dropout", 0.2)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("sequence_len", SEQUENCE_LEN)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("location", "Lagos, Nigeria")

        for epoch in range(EPOCHS):
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validation loss
            model.eval()
            with torch.no_grad():
                xv = torch.tensor(X_val).to(device)
                yv = torch.tensor(y_val).to(device)
                val_loss = criterion(model(xv), yv).item()

            avg_train = np.mean(train_losses)
            mlflow.log_metric("train_loss", round(avg_train, 6), step=epoch)
            mlflow.log_metric("val_loss",   round(val_loss, 6),  step=epoch)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} → Train Loss: {avg_train:.4f} | Val Loss: {val_loss:.4f}")

        # Final evaluation on test set
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(X_test).to(device)
            test_preds = model(xt).cpu().numpy()

        test_metrics = compute_metrics(y_test, test_preds)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # Season-level metrics
        test_index = test_sc.index[SEQUENCE_LEN:]
        test_series = pd.Series(y_test)
        log_season_metrics(test_series, test_preds, test_index, mlflow)

        mlflow.pytorch.log_model(model, "lstm_model",
                                  registered_model_name="agro-rainfall-forecaster")

        print(f"LSTM → Test MAE: {test_metrics['MAE']} | RMSE: {test_metrics['RMSE']} | R²: {test_metrics['R2']}")


# === MAIN ===
if __name__ == "__main__":
    train, val, test = load_splits()
    scaler = train_ridge(train, val, test)
    train_lstm(train, val, test, scaler)
    print("\nAll runs complete! View results at http://localhost:5000")