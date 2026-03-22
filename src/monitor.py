import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, date, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.database import (
    get_predictions_with_actuals,
    save_monitoring_result,
    get_monitoring_results,
)

MONITOR_LOG_PATH = "data/monitor_log.json"

# === DRIFT CONFIG ===
# Baseline MAE from your best Ridge model test performance
BASELINE_MAE        = 1.8536
MAE_DRIFT_THRESHOLD = 0.30   # flag drift if rolling MAE exceeds baseline by 30%
MIN_SAMPLES         = 7      # need at least 7 paired samples to evaluate


def compute_metrics(y_true: list, y_pred: list) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae":  round(float(mae),  4),
        "rmse": round(float(rmse), 4),
    }


def detect_drift(current_mae: float) -> dict:
    """
    Compare current rolling MAE against the baseline.
    Flag drift if MAE has degraded beyond the threshold.
    """
    degradation    = (current_mae - BASELINE_MAE) / BASELINE_MAE
    drift_detected = degradation > MAE_DRIFT_THRESHOLD

    return {
        "drift_detected":    bool(drift_detected),
        "baseline_mae":      BASELINE_MAE,
        "current_mae":       round(current_mae, 4),
        "degradation_pct":   round(float(degradation * 100), 2),
        "threshold_pct":     round(MAE_DRIFT_THRESHOLD * 100, 2),
    }


def run_monitoring(window_days: int = 30) -> dict | None:
    """
    Core monitoring function.
    Fetches all predictions that now have actuals, computes rolling
    performance metrics, detects drift, and saves results to DB.
    Returns the monitoring result dict or None if not enough data.
    """
    print(f"📊 Running model monitoring (window: {window_days} days)...")

    # Fetch paired predictions + actuals from DB
    pairs = get_predictions_with_actuals(days=window_days)

    if len(pairs) < MIN_SAMPLES:
        print(f"⚠️  Only {len(pairs)} paired samples available."
              f" Need at least {MIN_SAMPLES} to evaluate. Skipping.")
        return None

    df = pd.DataFrame(pairs)
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    df = df.sort_values("prediction_date")

    y_true = df["actual_rainfall"].tolist()
    y_pred = df["predicted_rainfall_mm"].tolist()

    # === Overall window metrics ===
    metrics = compute_metrics(y_true, y_pred)

    # === Drift detection ===
    drift = detect_drift(metrics["mae"])

    # === Rolling 7-day MAE trend ===
    df["error"] = (df["actual_rainfall"] - df["predicted_rainfall_mm"]).abs()
    df["rolling_mae_7d"] = df["error"].rolling(7, min_periods=3).mean()

    # === Per-season breakdown ===
    df["month"]          = df["prediction_date"].dt.month
    season_metrics       = {}
    season_map = {
        "peak_rain":  [6, 9, 10],
        "wet_season": [4, 5, 7],
        "aug_break":  [8],
        "dry_season": [11, 12, 1, 2, 3],
    }
    for season, months in season_map.items():
        subset = df[df["month"].isin(months)]
        if len(subset) >= 3:
            sm = compute_metrics(
                subset["actual_rainfall"].tolist(),
                subset["predicted_rainfall_mm"].tolist()
            )
            season_metrics[season] = sm
        else:
            season_metrics[season] = None

    # === Save to DB ===
    evaluated_on = date.today().strftime("%Y-%m-%d")
    save_monitoring_result(
        evaluated_on=evaluated_on,
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        num_samples=len(pairs),
        drift_detected=drift["drift_detected"],
    )

    result = {
        "evaluated_on":    evaluated_on,
        "num_samples":     len(pairs),
        "window_days":     window_days,
        "metrics":         metrics,
        "drift":           drift,
        "season_metrics":  season_metrics,
        "rolling_mae_7d":  df[["prediction_date", "rolling_mae_7d"]]
                             .dropna()
                             .assign(prediction_date=lambda x:
                                 x["prediction_date"].dt.strftime("%Y-%m-%d"))
                             .to_dict(orient="records"),
    }

    save_monitor_log(result)
    print_monitoring_report(result)
    return result


def print_monitoring_report(result: dict):
    drift   = result["drift"]
    metrics = result["metrics"]
    icon    = "🚨" if drift["drift_detected"] else "✅"

    print(f"\n{'='*56}")
    print(f"  📊 MODEL MONITORING REPORT — {result['evaluated_on']}")
    print(f"{'='*56}")
    print(f"  Samples evaluated : {result['num_samples']} (last {result['window_days']} days)")
    print(f"  MAE               : {metrics['mae']} mm")
    print(f"  RMSE              : {metrics['rmse']} mm")
    print(f"  Baseline MAE      : {drift['baseline_mae']} mm")
    print(f"  Degradation       : {drift['degradation_pct']:+.1f}%")
    print(f"  Drift threshold   : {drift['threshold_pct']}%")
    print(f"  {icon} Drift detected  : {drift['drift_detected']}")
    print(f"{'-'*56}")
    print("  Season breakdown:")
    for season, sm in result["season_metrics"].items():
        if sm:
            print(f"    {season:<15} MAE: {sm['mae']} | RMSE: {sm['rmse']}")
        else:
            print(f"    {season:<15} Not enough data yet")
    print(f"{'='*56}\n")

    if drift["drift_detected"]:
        print("🚨 DRIFT ALERT: Model performance has degraded significantly.")
        print("   Consider triggering an early retraining.")
        print("   Check the Streamlit dashboard for details.\n")


def save_monitor_log(result: dict, path: str = MONITOR_LOG_PATH):
    """Append monitoring result to a local JSON log for Streamlit to read."""
    os.makedirs("data", exist_ok=True)
    history = []
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []

    history.append(result)
    history = history[-365:]  # keep last year of daily results
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"💾 Monitor log saved to {path}")


def get_latest_monitoring_result() -> dict | None:
    """Return the most recent monitoring result from the log."""
    if not os.path.exists(MONITOR_LOG_PATH):
        return None
    with open(MONITOR_LOG_PATH, "r") as f:
        try:
            history = json.load(f)
            return history[-1] if history else None
        except json.JSONDecodeError:
            return None


def get_drift_history(days: int = 60) -> list:
    """Return monitoring results for the Streamlit drift chart."""
    results = get_monitoring_results(days=days)
    return results


if __name__ == "__main__":
    result = run_monitoring(window_days=30)
    if result is None:
        print("Not enough data yet — run daily predictions for at least"
              f" {MIN_SAMPLES} days first.")