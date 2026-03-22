import sqlite3
import os
from datetime import date, timedelta
from contextlib import contextmanager

DB_PATH = "data/agro_forecast.db"


@contextmanager
def get_connection():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date       TEXT NOT NULL UNIQUE,
                predicted_rainfall_mm REAL NOT NULL,
                based_on_data_up_to   TEXT NOT NULL,
                model_version         TEXT NOT NULL,
                created_at            TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS actuals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                actual_date     TEXT NOT NULL UNIQUE,
                actual_rainfall REAL NOT NULL,
                fetched_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS monitoring_results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluated_on    TEXT NOT NULL UNIQUE,
                mae             REAL,
                rmse            REAL,
                num_samples     INTEGER,
                drift_detected  INTEGER DEFAULT 0,
                created_at      TEXT DEFAULT (datetime('now'))
            );
        """)
    print("✅ Database initialised at", DB_PATH)


# === PREDICTIONS ===
def save_prediction(prediction_date: str,
                    predicted_rainfall_mm: float,
                    based_on_data_up_to: str,
                    model_version: str = "Ridge (α=100)"):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO predictions
                (prediction_date, predicted_rainfall_mm, based_on_data_up_to, model_version)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(prediction_date) DO UPDATE SET
                predicted_rainfall_mm = excluded.predicted_rainfall_mm,
                based_on_data_up_to   = excluded.based_on_data_up_to,
                model_version         = excluded.model_version,
                created_at            = datetime('now')
        """, (prediction_date, predicted_rainfall_mm, based_on_data_up_to, model_version))
    print(f"💾 Prediction saved → {prediction_date}: {predicted_rainfall_mm}mm")


def get_predictions(days: int = 30) -> list:
    """Fetch the last N days of predictions."""
    since = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM predictions
            WHERE prediction_date >= ?
            ORDER BY prediction_date ASC
        """, (since,)).fetchall()
    return [dict(r) for r in rows]


def get_prediction_for_date(target_date: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("""
            SELECT * FROM predictions WHERE prediction_date = ?
        """, (target_date,)).fetchone()
    return dict(row) if row else None


# === ACTUALS ===
def save_actual(actual_date: str, actual_rainfall: float):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO actuals (actual_date, actual_rainfall)
            VALUES (?, ?)
            ON CONFLICT(actual_date) DO UPDATE SET
                actual_rainfall = excluded.actual_rainfall,
                fetched_at      = datetime('now')
        """, (actual_date, actual_rainfall))
    print(f"💾 Actual saved → {actual_date}: {actual_rainfall}mm")


def get_actuals(days: int = 30) -> list:
    since = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM actuals
            WHERE actual_date >= ?
            ORDER BY actual_date ASC
        """, (since,)).fetchall()
    return [dict(r) for r in rows]


# === MONITORING ===
def save_monitoring_result(evaluated_on: str,
                            mae: float,
                            rmse: float,
                            num_samples: int,
                            drift_detected: bool):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO monitoring_results
                (evaluated_on, mae, rmse, num_samples, drift_detected)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(evaluated_on) DO UPDATE SET
                mae            = excluded.mae,
                rmse           = excluded.rmse,
                num_samples    = excluded.num_samples,
                drift_detected = excluded.drift_detected,
                created_at     = datetime('now')
        """, (evaluated_on, mae, rmse, num_samples, int(drift_detected)))
    print(f"💾 Monitoring result saved → {evaluated_on} | MAE: {mae} | Drift: {drift_detected}")


def get_monitoring_results(days: int = 60) -> list:
    since = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM monitoring_results
            WHERE evaluated_on >= ?
            ORDER BY evaluated_on ASC
        """, (since,)).fetchall()
    return [dict(r) for r in rows]


# === JOINED VIEW (for NannyML + Streamlit) ===
def get_predictions_with_actuals(days: int = 60) -> list:
    """Returns rows where both a prediction and actual exist — used for evaluation."""
    since = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                p.prediction_date,
                p.predicted_rainfall_mm,
                p.model_version,
                a.actual_rainfall
            FROM predictions p
            JOIN actuals a ON p.prediction_date = a.actual_date
            WHERE p.prediction_date >= ?
            ORDER BY p.prediction_date ASC
        """, (since,)).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()