import pandas as pd
import numpy as np
import requests
import joblib
from datetime import date, timedelta
import time
from src.database import (init_db, save_prediction, save_actual,
                           get_predictions_missing_actuals)
LATITUDE = 6.5244
LONGITUDE = 3.3792

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

def fetch_recent_weather(latitude: float, longitude: float, days: int = 21) -> pd.DataFrame:
    """Fetch last `days` days of weather from Open-Meteo."""
    end_date   = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "windspeed_10m_max",
            "et0_fao_evapotranspiration"
        ]),
        "timezone": "Africa/Lagos"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Weather API error: {response.status_code}")

    daily = response.json()["daily"]
    df = pd.DataFrame(daily)
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def build_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw weather data."""
    df = df.interpolate(method="time").ffill().bfill()

    df["precip_lag_1"] = df["precipitation_sum"].shift(1)
    df["precip_lag_3"] = df["precipitation_sum"].shift(3)
    df["precip_lag_7"] = df["precipitation_sum"].shift(7)

    df["precip_roll7_mean"]  = df["precipitation_sum"].rolling(7).mean()
    df["precip_roll7_std"]   = df["precipitation_sum"].rolling(7).std()
    df["precip_roll14_mean"] = df["precipitation_sum"].rolling(14).mean()
    df["precip_roll14_std"]  = df["precipitation_sum"].rolling(14).std()

    df["day_of_year"]   = df.index.dayofyear
    df["month"]         = df.index.month
    df["week"]          = df.index.isocalendar().week.astype(int)
    df["is_dry_season"] = df["month"].isin([11, 12, 1, 2, 3]).astype(int)
    df["is_peak_rain"]  = df["month"].isin([6, 9, 10]).astype(int)
    df["is_wet_season"] = df["month"].isin([4, 5, 6, 7, 9, 10]).astype(int)
    df["is_aug_break"]  = (df["month"] == 8).astype(int)

    df.dropna(inplace=True)
    return df


def predict_next_day_rainfall(latitude: float, longitude: float) -> dict:
    """Full inference pipeline — fetch, engineer, predict and save to DB."""
    model  = joblib.load("data/best_model.pkl")
    scaler = joblib.load("data/scaler.pkl")

    raw_df      = fetch_recent_weather(latitude, longitude)
    features_df = build_prediction_features(raw_df)

    latest_row = features_df[FEATURES].iloc[[-1]]
    scaled     = scaler.transform(latest_row)
    prediction = model.predict(scaled)[0]

    prediction_date = date.today().strftime("%Y-%m-%d")
    based_on        = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    predicted_mm    = round(float(max(prediction, 0)), 2)

    init_db()
    save_prediction(
        prediction_date=prediction_date,
        predicted_rainfall_mm=predicted_mm,
        based_on_data_up_to=based_on,
    )

    return {
        "predicted_rainfall_mm": predicted_mm,
        "prediction_date":       prediction_date,
        "based_on_data_up_to":   based_on,
        "model":                 "Ridge (α=100)",
    }
    
def predict_for_date(target_date: str, latitude: float, longitude: float) -> dict:
    """Predict rainfall for a specific date using weather data leading up to it."""
    model  = joblib.load("data/best_model.pkl")
    scaler = joblib.load("data/scaler.pkl")

    target = pd.to_datetime(target_date)
    fetch_end   = (target - timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_start = (target - timedelta(days=21)).strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": fetch_start,
        "end_date": fetch_end,
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "windspeed_10m_max",
            "et0_fao_evapotranspiration"
        ]),
        "timezone": "Africa/Lagos"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Weather API error: {response.status_code}")

    daily = response.json()["daily"]
    df = pd.DataFrame(daily)
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    features_df = build_prediction_features(df)

    if features_df.empty:
        raise Exception(f"Not enough data to build features for {target_date}")

    latest_row = features_df[FEATURES].iloc[[-1]]
    scaled     = scaler.transform(latest_row)
    prediction = model.predict(scaled)[0]

    return {
        "predicted_rainfall_mm": round(float(max(prediction, 0)), 2),
        "prediction_date": target_date,
        "based_on_data_up_to": fetch_end,
        "model": "Ridge (α=100)",
    }

def fetch_actual_for_date(
    target_date: str,
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    retries: int = 3,
    backoff: float = 5.0,
) -> float | None:
    """
    Fetch the real observed rainfall for a specific past date.
    Retries up to `retries` times with exponential backoff on failure.
    Returns the rainfall value or None if all attempts fail.
    """
    for attempt in range(1, retries + 1):
        try:
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": target_date,
                "end_date": target_date,
                "daily": "precipitation_sum",
                "timezone": "Africa/Lagos",
            }
            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                raise Exception(f"API returned {response.status_code}")

            value = response.json()["daily"]["precipitation_sum"][0]
            if value is None:
                raise Exception("API returned null rainfall value")

            print(f"✅ Fetched actual for {target_date}: {value}mm")
            return round(float(value), 2)

        except Exception as e:
            wait = backoff * (2 ** (attempt - 1))  # 5s, 10s, 20s
            print(f"⚠️  Attempt {attempt}/{retries} failed for {target_date}: {e}")
            if attempt < retries:
                print(f"   Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"❌ All {retries} attempts failed for {target_date}. Will retry next run.")
    return None


def backfill_missing_actuals(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
):
    """
    Check all prediction dates that have no actual yet and try to fill them in.
    Safe to call on every daily run — skips dates that already have actuals.
    """
    missing = get_predictions_missing_actuals()

    if not missing:
        print("No missing actuals to backfill.")
        return

    print(f"Backfilling {len(missing)} missing actual(s): {missing}")
    filled, failed = 0, 0

    for target_date in missing:
        value = fetch_actual_for_date(target_date, latitude, longitude)
        if value is not None:
            save_actual(target_date, value)
            filled += 1
        else:
            failed += 1

    print(f"\nBackfill complete → Filled: {filled} | Still missing: {failed}")
    
def run_daily_pipeline():
    print("Running daily prediction pipeline...")
    init_db()

    # 1. Make today's prediction and save it
    print("\nGenerating today's prediction...")
    result = predict_next_day_rainfall(LATITUDE, LONGITUDE)
    print(f"Predicted {result['predicted_rainfall_mm']}mm for {result['prediction_date']}")

    # 2. Fetch actual for yesterday (that's when yesterday's prediction can be verified)
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\nFetching actual rainfall for {yesterday}...")
    actual_val = fetch_actual_for_date(yesterday, LATITUDE, LONGITUDE)
    if actual_val is not None:
        save_actual(yesterday, actual_val)

    # 3. Backfill any other dates still missing actuals
    print("\nChecking for other missing actuals...")
    backfill_missing_actuals(LATITUDE, LONGITUDE)

    print("\nDaily pipeline complete.")
    return result


if __name__ == "__main__":
    run_daily_pipeline()