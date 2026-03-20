import requests
import pandas as pd
from datetime import date, timedelta
import os

# === CONFIG ===
LOCATION = "Lagos, Nigeria"
LATITUDE = 6.5244
LONGITUDE = 3.3792
START_DATE = (date.today() - timedelta(days=730)).strftime("%Y-%m-%d")  # 2 years back
END_DATE = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")      # yesterday (target available)

VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "et0_fao_evapotranspiration"
]

def fetch_weather_data(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(VARIABLES),
        "timezone": "Africa/Lagos"
    }

    print(f"Fetching weather data for {LOCATION}...")
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code} - {response.text}")

    data = response.json()
    daily = data["daily"]

    df = pd.DataFrame(daily)
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    print(f"Fetched {len(df)} days of data ({START_DATE} → {END_DATE})")
    return df


def save_raw_data(df: pd.DataFrame, path: str = "data/raw_weather.csv"):
    os.makedirs("data", exist_ok=True)
    df.to_csv(path)
    print(f"Saved raw data to {path}")


if __name__ == "__main__":
    df = fetch_weather_data()
    save_raw_data(df)
    print(df.tail())