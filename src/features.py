import pandas as pd
import numpy as np
import os

def build_features(input_path: str = "data/raw_weather.csv",
                   output_path: str = "data/features.csv") -> pd.DataFrame:

    df = pd.read_csv(input_path, index_col="date", parse_dates=True)
    print(f"📂 Loaded raw data: {df.shape}")

    # === 1. Handle Missing Values ===
    df = df.interpolate(method="time").ffill().bfill()
    print("✅ Missing values handled")

    # === 2. Lag Features (previous days' rainfall) ===
    df["precip_lag_1"] = df["precipitation_sum"].shift(1)
    df["precip_lag_3"] = df["precipitation_sum"].shift(3)
    df["precip_lag_7"] = df["precipitation_sum"].shift(7)

    # === 3. Rolling Statistics ===
    df["precip_roll7_mean"]  = df["precipitation_sum"].rolling(7).mean()
    df["precip_roll7_std"]   = df["precipitation_sum"].rolling(7).std()
    df["precip_roll14_mean"] = df["precipitation_sum"].rolling(14).mean()
    df["precip_roll14_std"]  = df["precipitation_sum"].rolling(14).std()

    # === 4. Calendar Features ===
    df["day_of_year"] = df.index.dayofyear
    df["month"]       = df.index.month
    df["week"]        = df.index.isocalendar().week.astype(int)

    # Season flags basaed on EDA
    df["is_peak_rain"]    = df["month"].isin([6, 9, 10]).astype(int)  # add October
    df["is_wet_season"]   = df["month"].isin([4, 5, 6, 7, 9, 10]).astype(int)
    df["is_aug_break"]    = df["month"].isin([8]).astype(int)         # August Break specifically
    df["is_dry_season"]   = df["month"].isin([11, 12, 1, 2, 3]).astype(int)
    
    # Transition months — rainfall is rising or falling sharply
    df["is_onset_month"] = df["month"].isin([4, 9]).astype(int)   # rains beginning
    df["is_retreat_month"] = df["month"].isin([7, 11]).astype(int) # rains ending

    # === 5. Target Column (next day's rainfall) ===
    df["precip_next_day"] = df["precipitation_sum"].shift(-1)

    # === 6. Drop rows with NaN (from lags + last row with no target) ===
    df.dropna(inplace=True)
    print(f"✅ Features engineered: {df.shape}")

    # === 7. Chronological Train / Val / Test Split ===
    n = len(df)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    print(f"📊 Split → Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # === 8. Save ===
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path)
    print(f"💾 Saved features to {output_path}")

    return train, val, test


if __name__ == "__main__":
    train, val, test = build_features()
    print("\nSample feature row:")
    print(train.tail(2).T)