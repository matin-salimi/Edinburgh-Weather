"""Data collection (ERA5 via Open-Meteo) and dataset preparation."""

import time
from datetime import date

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

from features import build_features
from config import (
    ARCHIVE_API_URL,
    DATA_TIMEZONE,
    DATASET_DIR,
    DATASET_PATH,
    END_DATE,
    EXPERIMENT,
    FEATURES,
    LATITUDE,
    LONGITUDE,
    PREDICTION_HORIZON,
    SEQUENCE_LENGTH,
    START_DATE,
    TARGET_INDEX,
    TRAIN_END,
    VAL_END,
)


# ==========================================
# Fetch Historical Weather Data (ERA5)
# ==========================================

def _year_ranges(start: date, end: date):
    """Split [start, end] into calendar-year chunks (keeps API calls small)."""
    for year in range(start.year, end.year + 1):
        yield max(date(year, 1, 1), start), min(date(year, 12, 31), end)


def _download_chunk(start: date, end: date, retries: int = 3) -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(FEATURES),
        "timezone": DATA_TIMEZONE,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                ARCHIVE_API_URL, params=params, timeout=120
            )
            response.raise_for_status()
            break
        except requests.RequestException as error:
            if attempt == retries:
                raise
            wait = 5 * attempt
            print(f"  Request failed ({error}). Retrying in {wait}s...")
            time.sleep(wait)

    hourly = response.json().get("hourly")

    if hourly is None:
        raise ValueError("Invalid API response: hourly data not found.")

    df = pd.DataFrame({"time": hourly["time"]})

    for feature in FEATURES:
        if feature not in hourly:
            raise ValueError(f"Missing weather feature: {feature}")
        df[feature] = hourly[feature]

    return df


def fetch_historical_weather() -> pd.DataFrame:
    start = date.fromisoformat(START_DATE)
    end = date.fromisoformat(END_DATE)

    print(f"Downloading ERA5 data for Edinburgh: {start} -> {end}")

    chunks = []

    for chunk_start, chunk_end in _year_ranges(start, end):
        print(f"  {chunk_start} -> {chunk_end}")
        chunks.append(_download_chunk(chunk_start, chunk_end))
        time.sleep(1)  # be polite to the free API

    df = pd.concat(chunks, ignore_index=True)

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = (
        df.drop_duplicates(subset="time", keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)

    print("\nHistorical data downloaded successfully!")
    print(f"Total records: {len(df)}")
    print(f"Missing values: {df[FEATURES].isnull().sum().sum()}")
    print(f"Dataset saved to: {DATASET_PATH}")

    return df


# ==========================================
# Load and validate dataset
# ==========================================

def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Historical dataset not found. Run `python3 data.py` first."
        )

    df = pd.read_csv(DATASET_PATH)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    if df[FEATURES].isnull().any().any():
        raise ValueError("Dataset contains missing weather values.")

    if not np.isfinite(df[FEATURES].to_numpy(dtype=np.float64)).all():
        raise ValueError("Dataset contains infinite values.")

    if df["time"].duplicated().any():
        raise ValueError("Dataset contains duplicate timestamps.")

    if not (df["time"].diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("Dataset contains non-hourly intervals.")

    return df


# ==========================================
# Create Sliding Window Sequences
# ==========================================

def create_sequences(scaled_data, target_start, target_end):
    """Build (input window, target window) pairs.

    `target_start` / `target_end` are row indices of the split. All TARGET
    values lie inside [target_start, target_end); input windows may look back
    into the previous split, which is legitimate because in real use the
    past is always known.

    Returns X, y and the row index of the first target hour of each sample.
    """
    first_target = max(SEQUENCE_LENGTH, target_start)
    last_target = target_end - PREDICTION_HORIZON

    X, y, indices = [], [], []

    for i in range(first_target, last_target + 1):
        X.append(scaled_data[i - SEQUENCE_LENGTH:i])
        y.append(scaled_data[i:i + PREDICTION_HORIZON, TARGET_INDEX])
        indices.append(i)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(indices),
    )


# ==========================================
# Prepare Training, Validation and Test Data
# ==========================================

def prepare_datasets() -> dict:
    df = load_dataset()

    times = df["time"]
    data = build_features(df)
    total_records = len(data)

    # Year-based chronological split
    train_end = int((times < pd.Timestamp(TRAIN_END, tz="UTC")).sum())
    val_end = int((times < pd.Timestamp(VAL_END, tz="UTC")).sum())

    # Fit scaler ONLY on training data (no leakage)
    scaler = StandardScaler()
    scaler.fit(data[:train_end])

    scaled_data = scaler.transform(data).astype(np.float32)

    X_train, y_train, _ = create_sequences(
        scaled_data, SEQUENCE_LENGTH, train_end
    )
    X_val, y_val, _ = create_sequences(scaled_data, train_end, val_end)
    X_test, y_test, idx_test = create_sequences(
        scaled_data, val_end, total_records
    )

    print("\nDataset preparation completed!")
    print(f"Experiment: {EXPERIMENT}   Input features: {data.shape[1]}")
    print(f"Period: {times.iloc[0].date()} -> {times.iloc[-1].date()}")
    print(f"Train: {len(X_train):>6} samples  (until {TRAIN_END})")
    print(f"Val:   {len(X_val):>6} samples  ({TRAIN_END} -> {VAL_END})")
    print(f"Test:  {len(X_test):>6} samples  (from {VAL_END})")
    print("Scaler fitted using training data only.")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "test_times": times.iloc[idx_test].reset_index(drop=True),
        "scaler": scaler,
    }


# ==========================================
# Run Data Collection
# ==========================================

if __name__ == "__main__":
    fetch_historical_weather()