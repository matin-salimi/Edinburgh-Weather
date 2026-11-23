
import requests
import pandas as pd
import numpy as np

from datetime import datetime, timedelta, timezone
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# ==========================================
# Edinburgh Weather Configuration
# ==========================================

LATITUDE = 55.9533
LONGITUDE = -3.1883

TIMEZONE = "UTC"

API_URL = (
    "https://historical-forecast-api.open-meteo.com/v1/forecast"
)

FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",
    "cloud_cover"
]

TARGET_INDEX = 0

SEQUENCE_LENGTH = 72
PREDICTION_HORIZON = 12


# ==========================================
# Fetch Historical Weather Data
# ==========================================

def fetch_historical_weather():

    today = datetime.now(timezone.utc).date()

    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=730)

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(FEATURES),
        "timezone": TIMEZONE,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm"
    }

    print("Downloading historical weather data...")

    response = requests.get(
        API_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError("Invalid API response: hourly data not found.")

    hourly_data = data["hourly"]

    df = pd.DataFrame({
        "time": hourly_data["time"]
    })

    for feature in FEATURES:

        if feature not in hourly_data:
            raise ValueError(
                f"Missing weather feature: {feature}"
            )

        df[feature] = hourly_data[feature]

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    df = df.sort_values("time").reset_index(drop=True)

    df = df.drop_duplicates(
        subset="time",
        keep="last"
    )

    project_dir = Path(__file__).resolve().parent

    dataset_path = (
        project_dir / "dataset" / "edinburgh_weather.csv"
    )

    df.to_csv(
        dataset_path,
        index=False
    )

    print("\nHistorical data downloaded successfully!")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    print(f"Total records: {len(df)}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    print(f"\nDataset saved to: {dataset_path}")

    return df


# ==========================================
# Create Sliding Window Sequences
# ==========================================

def create_sequences(
    scaled_data,
    target_start,
    target_end
):

    X = []
    y = []

    first_target = max(
        SEQUENCE_LENGTH,
        target_start
    )

    last_target = target_end - PREDICTION_HORIZON

    for i in range(first_target, last_target + 1):

        input_sequence = scaled_data[
            i - SEQUENCE_LENGTH:i
        ]

        target_sequence = scaled_data[
            i:i + PREDICTION_HORIZON,
            TARGET_INDEX
        ]

        X.append(input_sequence)
        y.append(target_sequence)

    X = np.array(X, dtype=np.float32)

    y = np.array(y, dtype=np.float32)

    return X, y


# ==========================================
# Prepare Training, Validation and Test Data
# ==========================================

def prepare_datasets():

    project_dir = Path(__file__).resolve().parent

    dataset_path = (
        project_dir / "dataset" / "edinburgh_weather.csv"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Historical dataset not found. "
            "Run data.py to download the data first."
        )

    df = pd.read_csv(
        dataset_path,
        parse_dates=["time"]
    )

    # Validate dataset
    if df[FEATURES].isnull().any().any():
        raise ValueError(
            "Dataset contains missing weather values."
        )

    if not np.isfinite(
        df[FEATURES].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Dataset contains infinite values."
        )

    if df["time"].duplicated().any():
        raise ValueError(
            "Dataset contains duplicate timestamps."
        )

    if not (
        df["time"].diff().dropna()
        == pd.Timedelta(hours=1)
    ).all():
        raise ValueError(
            "Dataset contains non-hourly intervals."
        )

    data = df[FEATURES].to_numpy(
        dtype=np.float64
    )

    total_records = len(data)

    # Chronological split
    train_end = int(total_records * 0.70)

    val_end = int(total_records * 0.85)

    # Fit scaler ONLY on training data
    scaler = StandardScaler()

    scaler.fit(data[:train_end])

    # Transform all data using training scaler
    scaled_data = scaler.transform(data).astype(
        np.float32
    )

    # Create sequences
    X_train, y_train = create_sequences(
        scaled_data,
        SEQUENCE_LENGTH,
        train_end
    )

    X_val, y_val = create_sequences(
        scaled_data,
        train_end,
        val_end
    )

    X_test, y_test = create_sequences(
        scaled_data,
        val_end,
        total_records
    )

    print("\nDataset preparation completed!")

    print("\nTraining:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("\nValidation:")
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    print("\nTesting:")
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("\nScaler fitted using training data only.")

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaler
    )


# ==========================================
# Run Data Collection
# ==========================================

if __name__ == "__main__":

    fetch_historical_weather()
