import requests
import pandas as pd
import numpy as np
import torch
import joblib

import matplotlib.pyplot as plt

from datetime import datetime, timezone
from pathlib import Path

from data import (
    LATITUDE,
    LONGITUDE,
    FEATURES,
    SEQUENCE_LENGTH,
    PREDICTION_HORIZON
)

from model import WeatherLSTM


# ==========================================
# Configuration
# ==========================================

API_URL = "https://api.open-meteo.com/v1/forecast"

TIMEZONE = "Europe/London"

WIND_SPEED_UNIT = "kmh"


# ==========================================
# Project Paths
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR / "models" / "edinburgh_lstm.pth"
)

SCALER_PATH = (
    PROJECT_DIR / "models" / "edinburgh_scaler.pkl"
)


# ==========================================
# Device Configuration
# ==========================================

def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    elif torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ==========================================
# Fetch Recent Weather Data
# ==========================================

def fetch_recent_weather():

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(FEATURES),
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": WIND_SPEED_UNIT,
        "precipitation_unit": "mm",
        "past_hours": 96,
        "forecast_hours": 12
    }

    print("Fetching recent weather data...")

    response = requests.get(
        API_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:

        raise ValueError(
            "Invalid API response: hourly data not found."
        )

    hourly = data["hourly"]

    df = pd.DataFrame({
        "time": pd.to_datetime(
            hourly["time"],
            utc=True
        )
    })

    for feature in FEATURES:

        if feature not in hourly:

            raise ValueError(
                f"Missing weather feature: {feature}"
            )

        df[feature] = hourly[feature]

    # Sort chronologically
    df = df.sort_values("time").reset_index(drop=True)

    # Keep only completed hourly timestamps
    current_hour = pd.Timestamp.now(
        tz="UTC"
    ).floor("h")

    df = df[
        df["time"] < current_hour
    ].copy()

    # Select the latest 72 hours
    df = df.tail(SEQUENCE_LENGTH)

    if len(df) != SEQUENCE_LENGTH:

        raise ValueError(
            f"Expected {SEQUENCE_LENGTH} hourly records, "
            f"received {len(df)}."
        )

    # Check hourly continuity
    if not (
        df["time"].diff().dropna()
        == pd.Timedelta(hours=1)
    ).all():

        raise ValueError(
            "Recent weather data is not hourly and continuous."
        )

    # Check missing and invalid values
    values = df[FEATURES].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(values).all():

        raise ValueError(
            "Recent weather data contains missing or invalid values."
        )

    print("Recent weather data fetched successfully.")

    print(f"Latest input timestamp: {df['time'].iloc[-1]}")

    return df


# ==========================================
# Load Trained Model and Scaler
# ==========================================

def load_model_and_scaler(device):

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Trained model not found. Run train.py first."
        )

    if not SCALER_PATH.exists():

        raise FileNotFoundError(
            "Scaler not found. Run evaluate.py first."
        )

    scaler = joblib.load(SCALER_PATH)

    model = WeatherLSTM().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    return model, scaler


# ==========================================
# Generate Temperature Predictions
# ==========================================

def plot_temperature_forecast(times, temperatures):

    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))

    time_labels = times.strftime("%H:%M")

    plt.plot(
        time_labels,
        temperatures,
        marker="o",
        linewidth=2
    )

    for i, temp in enumerate(temperatures):
        plt.annotate(
            f"{temp:.2f}°C",
            (i, temp),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center"
        )

    plt.title("Edinburgh Weather Forecast")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")

    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    chart_path = output_dir / "edinburgh_weather_forecast.png"

    plt.savefig(chart_path, dpi=300)

    print(f"\nForecast chart saved to: {chart_path}")

    plt.show()

def predict_temperature():

    device = get_device()

    print(f"\nUsing device: {device}")

    # Fetch recent weather data
    df = fetch_recent_weather()

    # Load model and scaler
    model, scaler = load_model_and_scaler(device)

    # Prepare input features
    input_data = df[FEATURES].to_numpy(
        dtype=np.float64
    )

    # Normalize using saved training scaler
    scaled_data = scaler.transform(
        input_data
    ).astype(np.float32)

    # Convert to PyTorch tensor
    X = torch.from_numpy(
        scaled_data
    ).unsqueeze(0).to(device)

    # Generate predictions
    with torch.no_grad():

        predictions = model(X)

    predictions = predictions.cpu().numpy().flatten()

    # Convert temperature back to Celsius
    temperature_mean = scaler.mean_[0]

    temperature_scale = scaler.scale_[0]

    predictions_celsius = (
        predictions * temperature_scale
        + temperature_mean
    )

    # Generate future timestamps
    last_timestamp = df["time"].iloc[-1]

    future_times = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=PREDICTION_HORIZON,
        freq="h"
    )

    # Convert UTC to Edinburgh local time
    local_times = future_times.tz_convert(
        TIMEZONE
    )

    # Create results table
    results = pd.DataFrame({

        "Date": local_times.strftime("%Y-%m-%d"),

        "Time": local_times.strftime("%H:%M"),

        "Predicted Temperature (°C)": np.round(
            predictions_celsius,
            2
        )

    })

    print("\n======================================")

    print("EDINBURGH WEATHER FORECAST")

    print("======================================")

    print(results.to_string(index=False))

    plot_temperature_forecast(
    local_times,
    predictions_celsius
    )

    print("\nAll predictions generated successfully.")


# ==========================================
# Run Prediction
# ==========================================

if __name__ == "__main__":

    predict_temperature()