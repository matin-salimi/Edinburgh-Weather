import argparse
from zoneinfo import ZoneInfo

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch

from config import (
    CHART_HISTORY_HOURS,
    DISPLAY_TIMEZONE,
    FEATURES,
    FORECAST_API_URL,
    FORECAST_CHART_PATH,
    FORECAST_LOG_PATH,
    LATITUDE,
    LONGITUDE,
    MODEL_PATH,
    OUTPUTS_DIR,
    PREDICTION_HORIZON,
    SCALER_PATH,
    SEQUENCE_LENGTH,
    TARGET_INDEX,
)
from features import build_features
from model import WeatherLSTM
from utils import get_device, to_celsius

# Chart colours
COLOR_OBSERVED = "#4b5563"
COLOR_LSTM = "#e4572e"
COLOR_API = "#2e86de"


# ==========================================
# Fetch recent weather (past 72 h + API forecast for the next hours)
# ==========================================

def fetch_recent_weather():
    """Return (history, api_forecast).

    history      -> the latest 72 complete hourly records (model input)
    api_forecast -> Open-Meteo's own temperature forecast, indexed by time
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(FEATURES),
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "past_hours": SEQUENCE_LENGTH + 24,
        "forecast_hours": PREDICTION_HORIZON + 2,
    }

    print("Fetching recent weather data...")

    response = requests.get(FORECAST_API_URL, params=params, timeout=60)
    response.raise_for_status()

    hourly = response.json().get("hourly")

    if hourly is None:
        raise ValueError("Invalid API response: hourly data not found.")

    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"], utc=True)})

    for feature in FEATURES:
        if feature not in hourly:
            raise ValueError(f"Missing weather feature: {feature}")
        df[feature] = hourly[feature]

    df = df.sort_values("time").reset_index(drop=True)

    current_hour = pd.Timestamp.now(tz="UTC").floor("h")

    history = df[df["time"] < current_hour].tail(SEQUENCE_LENGTH).copy()
    api_future = df[df["time"] >= current_hour].set_index("time")

    if len(history) != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected {SEQUENCE_LENGTH} hourly records, got {len(history)}."
        )

    if not (history["time"].diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("Recent weather data is not hourly and continuous.")

    if not np.isfinite(history[FEATURES].to_numpy(dtype=np.float64)).all():
        raise ValueError("Recent weather data contains missing values.")

    print("Recent weather data fetched successfully.")
    print(f"Latest input timestamp: {history['time'].iloc[-1]}")

    return history.reset_index(drop=True), api_future


# ==========================================
# Load trained model and scaler
# ==========================================

def load_model_and_scaler(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run train.py first.")

    if not SCALER_PATH.exists():
        raise FileNotFoundError("Scaler not found. Run train.py first.")

    scaler = joblib.load(SCALER_PATH)

    model = WeatherLSTM().to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    return model, scaler


# ==========================================
# Chart: LSTM vs Open-Meteo vs recent observations
# ==========================================

def plot_forecast_comparison(
    history_times,
    history_temps,
    future_times,
    lstm_temps,
    api_temps,
    show=True,
):
    tz = ZoneInfo(DISPLAY_TIMEZONE)

    hist_t = history_times[-CHART_HISTORY_HOURS:]
    hist_y = history_temps[-CHART_HISTORY_HOURS:]

    has_api = api_temps is not None and not np.isnan(api_temps).any()

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})

    fig, (ax, ax_diff) = plt.subplots(
        2, 1,
        figsize=(11, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # Connect the history line to the first forecast point
    now_time = hist_t[-1]
    now_temp = hist_y[-1]

    # Future region shading + "now" marker
    ax.axvspan(now_time, future_times[-1], color="#f3f4f6", zorder=0)
    ax_diff.axvspan(now_time, future_times[-1], color="#f3f4f6", zorder=0)
    for a in (ax, ax_diff):
        a.axvline(now_time, color="#9ca3af", linestyle=":", linewidth=1.2)

    ax.text(
        now_time, 1.01, " Forecast starts", transform=ax.get_xaxis_transform(),
        color="#6b7280", fontsize=9, va="bottom",
    )

    # Recent data
    ax.plot(hist_t, hist_y, color=COLOR_OBSERVED, linewidth=2,
            label="Recent data (model input)")

    # LSTM forecast
    ax.plot(
        [now_time, *future_times], [now_temp, *lstm_temps],
        color=COLOR_LSTM, linewidth=2.4, marker="o", markersize=5,
        label="LSTM forecast (this project)",
    )

    # Open-Meteo forecast
    if has_api:
        ax.plot(
            [now_time, *future_times], [now_temp, *api_temps],
            color=COLOR_API, linewidth=2.4, linestyle="--", marker="s",
            markersize=5, label="Open-Meteo forecast (NWP)",
        )
        ax.fill_between(
            future_times, lstm_temps, api_temps,
            color="#a78bfa", alpha=0.18, label="Difference",
        )

    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", frameon=False)

    # Annotate the peak and the final value of the LSTM forecast
    ax.margins(y=0.18)
    i_max = int(np.argmax(lstm_temps))
    for i in sorted({i_max, len(lstm_temps) - 1}):
        ax.annotate(
            f"{lstm_temps[i]:.1f}°C", (future_times[i], lstm_temps[i]),
            textcoords="offset points", xytext=(0, 10), ha="center",
            color=COLOR_LSTM, fontweight="bold",
        )

    # Title with a summary statistic
    if has_api:
        diff = lstm_temps - api_temps
        subtitle = (
            f"Mean |LSTM − Open-Meteo| = {np.mean(np.abs(diff)):.2f} °C   •   "
            f"Largest gap = {np.max(np.abs(diff)):.2f} °C"
        )

        colors = np.where(diff >= 0, COLOR_LSTM, COLOR_API)
        ax_diff.bar(future_times, diff, width=0.03, color=colors, alpha=0.85)
        ax_diff.axhline(0, color="#374151", linewidth=0.8)
        ax_diff.set_ylabel("LSTM − API (°C)")
    else:
        subtitle = "Open-Meteo forecast unavailable for this run"
        ax_diff.text(0.5, 0.5, "API forecast unavailable",
                     transform=ax_diff.transAxes, ha="center", va="center")

    fig.suptitle(
        "Edinburgh 12-hour Temperature Forecast",
        fontsize=15, fontweight="bold", y=0.97,
    )
    ax.set_title(subtitle, fontsize=10, color="#4b5563", pad=18)

    ax_diff.grid(True, linestyle="--", alpha=0.35)
    ax_diff.set_xlabel(f"Time ({DISPLAY_TIMEZONE})")
    ax_diff.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    ax_diff.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=tz))

    fig.text(
        0.01, 0.005,
        "Open-Meteo is a numerical weather prediction model, not ground truth. "
        "Neither forecast has been verified against observations yet.",
        fontsize=8, color="#6b7280",
    )

    FORECAST_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FORECAST_CHART_PATH, dpi=200, bbox_inches="tight")
    print(f"\nForecast chart saved to: {FORECAST_CHART_PATH}")

    if show:
        plt.show()

    plt.close(fig)


# ==========================================
# Forecast log (for later verification against real observations)
# ==========================================

def append_forecast_log(issued_at, future_times, lstm_temps, api_temps):
    log = pd.DataFrame({
        "issued_at_utc": issued_at,
        "target_time_utc": future_times,
        "lstm_temperature": np.round(lstm_temps, 3),
        "open_meteo_temperature": np.round(api_temps, 3),
    })

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    log.to_csv(
        FORECAST_LOG_PATH,
        mode="a",
        header=not FORECAST_LOG_PATH.exists(),
        index=False,
    )

    print(f"Forecast appended to log: {FORECAST_LOG_PATH}")


# ==========================================
# Main prediction pipeline
# ==========================================

def predict_temperature(show=True):
    device = get_device()
    print(f"\nUsing device: {device}")

    history, api_future = fetch_recent_weather()
    model, scaler = load_model_and_scaler(device)

    scaled = scaler.transform(build_features(history)).astype(np.float32)

    X = torch.from_numpy(scaled).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(X).cpu().numpy().flatten()

    lstm_temps = to_celsius(prediction, scaler)

    last_timestamp = history["time"].iloc[-1]
    future_times = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=PREDICTION_HORIZON,
        freq="h",
    )

    # Open-Meteo's own forecast for exactly the same hours
    api_temps = (
        api_future[FEATURES[TARGET_INDEX]]
        .reindex(future_times)
        .to_numpy(dtype=np.float64)
    )

    local_future = future_times.tz_convert(DISPLAY_TIMEZONE)
    local_history = history["time"].dt.tz_convert(DISPLAY_TIMEZONE)

    table = pd.DataFrame({
        "Date": local_future.strftime("%Y-%m-%d"),
        "Time": local_future.strftime("%H:%M"),
        "LSTM (°C)": np.round(lstm_temps, 2),
        "Open-Meteo (°C)": np.round(api_temps, 2),
        "Difference (°C)": np.round(lstm_temps - api_temps, 2),
    })

    print("\n======================================")
    print("EDINBURGH WEATHER FORECAST")
    print("======================================")
    print(table.to_string(index=False))

    plot_forecast_comparison(
        history_times=list(local_history),
        history_temps=history[FEATURES[TARGET_INDEX]].to_numpy(),
        future_times=list(local_future),
        lstm_temps=lstm_temps,
        api_temps=api_temps,
        show=show,
    )

    if not np.isnan(api_temps).any():
        append_forecast_log(
            issued_at=pd.Timestamp.now(tz="UTC").floor("h"),
            future_times=future_times,
            lstm_temps=lstm_temps,
            api_temps=api_temps,
        )

    print("\nAll predictions generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edinburgh 12-hour forecast")
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the chart without opening a window.",
    )
    args = parser.parse_args()

    predict_temperature(show=not args.no_show)