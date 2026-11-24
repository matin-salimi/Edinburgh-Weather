"""Central configuration for the Edinburgh LSTM weather forecasting project.

Every path, date, feature name and hyperparameter lives here so that
experiments are reproducible and nothing is hard-coded elsewhere.
"""

import os
from pathlib import Path

# ==========================================
# Paths
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "dataset"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"   # metrics, histories, forecast log
FIGURES_DIR = PROJECT_DIR / "figures"   # images used in the README

# ==========================================
# Experiments (ablation study)
# Select with an environment variable, e.g.:  EXPERIMENT=time python3 train.py
# ==========================================

EXPERIMENTS = {
    "baseline":    dict(time_features=False, residual=False, regularization=False),
    "time":        dict(time_features=True,  residual=False, regularization=False),
    "residual":    dict(time_features=False, residual=True,  regularization=False),
    "regularized": dict(time_features=False, residual=False, regularization=True),
    "time_residual": dict(time_features=True, residual=True, regularization=False),
    "full":        dict(time_features=True,  residual=True,  regularization=True),
}

# The model used by predict.py and for the headline results.
# (time_residual == full in accuracy; regularization had no measurable effect.)
FINAL_EXPERIMENT = "time_residual"

EXPERIMENT = os.environ.get("EXPERIMENT", FINAL_EXPERIMENT)

if EXPERIMENT not in EXPERIMENTS:
    raise ValueError(
        f"Unknown EXPERIMENT '{EXPERIMENT}'. Choose from: {list(EXPERIMENTS)}"
    )

# Random seed (override with e.g. SEED=1). Non-default seeds get their own files.
SEED = int(os.environ.get("SEED", "42"))
RUN_NAME = EXPERIMENT if SEED == 42 else f"{EXPERIMENT}_s{SEED}"

_flags = EXPERIMENTS[EXPERIMENT]

USE_TIME_FEATURES = _flags["time_features"]  # sin/cos of hour-of-day and day-of-year
RESIDUAL_TARGET = _flags["residual"]         # model predicts change from last temperature
USE_REGULARIZATION = _flags["regularization"]  # weight decay + LR scheduler + grad clipping

# ==========================================
# Output paths (one set per experiment)
# ==========================================

DATASET_PATH = DATASET_DIR / "edinburgh_era5.csv"

MODEL_PATH = MODELS_DIR / f"lstm_{RUN_NAME}.pth"
SCALER_PATH = MODELS_DIR / f"scaler_{RUN_NAME}.pkl"

HISTORY_PATH = OUTPUTS_DIR / f"training_history_{RUN_NAME}.json"
METRICS_PATH = OUTPUTS_DIR / f"metrics_{RUN_NAME}.json"
FORECAST_CHART_PATH = FIGURES_DIR / "edinburgh_weather_forecast.png"
FORECAST_LOG_PATH = OUTPUTS_DIR / "forecast_log.csv"

# ==========================================
# Location
# ==========================================

LATITUDE = 55.9533
LONGITUDE = -3.1883

DATA_TIMEZONE = "UTC"          # all stored data is in UTC
DISPLAY_TIMEZONE = "Europe/London"  # used only for charts and tables

# ==========================================
# Data source (ERA5 reanalysis via Open-Meteo Archive API)
# ==========================================

ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

# Fixed dates -> the dataset is identical on every machine.
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",
    "cloud_cover",
]

TARGET_INDEX = 0  # temperature_2m

# ==========================================
# Chronological split (whole calendar years)
# Train: 2015-2022 | Validation: 2023 | Test: 2024-2025
# Boundaries are the first timestamp of the NEXT split (UTC).
# ==========================================

TRAIN_END = "2023-01-01"
VAL_END = "2024-01-01"

# ==========================================
# Sliding window
# ==========================================

SEQUENCE_LENGTH = 72
PREDICTION_HORIZON = 12

# ==========================================
# Model
# ==========================================

TIME_FEATURE_NAMES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

INPUT_SIZE = len(FEATURES) + (len(TIME_FEATURE_NAMES) if USE_TIME_FEATURES else 0)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
OUTPUT_SIZE = PREDICTION_HORIZON

# ==========================================
# Training
# ==========================================

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10

WEIGHT_DECAY = 1e-4 if USE_REGULARIZATION else 0.0
USE_SCHEDULER = USE_REGULARIZATION   # ReduceLROnPlateau
GRAD_CLIP = 1.0 if USE_REGULARIZATION else None

# ==========================================
# Prediction chart
# ==========================================

CHART_HISTORY_HOURS = 24  # how many past hours to show on the forecast chart