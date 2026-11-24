"""Feature engineering shared by training and live prediction."""

import numpy as np
import pandas as pd

from config import FEATURES, USE_TIME_FEATURES


def build_features(df: pd.DataFrame) -> np.ndarray:
    """Return the model input matrix (n_rows, INPUT_SIZE).

    Column 0 is always temperature (the target variable). When enabled, four
    cyclical time features are appended so the network can know the hour of
    day and the season: sin/cos(hour) and sin/cos(day of year). Time is UTC,
    identical in training and prediction.
    """
    values = df[FEATURES].to_numpy(dtype=np.float64)

    if not USE_TIME_FEATURES:
        return values

    t = pd.to_datetime(df["time"], utc=True)

    hour = t.dt.hour.to_numpy() + t.dt.minute.to_numpy() / 60.0
    day_of_year = t.dt.dayofyear.to_numpy()
    days_in_year = np.where(t.dt.is_leap_year.to_numpy(), 366.0, 365.0)

    time_features = np.column_stack([
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * day_of_year / days_in_year),
        np.cos(2 * np.pi * day_of_year / days_in_year),
    ])

    return np.column_stack([values, time_features])