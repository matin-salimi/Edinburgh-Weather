"""Baseline forecasters used as a fair yardstick for the LSTM.

All functions return predictions in degrees Celsius with shape
(n_samples, PREDICTION_HORIZON).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config import (
    FEATURES,
    PREDICTION_HORIZON,
    SEQUENCE_LENGTH,
    TARGET_INDEX,
    TRAIN_END,
)
from data import load_dataset
from utils import to_celsius


def persistence(X_test, scaler):
    """Every future hour equals the last observed temperature."""
    last = to_celsius(X_test[:, -1, TARGET_INDEX], scaler)
    return np.repeat(last[:, np.newaxis], PREDICTION_HORIZON, axis=1)


def seasonal_persistence(X_test, scaler):
    """Each future hour equals the temperature at the same hour 24 h earlier."""
    assert PREDICTION_HORIZON <= 24, "Needs horizon <= 24 hours."

    start = SEQUENCE_LENGTH - 24
    window = X_test[:, start:start + PREDICTION_HORIZON, TARGET_INDEX]

    return to_celsius(window, scaler)


def climatology(test_times):
    """Mean training-set temperature for the same calendar month and hour."""
    df = load_dataset()
    train = df[df["time"] < pd.Timestamp(TRAIN_END, tz="UTC")]

    table = train.groupby(
        [train["time"].dt.month.rename("m"), train["time"].dt.hour.rename("h")]
    )[FEATURES[TARGET_INDEX]].mean()

    predictions = np.zeros((len(test_times), PREDICTION_HORIZON))

    for k in range(PREDICTION_HORIZON):
        target_time = test_times + pd.Timedelta(hours=k)
        keys = pd.MultiIndex.from_arrays(
            [target_time.dt.month, target_time.dt.hour]
        )
        predictions[:, k] = table.reindex(keys).to_numpy()

    return predictions


def ridge_regression(X_train, y_train, X_test, scaler, alpha=1.0):
    """Linear model on the flattened 72-hour window (fitted on train only)."""
    model = Ridge(alpha=alpha)
    model.fit(X_train.reshape(len(X_train), -1), y_train)

    scaled_pred = model.predict(X_test.reshape(len(X_test), -1))

    return to_celsius(scaled_pred, scaler)