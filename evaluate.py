import joblib
import numpy as np
import torch

from baselines import (
    climatology,
    persistence,
    ridge_regression,
    seasonal_persistence,
)
from config import EXPERIMENT, SEED, METRICS_PATH, MODEL_PATH, SCALER_PATH
from data import prepare_datasets
from model import WeatherLSTM
from utils import get_device, save_json, to_celsius


def regression_metrics(actual, predicted) -> dict:
    error = predicted - actual

    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
        "mae_per_horizon": np.mean(np.abs(error), axis=0).tolist(),
    }


def predict_in_batches(model, X, device, batch_size=1024):
    outputs = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start:start + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())

    return np.concatenate(outputs, axis=0)


def evaluate_model():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            "Trained model or scaler not found. Run train.py first."
        )

    datasets = prepare_datasets()
    X_train, y_train = datasets["X_train"], datasets["y_train"]
    X_test, y_test = datasets["X_test"], datasets["y_test"]

    scaler = joblib.load(SCALER_PATH)
    device = get_device()

    model = WeatherLSTM().to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    actual = to_celsius(y_test, scaler)

    predictions = {
        "Persistence": persistence(X_test, scaler),
        "Seasonal persistence (24h)": seasonal_persistence(X_test, scaler),
        "Climatology (month x hour)": climatology(datasets["test_times"]),
        "Ridge regression": ridge_regression(
            X_train, y_train, X_test, scaler
        ),
        "LSTM": to_celsius(
            predict_in_batches(model, X_test, device), scaler
        ),
    }

    results = {
        "experiment": EXPERIMENT,
        "seed": SEED,
        "test_samples": int(len(X_test)),
        "test_period": {
            "first_forecast_start": str(datasets["test_times"].iloc[0]),
            "last_forecast_start": str(datasets["test_times"].iloc[-1]),
        },
        "models": {
            name: regression_metrics(actual, pred)
            for name, pred in predictions.items()
        },
    }

    lstm_mae = results["models"]["LSTM"]["mae"]
    results["lstm_mae_improvement_percent"] = {
        name: float((1 - lstm_mae / m["mae"]) * 100)
        for name, m in results["models"].items()
        if name != "LSTM"
    }

    save_json(results, METRICS_PATH)

    # ---------- Report ----------
    print("\n========== MODEL EVALUATION ==========")
    print(f"Experiment: {EXPERIMENT}   Device: {device}   Test samples: {len(X_test)}\n")

    print(f"{'Model':<30}{'MAE':>8}{'RMSE':>8}{'Bias':>8}")
    for name, m in sorted(
        results["models"].items(), key=lambda item: item[1]["mae"]
    ):
        print(
            f"{name:<30}{m['mae']:>8.3f}{m['rmse']:>8.3f}{m['bias']:>+8.3f}"
        )

    print("\nLSTM MAE improvement vs each baseline:")
    for name, pct in results["lstm_mae_improvement_percent"].items():
        print(f"  {name:<30}{pct:>7.1f}%")

    names = list(results["models"])
    print("\nMAE by forecast horizon (°C):")
    print(f"{'':<6}" + "".join(f"{n[:12]:>14}" for n in names))
    for h in range(len(actual[0])):
        row = "".join(
            f"{results['models'][n]['mae_per_horizon'][h]:>14.3f}"
            for n in names
        )
        print(f"h+{h + 1:<4}{row}")

    print(f"\nMetrics saved at: {METRICS_PATH}")


if __name__ == "__main__":
    evaluate_model()