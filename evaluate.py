
import torch
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import prepare_datasets
from model import WeatherLSTM


def evaluate_model():

    project_dir = Path(__file__).resolve().parent
    models_dir = project_dir / "models"

    model_path = models_dir / "edinburgh_lstm.pth"
    scaler_path = models_dir / "edinburgh_scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError("Trained model not found.")

    # Load datasets and scaler
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaler
    ) = prepare_datasets()

    # Save scaler
    joblib.dump(scaler, scaler_path)

    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load trained model
    model = WeatherLSTM().to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    # Generate LSTM predictions
    X_test_tensor = torch.from_numpy(X_test).to(device)

    with torch.no_grad():
        predictions = model(X_test_tensor)

    predictions = predictions.cpu().numpy()

    # Convert normalized temperatures to Celsius
    temperature_mean = scaler.mean_[0]
    temperature_scale = scaler.scale_[0]

    predictions_celsius = (
        predictions * temperature_scale + temperature_mean
    )

    actual_celsius = (
        y_test * temperature_scale + temperature_mean
    )

    # Calculate LSTM metrics
    lstm_mae = mean_absolute_error(
        actual_celsius.flatten(),
        predictions_celsius.flatten()
    )

    lstm_rmse = np.sqrt(
        mean_squared_error(
            actual_celsius.flatten(),
            predictions_celsius.flatten()
        )
    )

    # ==========================================
    # Persistence Baseline
    # ==========================================

    last_temperatures = (
        X_test[:, -1, 0] * temperature_scale
        + temperature_mean
    )

    baseline_predictions = np.repeat(
        last_temperatures[:, np.newaxis],
        y_test.shape[1],
        axis=1
    )

    baseline_mae = mean_absolute_error(
        actual_celsius.flatten(),
        baseline_predictions.flatten()
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            actual_celsius.flatten(),
            baseline_predictions.flatten()
        )
    )

    # ==========================================
    # Results
    # ==========================================

    print("\n========== MODEL EVALUATION ==========")

    print(f"Device: {device}")
    print(f"Test Samples: {len(X_test)}")

    print("\n--- LSTM Model ---")
    print(f"MAE: {lstm_mae:.4f} °C")
    print(f"RMSE: {lstm_rmse:.4f} °C")

    print("\n--- Persistence Baseline ---")
    print(f"MAE: {baseline_mae:.4f} °C")
    print(f"RMSE: {baseline_rmse:.4f} °C")

    print("\n--- Comparison ---")

    if lstm_mae < baseline_mae:
        print("LSTM performs better than Baseline (MAE).")
    else:
        print("Baseline performs better than LSTM (MAE).")

    print("\nModel and scaler saved successfully.")

    print(f"Model: {model_path}")
    print(f"Scaler: {scaler_path}")


if __name__ == "__main__":
    evaluate_model()
