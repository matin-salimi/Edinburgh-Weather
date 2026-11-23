import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import prepare_datasets
from model import WeatherLSTM


# ==========================================
# Configuration
# ==========================================

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

PATIENCE = 10

MODEL_NAME = "edinburgh_lstm.pth"


# ==========================================
# Device Configuration
# ==========================================

def get_device():

    if torch.backends.mps.is_available():

        device = torch.device("mps")

    elif torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print(f"\nUsing device: {device}")

    return device


# ==========================================
# Create Data Loaders
# ==========================================

def create_data_loaders(
    X_train,
    y_train,
    X_val,
    y_val
):

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader


# ==========================================
# Training Function
# ==========================================

def train_model():

    # Project paths
    project_dir = Path(__file__).resolve().parent

    models_dir = project_dir / "models"

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = models_dir / MODEL_NAME

    # Device
    device = get_device()

    # Load datasets
    print("\nPreparing datasets...")

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaler
    ) = prepare_datasets()

    # Data loaders
    train_loader, val_loader = create_data_loaders(
        X_train,
        y_train,
        X_val,
        y_val
    )

    # Initialize model
    model = WeatherLSTM().to(device)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Early stopping variables
    best_val_loss = float("inf")

    patience_counter = 0

    print("\nStarting LSTM training...\n")

    # ======================================
    # Training Loop
    # ======================================

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)

            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # ==================================
        # Validation
        # ==================================

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(device)

                y_batch = y_batch.to(device)

                predictions = model(X_batch)

                loss = criterion(
                    predictions,
                    y_batch
                )

                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        # ==================================
        # Save Best Model
        # ==================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0

            torch.save(
                model.state_dict(),
                model_path
            )

            print("Best model saved.")

        else:

            patience_counter += 1

        # ==================================
        # Early Stopping
        # ==================================

        if patience_counter >= PATIENCE:

            print("\nEarly stopping triggered.")

            break

    print("\nTraining completed!")

    print(f"Best Validation Loss: {best_val_loss:.6f}")

    print(f"Model saved at: {model_path}")


# ==========================================
# Run Training
# ==========================================

if __name__ == "__main__":

    train_model()