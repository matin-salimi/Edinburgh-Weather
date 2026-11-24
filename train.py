import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    EPOCHS,
    HISTORY_PATH,
    LEARNING_RATE,
    MODEL_PATH,
    MODELS_DIR,
    PATIENCE,
    SCALER_PATH,
    SEED,
    EXPERIMENT,
    GRAD_CLIP,
    USE_SCHEDULER,
    WEIGHT_DECAY,
)
from data import prepare_datasets
from model import WeatherLSTM
from utils import get_device, save_json, set_seed


def create_data_loaders(X_train, y_train, X_val, y_val):
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader


def train_model():
    set_seed(SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"\nUsing device: {device}")
    print(f"Random seed: {SEED}")
    print(f"Experiment: {EXPERIMENT}   Seed: {SEED}")

    print("\nPreparing datasets...")
    datasets = prepare_datasets()

    # The scaler belongs to training, so it is saved here (not in evaluate.py).
    joblib.dump(datasets["scaler"], SCALER_PATH)
    print(f"Scaler saved at: {SCALER_PATH}")

    train_loader, val_loader = create_data_loaders(
        datasets["X_train"],
        datasets["y_train"],
        datasets["X_val"],
        datasets["y_val"],
    )

    model = WeatherLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        if USE_SCHEDULER
        else None
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {"train_loss": [], "val_loss": []}

    print("\nStarting LSTM training...\n")

    for epoch in range(1, EPOCHS + 1):

        # ---------- Training ----------
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()

            if GRAD_CLIP is not None:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                loss = criterion(model(X_batch), y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        # ---------- Checkpoint + early stopping ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("Best model saved.")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    history["seed"] = SEED
    history["experiment"] = EXPERIMENT
    save_json(history, HISTORY_PATH)

    print("\nTraining completed!")
    print(f"Best epoch: {best_epoch}")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Model saved at: {MODEL_PATH}")
    print(f"Training history saved at: {HISTORY_PATH}")


if __name__ == "__main__":
    train_model()