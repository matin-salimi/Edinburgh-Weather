import torch
import torch.nn as nn

from config import (
    DROPOUT,
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_LAYERS,
    OUTPUT_SIZE,
    RESIDUAL_TARGET,
    TARGET_INDEX,
)


class WeatherLSTM(nn.Module):
    """Stacked LSTM with a linear head for direct multi-step forecasting.

    Input:  (batch, SEQUENCE_LENGTH, INPUT_SIZE)
    Output: (batch, OUTPUT_SIZE)  -> standardized temperatures for the next hours
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        output_size: int = OUTPUT_SIZE,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        last_output = lstm_output[:, -1, :]
        output = self.fc(last_output)

        if RESIDUAL_TARGET:
            # Residual learning: the network only predicts the CHANGE relative
            # to the last observed (standardized) temperature.
            output = output + x[:, -1, TARGET_INDEX].unsqueeze(1)

        return output