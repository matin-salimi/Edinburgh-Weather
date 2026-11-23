import torch
import torch.nn as nn


class WeatherLSTM(nn.Module):

    def __init__(
        self,
        input_size=6,
        hidden_size=64,
        num_layers=2,
        output_size=12,
        dropout=0.2
    ):

        super(WeatherLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        lstm_output, (hidden_state, cell_state) = self.lstm(x)

        last_output = lstm_output[:, -1, :]

        predictions = self.fc(last_output)

        return predictions
    