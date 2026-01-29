import torch.nn as nn

class LSTMForecast(nn.Module):
    """
    Simple LSTM for time-series forecasting
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()

        #LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        #fully connected 
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x shape: (batch, sequence_length, input_size)
        """
        out, _ = self.lstm(x)

        # Take output from last timestep
        out = out[:, -1, :]

        #final prediction
        out = self.fc(out)

        return out.squeeze(-1)
