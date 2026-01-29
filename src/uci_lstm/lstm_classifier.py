import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size = 64):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        out,_ = self.lstm(x)
        out = out[:, -1, :]  #final timestep
        out = self.fc(out)
        return self.sigmoid(out).squeeze()