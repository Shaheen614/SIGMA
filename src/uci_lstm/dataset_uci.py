import torch
from torch.utils.data import Dataset

class UCITimeSeriesDataset(Dataset):
    def __init__(self, X, y, window_size=10):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.window_size]
        y_label = self.y[idx + self.window_size]
        return X_seq, y_label
    
    
