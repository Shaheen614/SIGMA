import numpy as np
from torch.utils.data import DataLoader

from data.timeseries_dataset import TimeSeriesDataset
from models.lstm_model import LSTMForecast
from train import train_model
from evaluate import evaluate_model
from visualize import plot_predictions

from sklearn.preprocessing import MinMaxScaler

#Create dummy time-series
np.random.seed(42)
time = np.arange(0, 300)
data = 50 + 10 * np.sin(0.05 * time) + np.random.normal(0, 2, size=300)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

#Dataset & DataLoader
WINDOW_SIZE = 24
BATCH_SIZE = 16

dataset = TimeSeriesDataset(data_scaled, WINDOW_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#Model
model = LSTMForecast()

#Train
train_model(model, loader, epochs=20)

#Evaluate
mae, rmse = evaluate_model(model, loader)
print("MAE:", mae)
print("RMSE:", rmse)

#Visualize
# regenerate predictions for plotting
actual, predicted = [], []

import torch
model.eval()
with torch.no_grad():
    for X, y in loader:
        preds = model(X)
        actual = scaler.inverse_transform(y.numpy().reshape(-1, 1)).flatten().tolist()
        predicted = scaler.inverse_transform(preds.numpy().reshape(-1, 1)).flatten()

plot_predictions(actual, predicted)
