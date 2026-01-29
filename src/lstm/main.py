from pyexpat import features
from unittest import loader
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from train import train_model
from evaluate import evaluate_model
from visualize import plot_predictions

from sklearn.preprocessing import MinMaxScaler

#----- PREPROCESSING -----

#load opsd
df = pd.read_csv("data/time_series_60min_singleindex.csv", parse_dates=["utc_timestamp"])

opsd = df[
    [
        "utc_timestamp",
        "AT_load_actual_entsoe_transparency",
        "AT_solar_generation_actual"
    ]
].dropna()

opsd = opsd.rename(columns={
    "AT_load_actual_entsoe_transparency": "load",
    "AT_solar_generation_actual": "solar"
})


#load pv data
pv = pd.read_csv("data/pv_gecad_2019.csv")

#combine date and time into utc_timestamp
pv["utc_timestamp"] = pd.to_datetime(
    pv["date"].astype(str) + " " + pv["time"].astype(str),
    errors="coerce"
)

pv["utc_timestamp"] = pv["utc_timestamp"].dt.tz_localize("UTC")

pv = pv.drop(columns=["date", "time"])

pv = pv.rename(columns={
    "generation (w)": "pv_power",
    "temperature (˚C)": "temperature"
})

pv = pv.dropna(subset=["utc_timestamp"])

#resample to hourly
pv = (
    pv.set_index("utc_timestamp")
      .resample("1h")
      .mean()
      .reset_index()
)

#merge opsd and pv
merged = opsd.merge(
    pv,
    on="utc_timestamp",
    how="inner"
)

features = merged[["load", "solar", "pv_power", "temperature"]] 

#---------------------------------------------------------------------

#train-test split
split_idx = int(0.8 * len(features))

train_data = features.iloc[:split_idx]
test_data  = features.iloc[split_idx:]

#scaling
feature_scaler = MinMaxScaler()
load_scaler = MinMaxScaler()

X_train = feature_scaler.fit_transform(train_data.values)
X_test = feature_scaler.transform(test_data.values)

y_train = load_scaler.fit_transform(train_data[["load"]].values).flatten()
y_test = load_scaler.transform(test_data[["load"]].values).flatten()


#Dataset & DataLoader
WINDOW_SIZE = 24
BATCH_SIZE = 16

#create datasets and dataloaders
train_dataset = TimeSeriesDataset(X_train, WINDOW_SIZE)
test_dataset  = TimeSeriesDataset(X_test, WINDOW_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


#Model
model = LSTMForecast(input_size=4)

#Train
train_model(model, train_loader, epochs=80)

#Evaluate
mae, rmse = evaluate_model(model, test_loader)
print("MAE:", mae)
print("RMSE:", rmse)

#Visualize
# regenerate predictions for plotting
actual, predicted = [], []

import torch
model.eval()
with torch.no_grad():
    for X, y in test_loader:
        preds = model(X)

        actual.extend(
            load_scaler.inverse_transform(
                y.numpy().reshape(-1, 1)
            ).flatten()
        )

        predicted.extend(
            load_scaler.inverse_transform(
                preds.numpy().reshape(-1, 1)
            ).flatten()
        )

plot_predictions(actual, predicted)


