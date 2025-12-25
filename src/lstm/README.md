# LSTM Time-Series Forecasting Module

This module implements the LSTM-based time-series forecasting component
of the Smart Grid Fault Detection project.

## Purpose
The LSTM model learns temporal patterns from historical time-series data
(voltage, current, power, frequency, etc.) and predicts future values.
This supports:
- Grid instability detection
- Power quality analysis
- Trend-based anomaly detection

## Contents
- `dataset.py` – Sliding window time-series dataset
- `model.py` – LSTM forecasting model
- `train.py` – Training loop
- `evaluate.py` – MAE and RMSE evaluation
- `visualize.py` – Prediction vs actual plots

## Current Status
- LSTM skeleton implemented
- Sliding window preprocessing complete
- Training and evaluation pipeline functional
- Uses dummy data for initial validation

## Future Extensions
- Integration with OPSD dataset
- GRU comparison
- Anomaly detection using prediction error
- Multi-feature inputs (irradiance, temperature)

## Member
Mishel
