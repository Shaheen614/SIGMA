import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, dataloader):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in dataloader:
            preds = model(X)
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    return mae, rmse
