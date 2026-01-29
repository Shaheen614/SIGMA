import torch
from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_classifier(model, loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in loader:
            preds = model(X)
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())

    accuracy = accuracy_score(y_true, [p > 0.5 for p in y_pred])
    auroc = roc_auc_score(y_true, y_pred)

    return accuracy, auroc
