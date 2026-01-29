from torch.utils.data import DataLoader
from preprocess_uci import X_train_scaled, y_train, X_test_scaled, y_test
from dataset_uci import UCITimeSeriesDataset
from lstm_classifier import LSTMClassifier
from train_uci import train_model
from evaluate_uci import evaluate_classifier

WINDOW = 4
BATCH_SIZE = 32

train_dataset = UCITimeSeriesDataset(X_train_scaled, y_train.values, window_size=WINDOW)
test_dataset = UCITimeSeriesDataset(X_test_scaled, y_test.values, window_size=WINDOW)   

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = LSTMClassifier(input_size=X_train_scaled.shape[1])

train_model(model, train_loader, epochs=50)

acc, auc, = evaluate_classifier(model, test_loader)
print(f"Test Accuracy: {acc:.4f}, Test AUROC: {auc:.4f}")



