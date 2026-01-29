import torch
import torch.nn as nn

def train_model(model, loader, epochs = 50, lr =0.01):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X, y in loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(loader):.4f}")
        