import torch
import torch.nn as nn

def train_model(model, dataloader, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X, y in dataloader:
            optimizer.zero_grad()

            predictions = model(X)
            loss = criterion(predictions, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss / len(dataloader):.4f}")
