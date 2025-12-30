import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Autoenc(nn.Module):
                         def __init__(self, in_dim=100, lat_dim=32):
                          super(Autoenc, self).__init__()
        
                          self.enc = nn.Sequential(
                           nn.Linear(in_dim, 128),
                           nn.ReLU(),
                           nn.Linear(128, 64),
                            nn.ReLU(),
                           nn.Linear(64, lat_dim),
                           nn.ReLU()
        )
        
                          self.dec = nn.Sequential(
                          nn.Linear(lat_dim, 64),
                           nn.ReLU(),
                           nn.Linear(64, 128),
                          nn.ReLU(),
                           nn.Linear(128, in_dim),
                           nn.Sigmoid() 
        )

                         def forward(self, x):
                             z = self.enc(x)
                             out = self.dec(z)
                             return out

df = pd.read_csv("data/Data_for_UCI_named.csv")
df = df.select_dtypes(include=['float64','int64'])


scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df.values)
data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


input_dim = data_tensor.shape[1]  # match dataset features
model = Autoenc(in_dim=input_dim, lat_dim=32)
critic = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 80
for epoch in range(epochs):
                   for batch in dataloader:
                    x = batch[0]
                    optimizer.zero_grad()
                    output = model(x)
                    loss = critic(output, x)  
                    loss.backward()
                    optimizer.step()
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
