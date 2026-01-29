import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/uci_grid_stability.csv")

X = df.drop(columns=["stab", "stabf"])
y = (df["stabf"] == "stable").astype(int)

split = int(0.8 * len(df))

X_train = X.iloc[:split]
y_train = y.iloc[:split]

X_test = X.iloc[split:]
y_test = y.iloc[split:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)


