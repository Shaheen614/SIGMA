import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(actual, predicted, n=100):
    sns.set(style="darkgrid")

    plt.figure(figsize=(10, 4))
    plt.plot(actual[:n], label="Actual")
    plt.plot(predicted[:n], label="Predicted")
    plt.legend()
    plt.title("LSTM Forecast vs Actual")
    plt.show()
