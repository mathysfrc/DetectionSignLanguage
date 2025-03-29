import pandas as pd
import numpy as np
from neural_network import NeuralNetwork


def load_and_prepare_data():
    data = pd.read_csv('src/datas/data_formatted.csv', header=None)

    print(f"Nombre total de colonnes dans le CSV : {data.shape[1]}")

    X = data.iloc[:, :42].values
    y_onehot = data.iloc[:, 42:].values
    y = np.argmax(y_onehot, axis=1)

    print(f"Forme de X : {X.shape}")
    print(f"Exemple de y : {y[:5]}")

    indices = np.random.permutation(X.shape[0])
    train_idx, val_idx = indices[:250], indices[250:300]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    return X_train, X_val, y_train, y_val


def train():
    X_train, X_val, y_train, y_val = load_and_prepare_data()

    nn = NeuralNetwork()
    nn.train(X_train, y_train, epochs=1000, learning_rate=0.5)  # Plus d'epochs, learning_rate augmenté

    predictions = nn.predict(X_val)
    accuracy = np.mean(predictions == (y_val + 1))
    print(f"Validation Accuracy: {accuracy:.4f}")

    np.savez('model_weights.npz',
             w1=nn.weights1, b1=nn.bias1,
             w2=nn.weights2, b2=nn.bias2,
             w3=nn.weights3, b3=nn.bias3)


if __name__ == "__main__":
    train()