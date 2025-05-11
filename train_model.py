import pandas as pd
import numpy as np
from neural_network import NeuralNetwork

# Fonction pour convertir les labels en one-hot
def convert_to_one_hot(y, num_classes=5):
    return np.eye(num_classes)[y]

def load_and_prepare_data():
    # Charger les données
    data = pd.read_csv('src/datas/data_formatted.csv', header=None)

    # Afficher le nombre de colonnes et les dimensions de X
    print(f"Nombre total de colonnes dans le CSV : {data.shape[1]}")

    # Séparer les features (X) et les labels (y)
    X = data.iloc[:, :42].values  # Les 42 premières colonnes sont les features
    y_onehot = data.iloc[:, 42:].values  # Les colonnes suivantes sont les labels one-hot
    y = np.argmax(y_onehot, axis=1)  # Transformation en labels catégoriels

    # Convertir y en vecteur one-hot
    y_one_hot = convert_to_one_hot(y)

    print(f"Forme de X : {X.shape}")
    print(f"Exemple de y : {y[:5]}")

    # Division des données en train et validation selon les intervalles spécifiés
    train_idx = np.concatenate([
        np.arange(0, 50),    # Pour les indices 1 à 50
        np.arange(60, 110),  # Pour les indices 61 à 110
        np.arange(120, 170), # Pour les indices 121 à 170
        np.arange(180, 230), # Pour les indices 181 à 230
        np.arange(240, 290)  # Pour les indices 241 à 290
    ])

    val_idx = np.concatenate([
        np.arange(50, 60),   # Pour les indices 51 à 60
        np.arange(110, 120), # Pour les indices 111 à 120
        np.arange(170, 180), # Pour les indices 171 à 180
        np.arange(230, 240), # Pour les indices 231 à 240
        np.arange(290, 300)  # Pour les indices 291 à 300
    ])

    # Création des ensembles d'entraînement et de validation
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_one_hot[train_idx], y_one_hot[val_idx]

    return X_train, X_val, y_train, y_val

def train():
    # Charger et préparer les données
    X_train, X_val, y_train, y_val = load_and_prepare_data()

    # Initialiser le modèle
    nn = NeuralNetwork()

    # Entraînement du modèle
    nn.train(X_train, y_train, epochs=1000, learning_rate=0.5)  # Plus d'epochs, learning_rate augmenté

    # Faire des prédictions sur le jeu de validation
    predictions = nn.predict(X_val)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_val, axis=1))  # Comparaison avec les labels réels
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Sauvegarder les poids du modèle
    np.savez('model_weights.npz',
             w1=nn.weights_input_hidden, b1=nn.bias_input_hidden,
             w2=nn.weights_hidden_middle, b2=nn.bias_hidden_middle,
             w3=nn.weights_middle_output, b3=nn.bias_middle_output)

if __name__ == "__main__":
    train()
