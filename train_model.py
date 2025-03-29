import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from neural_network import NeuralNetwork


def load_model():
    nn = NeuralNetwork()
    weights = np.load('model_weights.npz')
    nn.weights1 = weights['w1']
    nn.bias1 = weights['b1']
    nn.weights2 = weights['w2']
    nn.bias2 = weights['b2']
    nn.weights3 = weights['w3']
    nn.bias3 = weights['b3']
    return nn


def preprocess_image(features):
    if len(features) != 42:
        raise ValueError("L'image doit avoir 42 coordonnées")
    return np.array([features])


def predict(features):
    nn = load_model()
    X = preprocess_image(features)
    probabilities = nn.forward(X)
    print(f"Probabilités par classe (A, B, C, D, E) : {probabilities[0]}")
    prediction = nn.predict(X)[0]

    print(f"Première coordonnée utilisée : {features[0]}")
    print(f"Classe prédite (numéro) : {prediction}")

    letters = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
    return letters[prediction]


def extract_features_from_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"L'image {image_path} n'existe pas")

    print(f"Chargement de l'image : {image_path}")

    data = pd.read_csv('src/datas/data_formatted.csv', header=None)

    filename = os.path.basename(image_path)
    try:
        image_num = int(''.join(filter(str.isdigit, filename))) - 1
    except ValueError:
        raise ValueError(f"Impossible d'extraire le numéro de l'image depuis {filename}")

    if image_num < 0 or image_num >= len(data):
        raise ValueError(f"Numéro d'image {image_num} hors plage (0 à {len(data) - 1})")

    features = data.iloc[image_num, :42].values
    true_label_onehot = data.iloc[image_num, 42:].values
    true_label = np.argmax(true_label_onehot) + 1

    return features, true_label


def predict_from_image_path(image_path):
    features, true_label = extract_features_from_image(image_path)
    predicted_letter = predict(features)
    return predicted_letter, true_label


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['A', 'B', 'C', 'D', 'E'], yticklabels=['A', 'B', 'C', 'D', 'E'])
    plt.title("Matrice de confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Véritables labels")
    plt.show()


def plot_accuracy_loss(loss_history, accuracy_history):
    plt.figure(figsize=(12, 6))

    # Courbe de la perte
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title("Courbe de la perte")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Courbe de la précision
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)
    plt.title("Courbe de la précision")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Remplacez cette ligne avec le chemin réel de l'image à tester
    image_path = "src/validationDataset/52.jpg"  # Ajusté selon votre sortie

    predicted_letter, true_label = predict_from_image_path(image_path)

    # Afficher le résultat
    letters = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
    print(f"Image : {image_path}")
    print(f"Lettre prédite : {predicted_letter}")
    print(f"Lettre réelle (selon CSV) : {letters[true_label]}")

    # Pour afficher la matrice de confusion et les courbes (s'il y a des données d'entraînement disponibles)
    # Ces deux fonctions sont appelées seulement si vous avez les historiques de perte et précision.
    plot_accuracy_loss([0.5, 0.4, 0.3], [0.7, 0.75, 0.8])  # Exemple avec des données fictives
    plot_confusion_matrix([1, 2, 1, 3], [1, 2, 1, 4])  # Exemple avec des données fictives
