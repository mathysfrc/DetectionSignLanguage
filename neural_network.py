import numpy as np

class NeuralNetwork:
    def __init__(self):

        # Initialisation des poids et biais avec de petites valeurs aléatoires
        self.weights_input_hidden = np.random.randn(42,
                                                    64) * 0.01  # Poids entre la couche d'entrée  et la couche cachée
        self.bias_input_hidden = np.zeros((1, 64))  # Biais pour la couche cachée

        self.weights_hidden_middle = np.random.randn(64,
                                                     32) * 0.01  # Poids entre la couche cachée et la couche intermédiaire
        self.bias_hidden_middle = np.zeros((1, 32))  # Biais pour la couche intermédiaire

        self.weights_middle_output = np.random.randn(32,
                                                     5) * 0.01  # Poids entre la couche intermédiairenet la couche de sortie
        self.bias_middle_output = np.zeros((1, 5))  # Biais pour la couche de sortie


    # Définition des différentes fonctions d'activation et de leur dérivée
    def relu(self, x):
        # La ReLU est définie comme max(0, x), ce qui signifie que les valeurs négatives deviennent 0,
        # et les valeurs positives restent inchangées.
        return np.maximum(0, x)

    def relu_derivative(self, x):
        # La dérivée de ReLU est 0 pour les entrées négatives et 1 pour les entrées positives.
        # Cela est utilisé pendant la rétropropagation pour ajuster les poids en fonction de l'erreur.
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        # Fonction Softmax pour la normalisation des sorties de la couche de sortie
        # Softmax transforme un vecteur de scores (logits) en une probabilité pour chaque classe.
        exp_x = np.exp(
            x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        # Propagation avant (Forward Pass)

        # Calcul de l'entrée et de la sortie de la couche cachée
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.relu(self.hidden_input)  # Activation ReLU de la couche cachée

        # Calcul de l'entrée et de la sortie de la couche intermédiaire
        self.middle_input = np.dot(self.hidden_output, self.weights_hidden_middle) + self.bias_hidden_middle
        self.middle_output = self.relu(self.middle_input)  # Activation ReLU de la couche intermédiaire

        # Calcul de l'entrée et de la sortie de la couche de sortie
        self.output_input = np.dot(self.middle_output, self.weights_middle_output) + self.bias_middle_output
        self.output = self.softmax(self.output_input)  # Activation Softmax de la couche de sortie

        return self.output

    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]

        y_onehot = np.zeros((m, 5))  # 5 classes possibles
        y_onehot[np.arange(m), y] = 1

        # 6)
        # Calcul de l'erreur de sortie (perte de la couche de sortie)
        dz3 = output - y_onehot  # Calcul d'erreur de la couche de sortie
        dw3 = np.dot(self.middle_output.T, dz3) / m  # Gradient de la perte par rapport aux poids de la couche de sortie
        db3 = np.sum(dz3, axis=0,
                     keepdims=True) / m  # Gradient de la perte par rapport aux biais de la couche de sortie

        # 6.1)
        # Calcul de l'erreur et du gradient pour la couche intermédiaire
        da2 = np.dot(dz3,
                     self.weights_middle_output.T)  # Propagation de l'erreur à travers les poids de la couche intermédiaire
        dz2 = da2 * self.relu_derivative(
            self.middle_input)  # Application de la dérivée de ReLU sur l'entrée de la couche intermédiaire
        dw2 = np.dot(self.hidden_output.T,
                     dz2) / m  # Gradient de la perte par rapport aux poids de la couche intermédiaire
        db2 = np.sum(dz2, axis=0,
                     keepdims=True) / m  # Gradient de la perte par rapport aux biais de la couche intermédiaire

        # 6.2) Calcul de l'erreur et du gradient pour la couche cachée
        da1 = np.dot(dz2,
                     self.weights_hidden_middle.T)  # Propagation de l'erreur à travers les poids de la couche cachée
        dz1 = da1 * self.relu_derivative(
            self.hidden_input)  # Application de la dérivée de ReLU sur l'entrée de la couche cachée
        dw1 = np.dot(X.T, dz1) / m  # Gradient de la perte par rapport aux poids de la couche cachée
        db1 = np.sum(dz1, axis=0, keepdims=True) / m  # Gradient de la perte par rapport aux biais de la couche cachée

        # Mise à jour des poids et biais
        self.weights_middle_output -= learning_rate * dw3  # Mise à jour des poids de la couche de sortie en utilisant le gradient
        self.bias_middle_output -= learning_rate * db3  # Mise à jour des biais de la couche de sortie

        self.weights_hidden_middle -= learning_rate * dw2  # Mise à jour des poids de la couche intermédiaire
        self.bias_hidden_middle -= learning_rate * db2  # Mise à jour des biais de la couche intermédiaire

        self.weights_input_hidden -= learning_rate * dw1  # Mise à jour des poids de la couche cachée
        self.bias_input_hidden -= learning_rate * db1  # Mise à jour des biais de la couche cachée

    def train(self, X, y, epochs, learning_rate):
        # Entraînement du modèle
        for epoch in range(epochs):
            output = self.forward(X)  # Propagation avant
            self.backward(X, y, output, learning_rate)  # Rétropropagation


    def predict(self, X):
        # Prédiction des classes à partir des entrées
        output = self.forward(X)
        return np.argmax(output, axis=1) + 1  # Retourne la classe prédite (entre 1 et 5)
