import numpy as np

class NeuralNetwork:
    def __init__(self):
        # Initialisation des poids et biais avec de petites valeurs aléatoires pour chaque couche
        self.weights_input_hidden = np.random.randn(42, 64) * 0.01
        self.bias_input_hidden = np.zeros((1, 64))

        self.weights_hidden_middle = np.random.randn(64, 32) * 0.01
        self.bias_hidden_middle = np.zeros((1, 32))

        self.weights_middle_output = np.random.randn(32, 5) * 0.01
        self.bias_middle_output = np.zeros((1, 5))

    # Fonction d'activation ReLU
    def relu(self, x):
        return np.maximum(0, x)

    # Dérivée de ReLU
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    # Fonction Softmax
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # Propagation avant
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.relu(self.hidden_input)

        self.middle_input = np.dot(self.hidden_output, self.weights_hidden_middle) + self.bias_hidden_middle
        self.middle_output = self.relu(self.middle_input)

        self.output_input = np.dot(self.middle_output, self.weights_middle_output) + self.bias_middle_output
        self.output = self.softmax(self.output_input)

        return self.output

    # Fonction de rétropropagation avec erreur quadratique
    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]  # Nombre d'exemples dans le batch

        # Calcul de l'erreur pour la couche de sortie (différence entre la sortie et la vérité terrain)
        dz3 = output - y  # Erreur quadratique

        dw3 = np.dot(self.middle_output.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        # Calcul de l'erreur pour la couche intermédiaire
        da2 = np.dot(dz3, self.weights_middle_output.T)
        dz2 = da2 * self.relu_derivative(self.middle_input)

        dw2 = np.dot(self.hidden_output.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Calcul de l'erreur pour la couche cachée
        da1 = np.dot(dz2, self.weights_hidden_middle.T)
        dz1 = da1 * self.relu_derivative(self.hidden_input)

        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Mise à jour des poids et biais
        self.weights_middle_output -= learning_rate * dw3
        self.bias_middle_output -= learning_rate * db3

        self.weights_hidden_middle -= learning_rate * dw2
        self.bias_hidden_middle -= learning_rate * db2

        self.weights_input_hidden -= learning_rate * dw1
        self.bias_input_hidden -= learning_rate * db1

    # Fonction d'entraînement du modèle
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)

            # Affichage de la précision
            if epoch % 10 == 0:
                accuracy = self.compute_accuracy(X, y)
                print(f"Epoch {epoch}/{epochs}, Accuracy: {accuracy:.4f}")

            # Rétropropagation
            self.backward(X, y, output, learning_rate)

    # Prédiction
    def predict(self, X):
        output = self.forward(X)
        return output

    # Calcul de la précision
    def compute_accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
