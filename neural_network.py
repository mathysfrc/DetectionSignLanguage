import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.weights1 = np.random.randn(42, 64) * 0.01
        self.bias1 = np.zeros((1, 64))
        self.weights2 = np.random.randn(64, 32) * 0.01
        self.bias2 = np.zeros((1, 32))
        self.weights3 = np.random.randn(32, 5) * 0.01
        self.bias3 = np.zeros((1, 5))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.a3 = self.softmax(self.z3)
        return self.a3

    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]
        y_onehot = np.zeros((m, 5))
        y_onehot[np.arange(m), y] = 1

        dz3 = output - y_onehot
        dw3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        da2 = np.dot(dz3, self.weights3.T)
        dz2 = da2 * self.relu_derivative(self.z2)
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = np.dot(dz2, self.weights2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.weights3 -= learning_rate * dw3
        self.bias3 -= learning_rate * db3
        self.weights2 -= learning_rate * dw2
        self.bias2 -= learning_rate * db2
        self.weights1 -= learning_rate * dw1
        self.bias1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

            m = X.shape[0]
            loss = -np.sum(np.log(output[np.arange(m), y] + 1e-15)) / m
            accuracy = np.mean(np.argmax(output, axis=1) == y)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1) + 1  # Retourne classe 1-5