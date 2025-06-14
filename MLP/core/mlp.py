import numpy as np
from layer import Layer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class MLP:
    def __init__(
        self, layer_sizes, activation_function="relu", learning_rate=0.1, verbose=False,
        optimizer="sgd", weight_init="normal", weight_scale=1.0, task="classification"
    ):
        self.layers = []
        self.learning_rate = learning_rate
        self.activation_name = activation_function
        self.activation, self.activation_derivative = self.get_activation_function(activation_function)
        self.verbose = verbose
        self.training_losses = []
        self.validation_losses = []
        self.gradient_clip_threshold = 1.0
        self.optimizer = optimizer
        self.weight_init = weight_init
        self.weight_scale = weight_scale
        self.task = task

        for i in range(1, len(layer_sizes)):
            input_size = layer_sizes[i - 1]
            output_size = layer_sizes[i]
            scale = self.weight_scale
            self.layers.append(Layer(
                output_size, input_size, self.activation,
                weight_scale=scale, weight_init=self.weight_init
            ))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]

    def forward(self, X):
        activations = [X]
        zs = []
        a = X
        for i, layer in enumerate(self.layers):
            z, a = layer.forward(a)
            zs.append(z)
            if self.task == "classification" and i == len(self.layers) - 1:
                a = self.softmax(z)
            activations.append(a)
        return zs, activations

    def backward(self, zs, activations, Y, l2_lambda=0.0, l1_lambda=0.0, elastic_alpha=0.0):
        m = Y.shape[0]
        grads = []
        if self.task == "classification":
            delta = activations[-1] - Y
        else:
            delta = activations[-1] - Y
        for l in reversed(range(len(self.layers))):
            z = zs[l]
            if self.task == "classification" and l == len(self.layers) - 1:
                dZ = delta
            else:
                dZ = delta * self.activation_derivative(activations[l+1])
            a_prev = activations[l]
            dW = np.dot(dZ.T, a_prev) / m
            dB = np.sum(dZ.T, axis=1, keepdims=True) / m
            l1_penalty = l1_lambda * np.sign(self.layers[l].weights)
            l2_penalty = l2_lambda * self.layers[l].weights
            elastic_penalty = elastic_alpha * l1_penalty + (1 - elastic_alpha) * l2_penalty
            dW += elastic_penalty
            grads.insert(0, (dW, dB))
            if l > 0:
                delta = np.dot(dZ, self.layers[l].weights)
        return grads

    def update_params(self, grads):
        for layer, (dW, dB) in zip(self.layers, grads):
            layer.update(dW, dB, self.learning_rate)

    def train(
        self, training_data, labels, epochs, l2_lambda=0.0, l1_lambda=0.0, elastic_alpha=0.0,
        validation_data=None, validation_labels=None, early_stopping=False, patience=5, min_delta=1e-4,
        batch_size=32, debug_small_set=False, learning_rate=None
    ):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        print(f"Training MLP with {len(self.layers)} layers, learning rate: {self.learning_rate}, optimizer: {self.optimizer}")
        self.training_losses = []
        best_val_loss = float("inf")
        patience_counter = 0

        normalized_data = self.normalize_data(training_data)
        n_samples = normalized_data.shape[0]

        if debug_small_set:
            normalized_data = normalized_data[:100]
            labels = labels[:100]
            n_samples = 100

        for epoch in range(epochs):
            total_loss = 0.0
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for start in tqdm(range(0, n_samples, batch_size), desc=f"Epoch {epoch+1}"):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                X_batch = normalized_data[batch_idx]
                Y_batch = labels[batch_idx]
                zs, activations = self.forward(X_batch)
                outputs = activations[-1]
                if self.task == "classification":
                    loss = self.cross_entropy_loss(outputs, Y_batch)
                else:
                    loss = np.mean((outputs - Y_batch) ** 2)
                total_loss += loss * X_batch.shape[0]
                grads = self.backward(zs, activations, Y_batch, l2_lambda, l1_lambda, elastic_alpha)
                self.update_params(grads)
            avg_training_loss = total_loss / n_samples
            self.training_losses.append(avg_training_loss)
            print(f"Epoch {epoch + 1}, Training Loss: {avg_training_loss}", flush=True)

            if early_stopping and validation_data is not None and validation_labels is not None:
                val_loss = self.evaluate(validation_data, validation_labels)
                self.validation_losses.append(val_loss)
                if val_loss + min_delta < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return avg_training_loss

    def plot_training_history(self):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.training_losses) + 1)
        plt.plot(epochs, self.training_losses, 'b-', label='Training Loss')
        if self.validation_losses:
            plt.plot(epochs, self.validation_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')

    def plot_confusion_matrix(self, test_data, test_labels):
        matrix = self.confusion_matrix(test_data, test_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png')

    def plot_predictions(self, X, y_true, filename="pred_vs_true.png"):
        y_pred = self.predict(self.normalize_data(X))
        if self.task == "classification":
            y_true_labels = np.argmax(y_true, axis=1)
            y_pred_labels = np.argmax(y_pred, axis=1)
            plt.figure(figsize=(8, 6))
            plt.scatter(range(len(y_true_labels)), y_true_labels, label="True", alpha=0.6)
            plt.scatter(range(len(y_pred_labels)), y_pred_labels, label="Predicted", alpha=0.6)
            plt.xlabel("Sample")
            plt.ylabel("Class")
            plt.legend()
            plt.title("Predicted vs True Classes")
        else:
            plt.figure(figsize=(8, 6))
            plt.scatter(y_true, y_pred, alpha=0.6)
            plt.xlabel("True Value")
            plt.ylabel("Predicted Value")
            plt.title("Predicted vs True Values")
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def evaluate(self, validation_data, validation_labels):
        total_loss = 0.0
        normalized_data = self.normalize_data(validation_data)
        for start in range(0, normalized_data.shape[0], 32):
            end = min(start + 32, normalized_data.shape[0])
            X_batch = normalized_data[start:end]
            Y_batch = validation_labels[start:end]
            _, activations = self.forward(X_batch)
            outputs = activations[-1]
            if self.task == "classification":
                total_loss += self.cross_entropy_loss(outputs, Y_batch) * X_batch.shape[0]
            else:
                total_loss += np.sum((outputs - Y_batch) ** 2)
        return total_loss / normalized_data.shape[0]

    def save_model(self, filename):
        weights = [layer.weights for layer in self.layers]
        biases = [layer.bias for layer in self.layers]
        np.savez(filename, weights=np.array(weights, dtype=object), biases=np.array(biases, dtype=object))

    def load_model(self, filename):
        data = np.load(filename, allow_pickle=True)
        weights = data['weights']
        biases = data['biases']
        for layer, w, b in zip(self.layers, weights, biases):
            layer.weights = w
            layer.bias = b

    def predict(self, inputs):
        _, activations = self.forward(inputs)
        return activations[-1]

    def confusion_matrix(self, test_data, test_labels):
        num_classes = test_labels.shape[1]
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        normalized_data = self.normalize_data(test_data)
        for start in range(0, normalized_data.shape[0], 32):
            end = min(start + 32, normalized_data.shape[0])
            X_batch = normalized_data[start:end]
            Y_batch = test_labels[start:end]
            outputs = self.predict(X_batch)
            preds = np.argmax(outputs, axis=1)
            actuals = np.argmax(Y_batch, axis=1)
            for a, p in zip(actuals, preds):
                matrix[a][p] += 1
        print("Confusion Matrix:")
        print(matrix)
        return matrix

    def get_activation_function(self, name):
        if name == "sigmoid":
            return (lambda x: 1 / (1 + np.exp(-x)), lambda x: x * (1 - x))
        elif name == "tanh":
            return (np.tanh, lambda x: 1 - x ** 2)
        elif name == "relu":
            return (lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float))
        else:
            raise ValueError("Unknown activation function")

    def normalize_data(self, data):
        return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
