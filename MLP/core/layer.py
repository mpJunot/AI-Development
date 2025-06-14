import numpy as np

class Layer:
    def __init__(self, num_nodes, input_size, activation, weight_scale=1.0, weight_init="normal"):
        if weight_init == "normal":
            self.weights = np.random.normal(0, weight_scale, (num_nodes, input_size))
            self.bias = np.random.normal(0, weight_scale, (num_nodes, 1))
        elif weight_init == "uniform":
            self.weights = np.random.uniform(-weight_scale, weight_scale, (num_nodes, input_size))
            self.bias = np.random.uniform(-weight_scale, weight_scale, (num_nodes, 1))
        elif weight_init == "xavier":
            limit = np.sqrt(6 / (input_size + num_nodes))
            self.weights = np.random.uniform(-limit, limit, (num_nodes, input_size))
            self.bias = np.zeros((num_nodes, 1))
        elif weight_init == "he":
            self.weights = np.random.randn(num_nodes, input_size) * np.sqrt(2. / input_size)
            self.bias = np.zeros((num_nodes, 1))
        else:
            raise ValueError("Unknown weight_init")
        self.activation = activation

    def forward(self, inputs):
        z = np.dot(inputs, self.weights.T) + self.bias.T
        a = self.activation(z)
        return z, a

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, dW, dB, lr):
        self.weights -= lr * dW
        self.bias -= lr * dB
