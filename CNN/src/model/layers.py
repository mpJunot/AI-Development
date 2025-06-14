import numpy as np
from typing import Tuple, List

class Layer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def update(self, learning_rate: float) -> None:
        pass

class Conv2D(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, weight_init=None):
        """
        Convolutional layer.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride of the convolution.
        :param padding: Padding added to the input.
        :param weight_init: Function for weight initialization.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and biases
        if weight_init:
            self.weights = weight_init((out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.biases = np.zeros(out_channels)

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, in_channels, in_height, in_width = x.shape
        out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        out = np.zeros((batch_size, self.out_channels, out_height, out_width))

        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for i in range(out_height):
            for j in range(out_width):
                x_slice = x_padded[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                out[:, :, i, j] = np.tensordot(x_slice, self.weights, axes=([1, 2, 3], [1, 2, 3])) + self.biases

        self.cache = (x, x_padded)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x, x_padded = self.cache
        batch_size, in_channels, in_height, in_width = x.shape
        out_height, out_width = grad.shape[2], grad.shape[3]

        grad_x_padded = np.zeros_like(x_padded)
        grad_weights = np.zeros_like(self.weights)
        grad_biases = np.sum(grad, axis=(0, 2, 3))

        for i in range(out_height):
            for j in range(out_width):
                x_slice = x_padded[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                grad_weights += np.tensordot(grad[:, :, i, j], x_slice, axes=[0, 0])
                grad_x_padded[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += np.tensordot(grad[:, :, i, j], self.weights, axes=[1, 0])

        grad_x = grad_x_padded[:, :, self.padding:in_height+self.padding, self.padding:in_width+self.padding]

        self.grad_weights = grad_weights
        self.grad_biases = grad_biases
        return grad_x

    def update(self, optimizer, **kwargs) -> None:
        """
        Update weights and biases using the optimizer.
        :param optimizer: Optimizer object (e.g., SGD, Adam).
        """
        optimizer.update({"weights": self.weights, "biases": self.biases},
                         {"weights": self.grad_weights, "biases": self.grad_biases})

class MaxPool2D(Layer):
    def __init__(self, pool_size: int, stride: int):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, in_channels, in_height, in_width = x.shape
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1
        out = np.zeros((batch_size, in_channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                x_slice = x[:, :, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size]
                out[:, :, i, j] = np.max(x_slice, axis=(2, 3))

        self.cache = x
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self.cache
        batch_size, in_channels, in_height, in_width = x.shape
        out_height, out_width = grad.shape[2], grad.shape[3]

        grad_x = np.zeros_like(x)

        for i in range(out_height):
            for j in range(out_width):
                x_slice = x[:, :, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size]
                max_x_slice = np.max(x_slice, axis=(2, 3), keepdims=True)
                mask = (x_slice == max_x_slice)
                grad_x[:, :, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size] += grad[:, :, i, j][:, :, None, None] * mask

        return grad_x

class Flatten(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(self.cache)

class Dense(Layer):
    def __init__(self, input_dim: int, output_dim: int, weight_init=None):
        if weight_init:
            self.weights = weight_init((input_dim, output_dim))
        else:
            self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self.cache
        grad_x = np.dot(grad, self.weights.T)
        self.grad_weights = np.dot(x.T, grad)
        self.grad_biases = np.sum(grad, axis=0)
        return grad_x

    def update(self, optimizer, **kwargs) -> None:
        optimizer.update({"weights": self.weights, "biases": self.biases},
                         {"weights": self.grad_weights, "biases": self.grad_biases})

class BatchNorm2D(Layer):
    def __init__(self, num_features: int, momentum: float = 0.9, epsilon: float = 1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        original_shape = x.shape
        if x.ndim == 2:
            x = x[:, :, None, None]

        if training:
            batch_mean = np.mean(x, axis=(0, 2, 3))
            batch_var = np.var(x, axis=(0, 2, 3))
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            self.cache = (x, batch_mean, batch_var)
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        x_normalized = (x - batch_mean[None, :, None, None]) / np.sqrt(batch_var[None, :, None, None] + self.epsilon)
        out = self.gamma[None, :, None, None] * x_normalized + self.beta[None, :, None, None]

        if original_shape != x.shape:
            out = out.reshape(original_shape)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        original_shape = grad.shape
        if grad.ndim == 2:
            grad = grad[:, :, None, None]

        x, batch_mean, batch_var = self.cache
        batch_size, num_features, height, width = x.shape

        x_normalized = (x - batch_mean[None, :, None, None]) / np.sqrt(batch_var[None, :, None, None] + self.epsilon)
        grad_gamma = np.sum(grad * x_normalized, axis=(0, 2, 3))
        grad_beta = np.sum(grad, axis=(0, 2, 3))

        grad_x_normalized = grad * self.gamma[None, :, None, None]
        grad_var = np.sum(grad_x_normalized * (x - batch_mean[None, :, None, None]) * -0.5 * (batch_var[None, :, None, None] + self.epsilon)**-1.5, axis=(0, 2, 3))
        grad_mean = np.sum(grad_x_normalized * -1 / np.sqrt(batch_var[None, :, None, None] + self.epsilon), axis=(0, 2, 3)) + grad_var * np.sum(-2 * (x - batch_mean[None, :, None, None]), axis=(0, 2, 3)) / batch_size

        grad_x = grad_x_normalized / np.sqrt(batch_var[None, :, None, None] + self.epsilon) + grad_var[None, :, None, None] * 2 * (x - batch_mean[None, :, None, None]) / batch_size + grad_mean[None, :, None, None] / batch_size

        if original_shape != grad.shape:
            grad_x = grad_x.reshape(original_shape)

        self.grad_gamma = grad_gamma
        self.grad_beta = grad_beta
        return grad_x

    def update(self, optimizer, **kwargs) -> None:
        optimizer.update({"gamma": self.gamma, "beta": self.beta},
                         {"gamma": self.grad_gamma, "beta": self.grad_beta})

class Dropout(Layer):
    def __init__(self, p: float):
        self.p = p

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.mask = np.random.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p)
            return x * self.mask
        else:
            return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask

class ReLU(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return np.maximum(0, x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self.cache
        grad_x = grad * (x > 0)
        return grad_x

class LeakyReLU(Layer):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self.cache
        grad_x = grad * np.where(x > 0, 1, self.alpha)
        return grad_x

class Softmax(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad
