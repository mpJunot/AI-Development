import numpy as np
from typing import Tuple
from model.layers import Conv2D, Dense, Flatten, BatchNorm2D, ReLU, LeakyReLU, MaxPool2D, Dropout, Softmax
from utils.initializers import xavier_init, he_init
from optimizers.regularization import l1_regularization, l2_regularization, elastic_net_regularization

class LeNet5:
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, activation: str = "relu", weight_init: str = "xavier"):
        """
        Initialize LeNet5 model.
        :param input_shape: Tuple of (channels, height, width)
        :param num_classes: Number of output classes
        :param activation: Activation function type ("relu" or "leaky_relu")
        :param weight_init: Weight initialization method ("xavier" or "he")
        """
        if not isinstance(input_shape, tuple) or len(input_shape) != 3:
            raise ValueError("input_shape must be a tuple of (channels, height, width)")

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation = activation
        self.weight_init = weight_init

        C, H, W = input_shape
        self.layers = [
            Conv2D(in_channels=C, out_channels=6, kernel_size=5, stride=1, padding=2, weight_init=self._get_initializer()),
            BatchNorm2D(num_features=6),
            self._get_activation(),
            MaxPool2D(pool_size=2, stride=2),
            Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, weight_init=self._get_initializer()),
            BatchNorm2D(num_features=16),
            self._get_activation(),
            MaxPool2D(pool_size=2, stride=2),
            Flatten(),
            Dense(input_dim=16*5*5, output_dim=120, weight_init=self._get_initializer()),
            BatchNorm2D(num_features=120),
            self._get_activation(),
            Dropout(p=0.5),
            Dense(input_dim=120, output_dim=84, weight_init=self._get_initializer()),
            BatchNorm2D(num_features=84),
            self._get_activation(),
            Dropout(p=0.5),
            Dense(input_dim=84, output_dim=num_classes, weight_init=self._get_initializer()),
            Softmax()
        ]

    def _get_activation(self):
        if self.activation == "relu":
            return ReLU()
        elif self.activation == "leaky_relu":
            return LeakyReLU(alpha=0.01)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _get_initializer(self):
        if self.weight_init == "xavier":
            return xavier_init
        elif self.weight_init == "he":
            return he_init
        else:
            raise ValueError(f"Unsupported weight initialization: {self.weight_init}")

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            if isinstance(layer, (BatchNorm2D, Dropout)):
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, optimizer, **kwargs) -> None:
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update(optimizer, **kwargs)

    def compute_regularization_loss(self, reg_type: str, lambda_l1: float = 0.0, lambda_l2: float = 0.0) -> float:
        weights = {f"layer_{i}": layer.weights for i, layer in enumerate(self.layers) if hasattr(layer, 'weights')}
        if reg_type == "l1":
            return l1_regularization(weights, lambda_l1)
        elif reg_type == "l2":
            return l2_regularization(weights, lambda_l2)
        elif reg_type == "elastic_net":
            return elastic_net_regularization(weights, lambda_l1, lambda_l2)
        return 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x, training=False), axis=1)

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        return np.sum(log_likelihood) / m

    def accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.argmax(y_pred, axis=1) == y_true)
