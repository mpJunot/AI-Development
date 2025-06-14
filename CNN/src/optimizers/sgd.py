import numpy as np

class SGD:
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Stochastic Gradient Descent optimizer with optional momentum.
        :param learning_rate: Learning rate for the optimizer.
        :param momentum: Momentum factor.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def update(self, params: dict, grads: dict):
        for key in params.keys():
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(grads[key])
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            params[key] += self.velocity[key]
