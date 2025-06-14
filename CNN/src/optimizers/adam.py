import numpy as np

class Adam:
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params: dict, grads: dict):
        """
        Update parameters using Adam optimization.
        :param params: Dictionary of parameters to update (e.g., weights, biases).
        :param grads: Dictionary of gradients corresponding to the parameters.
        """
        self.t += 1

        for key in params.keys():
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            if self.m[key].shape != grads[key].shape:
                self.m[key] = np.zeros_like(grads[key])
                self.v[key] = np.zeros_like(grads[key])

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]

            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # Compute bias-corrected moments
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def reset(self):
        """Reset the optimizer's state."""
        self.m = {}
        self.v = {}
        self.t = 0
