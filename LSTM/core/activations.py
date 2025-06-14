import numpy as np

class Activation:
    def __init__(self):
        """Base activation class"""
        self.input = None
        self.output = None

    def forward(self, x):
        """Forward pass"""
        raise NotImplementedError

    def backward(self, grad):
        """Backward pass"""
        raise NotImplementedError

class Sigmoid(Activation):
    def forward(self, x):
        """Forward pass of sigmoid activation"""
        self.input = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad):
        """Backward pass of sigmoid activation"""
        return grad * self.output * (1 - self.output)

class Tanh(Activation):
    def forward(self, x):
        """Forward pass of tanh activation"""
        self.input = x
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad):
        """Backward pass of tanh activation"""
        return grad * (1 - self.output ** 2)

class ReLU(Activation):
    def forward(self, x):
        """Forward pass of ReLU activation"""
        self.input = x
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, grad):
        """Backward pass of ReLU activation"""
        return grad * (self.input > 0)

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        """Leaky ReLU activation"""
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        """Forward pass of Leaky ReLU activation"""
        self.input = x
        self.output = np.where(x > 0, x, self.alpha * x)
        return self.output

    def backward(self, grad):
        """Backward pass of Leaky ReLU activation"""
        return grad * np.where(self.input > 0, 1, self.alpha)

class ELU(Activation):
    def __init__(self, alpha=1.0):
        """ELU activation"""
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        """Forward pass of ELU activation"""
        self.input = x
        self.output = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        return self.output

    def backward(self, grad):
        """Backward pass of ELU activation"""
        return grad * np.where(self.input > 0, 1, self.alpha * np.exp(self.input))

class Softmax(Activation):
    def forward(self, x):
        """Forward pass of softmax activation"""
        self.input = x
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.output

    def backward(self, grad):
        """Backward pass of softmax activation"""
        batch_size = self.output.shape[0]
        jacobian = np.zeros((batch_size, self.output.shape[1], self.output.shape[1]))

        for i in range(batch_size):
            for j in range(self.output.shape[1]):
                for k in range(self.output.shape[1]):
                    if j == k:
                        jacobian[i, j, k] = self.output[i, j] * (1 - self.output[i, j])
                    else:
                        jacobian[i, j, k] = -self.output[i, j] * self.output[i, k]

        return np.einsum('ijk,ik->ij', jacobian, grad)

class GELU(Activation):
    def forward(self, x):
        """Forward pass of GELU activation"""
        self.input = x
        self.output = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        return self.output

    def backward(self, grad):
        """Backward pass of GELU activation"""
        x = self.input
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        pdf = 0.5 * np.sqrt(2 / np.pi) * (1 + 0.134145 * x**2) * (1 / np.cosh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))**2
        return grad * (cdf + x * pdf)
