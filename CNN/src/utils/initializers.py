import numpy as np

def xavier_init(shape: tuple) -> np.ndarray:
    """Xavier initialization for weights."""
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)

def he_init(shape: tuple) -> np.ndarray:
    """He initialization for weights."""
    fan_in = shape[0]
    std = np.sqrt(2 / fan_in)
    return np.random.randn(*shape) * std

def zeros_init(shape: tuple) -> np.ndarray:
    """Zero initialization for weights."""
    return np.zeros(shape)

def ones_init(shape: tuple) -> np.ndarray:
    """One initialization for weights."""
    return np.ones(shape)
