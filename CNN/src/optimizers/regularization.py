import numpy as np

def l1_regularization(weights: dict, lambda_l1: float) -> float:
    """Compute L1 regularization loss."""
    return lambda_l1 * sum(np.sum(np.abs(w)) for w in weights.values())

def l2_regularization(weights: dict, lambda_l2: float) -> float:
    return lambda_l2 * sum(np.sum(w ** 2) for w in weights.values())

def elastic_net_regularization(weights: dict, lambda_l1: float, lambda_l2: float) -> float:
    """Compute ElasticNet regularization loss."""
    return l1_regularization(weights, lambda_l1) + l2_regularization(weights, lambda_l2)
