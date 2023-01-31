import numpy as np


def relu(tensor: np.array) -> np.array:
    return np.max(0, tensor)


def leaky_relu(tensor: np.array, alpha: int = 0.01) -> np.array:
    return np.max(0, tensor) + alpha * min(0, tensor)


def sigmoid(tensor: np.array) -> np.array:
    return 1/(1 + np.exp(-tensor))


def softmax(tensor: np.array, dim: int = 1) -> np.array:
    return np.exp(tensor) / np.sum(np.exp(tensor), axis=dim, keepdims=True)


def tanh(tensor: np.array) -> np.array:
    return np.tanh(tensor)
