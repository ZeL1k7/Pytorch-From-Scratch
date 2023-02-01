import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):

    def __init__(self):
        self._tensor = None

    @abstractmethod
    def forward(self, tensor: np.array) -> np.array:
        self._tensor = np.copy(tensor)
        return tensor

    @abstractmethod
    def backward(self, grad_input_tensor: np.array) -> np.array:
        return self._tensor @ grad_input_tensor


class Identity(Layer):

    def __init__(self):
        self._tensor = None

    def forward(self, tensor: np.array) -> np.array:
        self._tensor = np.copy(tensor)
        return tensor

    def backward(self, grad_input_tensor: np.array) -> np.array:
        return np.zeros(self._tensor.shape) @ grad_input_tensor


class Linear(Layer):

    def __init__(self, in_channels: int, out_channels: int):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._tensor = None
        self._weights = np.random.normal(low=-np.sqrt(1/in_channels),
                                         high=np.sqrt(1/in_channels),
                                         size=(in_channels, out_channels))

    def forward(self, tensor: np.array) -> np.array:
        self._tensor = tensor
        return tensor @ self._weights.T

    def backward(self, grad_input_tensor: np.array) -> np.array:
        return self._tensor.T  @ grad_input_tensor
