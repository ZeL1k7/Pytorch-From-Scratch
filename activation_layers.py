import numpy as np
from abc import ABC, abstractmethod
from activation_functions import relu, leaky_relu, sigmoid, softmax, tanh


class ActivationLayer(ABC):

    def __init__(self):
        self._tensor = None

    @abstractmethod
    def forward(self, tensor: np.array) -> np.array:
        self._tensor = np.copy(tensor)
        return tensor

    @abstractmethod
    def derivative(self, tensor: np.array) -> np.array:
        return tensor

    @abstractmethod
    def backward(self, grad_input_tensor: np.array) -> np.array:
        return self.derivative(self._tensor) @ grad_input_tensor


class ReLU(ActivationLayer):

    def __init__(self):
        self._tensor = None

    def forward(self, tensor: np.array) -> np.array:
        self._tensor = np.copy(tensor)
        return relu(tensor)

    def derivative(self, tensor: np.array) -> np.array:
        return np.greater(tensor, 0)

    def backward(self, grad_input_tensor: np.array) -> np.array:
        return self.derivative(self._tensor) @ grad_input_tensor


class LeakyReLU(ActivationLayer):

    def __init__(self):
        self._tensor = None

    def forward(self, tensor: np.array) -> np.array:
        self._tensor = np.copy(tensor)
        return leaky_relu(tensor)

    def derivative(self, tensor: np.array) -> np.array:
        return ...

    def backward(self, tensor: np.array):
        return ...


class Sigmoid(ActivationLayer):

    def __init__(self):
        self._tensor = None

    def forward(self, tensor: np.array) -> np.array:
        self._tensor = np.copy(tensor)
        return sigmoid(tensor)

    def derivative(self, tensor):
        return sigmoid(tensor) * (1 - sigmoid(tensor))

    def backward(self, grad_input_tensor: np.array) -> np.array:
        return self.derivative(self._tensor) @ grad_input_tensor


class Softmax(ActivationLayer):

    def __init__(self):
        self._tensor = None

    def forward(self, tensor: np.array) -> np.array:
        self._tensor = np.copy(tensor)
        return softmax(tensor)

    def derivative(self, tensor: np.array) -> np.array:
        ...

    def backward(self, grad_input_tensor: np.array) -> np.array:
        return self.derivative(self._tensor) @ grad_input_tensor


class Tanh(ActivationLayer):

    def __init__(self):
        self._tensor = None

    def forward(self, tensor: np.array) -> np.array:
        self._tensor = np.copy(tensor)
        return tanh(tensor)

    def derivative(self, tensor: np.array) -> np.array:
        return 1 - (tanh(tensor) * tanh(tensor))

    def backward(self, grad_input_tensor: np.array) -> np.array:
        return self.derivative(self._tensor) @ grad_input_tensor
