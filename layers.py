import numpy as np
from abc import ABC, abstractmethod
from activation_functions import relu, leaky_relu, sigmoid, softmax, tanh


class Layer(ABC):

    def __init__(self):
        self._tensor = None

    @abstractmethod
    def forward(self, tensor: np.array) -> np.array:
        self._tensor = tensor.copy()
        return tensor

    @abstractmethod
    def derivative(self, tensor: np.array) -> np.array:
        return tensor

    @abstractmethod
    def backward(self, grad_input_tensor: np.array) -> np.array:
        return grad_input_tensor @ self.derivative(self._tensor)


class Identity(Layer):

    def __init__(self):
        self._tensor = None

    def forward(self, tensor: np.array) -> np.array:
        self._tensor = tensor.copy()
        return tensor

    def derivative(self, tensor: np.array) -> np.array:
        return np.zeros(tensor.shape)

    def backward(self, grad_input_tensor: np.array) -> np.array:
        return grad_input_tensor @ self.derivative(self._tensor)


class ReLU(Layer):

    def __init__(self):
        self._tensor = None

    def forward(self, tensor: np.array) -> np.array:
        self._tensor = tensor.copy()
        return relu(tensor)

    def derivative(self, tensor: np.array) -> np.array:
        return np.greater(tensor, 0)

    def backward(self, grad_input_tensor: np.array) -> np.array:
        return grad_input_tensor @ self.derivative(self._tensor)


class LeakyReLU(Layer):

    def __init__(self):
        self._tensor = None

    def forward(self, tensor: np.array) -> np.array:
        self._tensor = tensor.copy()
        return leaky_relu(tensor)

    def derivative(self, tensor: np.array) -> np.array:
        return ...

    def backward(self, tensor: np.array):
        return ...


class Sigmoid(Layer):

    def __init__(self):
        self._tensor = None

    def forward(self, tensor: np.array) -> np.array:
        self._tensor = tensor.copy()
        return sigmoid(tensor)

    def derivative(self, tensor):
        return sigmoid(tensor) * (1 - sigmoid(tensor))

    def backward(self, grad_input_tensor: np.array) -> np.array:
        return grad_input_tensor @ self.derivative(self._tensor)


class Softmax(Layer):

    def __init__(self):
        self._tensor = None

    def forward(self, tensor: np.array) -> np.array:
        self._tensor = tensor.copy()
        return softmax(tensor)

    def derivative(self, tensor: np.array) -> np.array:
        ...

    def backward(self, grad_input_tensor: np.array) -> np.array:
        return grad_input_tensor @ self.derivative(self._tensor)


class Tanh(Layer):

    def __init__(self):
        self._tensor = None

    def forward(self, tensor: np.array) -> np.array:
        self._tensor = tensor.copy()
        return tanh(tensor)

    def derivative(self, tensor: np.array) -> np.array:
        return 1 - (tanh(tensor) * tanh(tensor))

    def backward(self, grad_input_tensor: np.array) -> np.array:
        return grad_input_tensor @ self.derivative(self._tensor)
