import numpy as np
from abc import ABC, abstractmethod
from activation_functions import relu, leaky_relu, sigmoid, softmax, tanh


class Layer(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def forward(self):
        ...

    @abstractmethod
    def derivative(self):
        ...

    @abstractmethod
    def backward(self):
        ...


class Identity(Layer):

    def forward(self, tensor: np.array) -> np.array:
        return tensor

    def derivative(self, tensor: np.array) -> np.array:
        return np.zeros(tensor.shape)

    def backward(self) -> np.array:
        ...


class ReLU(Layer):
    def forward(self, tensor: np.array) -> np.array:
        return relu(tensor)

    def derivative(self, tensor: np.array) -> np.array:
        return np.greater(tensor, 0)

    def backward(self) -> np.array:
        ...


class LeakyReLU(Layer):
    def forward(self, tensor: np.array) -> np.array:
        return leaky_relu(tensor)

    def derivative(self, tensor: np.array) -> np.array:
        return np.greater(tensor, 0)

    def backward(self):
        ...


class Sigmoid(Layer):
    def forward(self, tensor: np.array) -> np.array:
        return sigmoid(tensor)

    def derivative(self, tensor):
        return sigmoid(tensor) * (1 - sigmoid(tensor))

    def backward(self) -> np.array:
        ...


class Softmax(Layer):
    def forward(self, tensor: np.array) -> np.array:
        return softmax(tensor)

    def derivative(self, tensor: np.array) -> np.array:
        return ...

    def backward(self) -> np.array:
        ...


class Tanh(Layer):
    def forward(self, tensor: np.array) -> np.array:
        return tanh(tensor)

    def derivative(self, tensor: np.array) -> np.array:
        return 1 - (tanh(tensor) * tanh(tensor))

    def backward(self) -> np.array:
        ...
