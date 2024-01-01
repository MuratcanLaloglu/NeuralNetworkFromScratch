from network import Module
import numpy as np


class ReLu:
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x, grad, lr, prev_hidden):
        return np.multiply(grad, np.heaviside(prev_hidden, 0))
