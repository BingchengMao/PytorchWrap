import copy

import numpy as np


class TorchWeights:
    """
    Torch weights class
    """

    def __init__(self, weights):
        self.w = copy.deepcopy(weights)

    def __add__(self, other):
        """
        magic function for operator '+'
        Operator Overloading  W1 + W2 or W1 + c
        c is a constant
        """
        weights = dict()
        if type(other) == int or type(other) == float:  # if there is a constant.
            for name, value in self.w.items():
                weights[name] = other + value
        else:
            for name, value in self.w.items():
                weights[name] = other.w[name] + value
        return TorchWeights(weights)

    def __radd__(self, other):
        """
        magic function for operator '+'
        Operator Overloading  W1 + W2 or c + W1
        c is a constant
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        magic function for operator '-'
        Operator Overloading  W1 - W2 or W1 - c
        c is a constant
        """
        weights = {}
        if type(other) == int or type(other) == float:  # if there is a constant.
            for name, value in self.w.items():
                weights[name] = value - other
        else:
            for name, value in self.w.items():
                weights[name] = value - other.w[name]
        return TorchWeights(weights)

    def __rsub__(self, other):
        """
        magic function for operator '-'
        Operator Overloading  W1 - W2 or c - W1
        c is a constant
        """
        weights = {}
        if type(other) == int or type(other) == float:  # if there is a constant.
            for name, value in self.w.items():
                weights[name] = other - value
        else:
            for name, value in self.w.items():
                weights[name] = other.w[name] - value
        return TorchWeights(weights)

    def __mul__(self, other):
        """
        magic function for operator '*'
        Operator Overloading  W1 * W2 (element wise) or W1 * c
        c is a constant
        """
        weights = {}
        if type(other) == int or type(other) == float:  # if there is a constant.
            for name, value in self.w.items():
                weights[name] = other * value
        else:
            for name, value in self.w.items():
                weights[name] = other.w[name] * value
        return TorchWeights(weights)

    def __rmul__(self, other):
        """
        magic function for operator '*'
        Operator Overloading  W1 * W2 (element wise) or c * W1
        c is a constant
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        magic function for operator '/'
        Operator Overloading  W1 / W2 (element wise) or W1 / c
        c is a constant
        """
        weights = {}
        if type(other) == int or type(other) == float:  # if there is a constant.
            for name, value in self.w.items():
                weights[name] = value / other
        else:
            for name, value in self.w.items():
                weights[name] = value / other.w[name]
        return TorchWeights(weights)

    def __rtruediv__(self, other):
        """
        magic function for operator '/'
        Operator Overloading  W1 / W2 (element wise) or c / W1
        c is a constant
        """
        weights = {}
        if type(other) == int or type(other) == float:  # if there is a constant.
            for name, value in self.w.items():
                weights[name] = other / value
        else:
            for name, value in self.w.items():
                weights[name] = other.w[name] / value
        return TorchWeights(weights)

    def __pow__(self, other):
        """
        magic function for operator '**'
        Operator Overloading  W1 ** W2 (element wise) or W1 ** c
        c is a constant
        """
        weights = {}
        if type(other) == int or type(other) == float:  # if there is a constant.
            for name, value in self.w.items():
                weights[name] = value ** other
        else:
            for name, value in self.w.items():
                weights[name] = value ** other.w[name]
        return TorchWeights(weights)

    def __rpow__(self, other):
        """
        magic function for operator '**'
        Operator Overloading  W1 ** W2 (element wise) or c ** W1
        c is a constant
        """
        weights = {}
        if type(other) == int or type(other) == float:  # if there is a constant.
            for name, value in self.w.items():
                weights[name] = other ** value
        else:
            for name, value in self.w.items():
                weights[name] = other.w[name] ** value
        return TorchWeights(weights)

    def __pos__(self):
        return TorchWeights(self.w)

    def __neg__(self):
        weights = {}
        for name, value in self.w.items():
            weights[name] = -value
        return TorchWeights(weights)

    def __abs__(self):
        weights = {}
        for name, value in self.w.items():
            weights[name] = abs(value)
        return TorchWeights(weights)

    def __contains__(self, item):
        return item in self.w

    def __getitem__(self, key):
        return self.w[key]

    def items(self):
        return self.w.items()

    def sign(self):
        res = dict()
        for name, val in self.w.items():
            res[name] = np.sign(val)
        return TorchWeights(res)

    def __setitem__(self, key, weight):
        self.w[key] = weight

    def __len__(self):
        if isinstance(self.w, dict):
            return len(self.w)
        else:
            return 0

    def __str__(self):
        return str(self.w)

    def __repr__(self):
        return str(self.w)
