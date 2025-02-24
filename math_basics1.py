import numpy as np
print("Python list operations:")
a = [1,2,3]
b = [4,5,6]
print("a+b:", a+b)
try:
    print(a*b)
except TypeError:
    print("a*b has no meaning for Python lists")
print()
print("numpy array operations:")
a = np.array([1,2,3])
b = np.array([4,5,6])
print("a+b:", a+b)
print("a*b:", a*b)
print('a:')
print(a)
print('a.sum(axis=0):', a.sum(axis=0))

import numpy as np
from numpy import ndarray
from typing import Callable

class NeuralNetwork:
    def __init__(self,
                layers,
                loss,
                learning_rate: float = 0.01) -> None:
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate

    def square(self, x: ndarray) -> ndarray:
        """Computes the square of input array"""
        return x * x

   

    
def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          delta: float = 0.001) -> ndarray:
    '''
    Evaluates the derivative of a function "func" at every element in the "input_" aray.
    '''
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)
def f(input_: ndarray) -> ndarray:
    """
    Transform input array using some mathematical operation
    Args:
        input_: input array
    Returns:
        transformed array
    """
    # Example transformation (you can modify this)
    return np.square(input_)

input_array = np.array([1.0, 2.0, 3.0])
result = f(input_array)
derivative = deriv(f, input_array)

print("Input:", input_array)
print("f(x):", result)
print("f'(x):", derivative)