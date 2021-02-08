"""
Contains functions for numerical optimisation analysis
"""

from typing import Union
import numpy as np
from numpy import ndarray


def f1(x: ndarray, a: ndarray = 1000, result=0):
    for i in range(x.shape[0]):
        result += np.multiply(x[i]**2, a**(i / (x.shape[0]-1)))

    return result


def f2(x):
    return x
