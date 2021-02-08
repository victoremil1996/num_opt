"""
Contains functions for numerical optimisation analysis
"""

from typing import Union
import numpy as np
from numpy import ndarray
import math


def f1(x: ndarray, a: ndarray = 1000, result=0):
    for i in range(x.shape[0]):
        result += np.multiply(x[i]**2, a**(i / (x.shape[0]-1)))

    return result


def f3(x: ndarray, epsilon=10**(-16)):

    return math.log(epsilon+f1(x))


def fh(x, q=10**8):

    return math.log(1+math.exp(q*x)) / q
