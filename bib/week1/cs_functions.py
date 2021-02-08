"""
Contains functions for numerical optimisation analysis
"""

from typing import Union
import numpy as np
from numpy import ndarray
import math


def f1(x: ndarray, a: ndarray = 1000):
    fv = 0
    for i in range(x.shape[0]):
        fv += x[i]**2 * a**(i / (x.shape[0]-1))

    grad = 0
    for i in range(x.shape[0]):
        grad += 2 * x[i] * a ** (i / (x.shape[0] - 1))

    hess = 0
    for i in range(x.shape[0]):
        hess += 2 * a ** (i / (x.shape[0] - 1))

    return fv, grad, hess


def f2(x: ndarray):
    if x.shape[0] != 2:
        print("Fail, x must be two-dimensional")
    else:
        fv = (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
        grad = np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
        hess = np.array([[-400*(x[1]-x[0])+800*x[0]**2+2, -400*x[0]], [-400*x[0], 200]])

    return fv, grad, hess


def f3(x: ndarray, epsilon=10**(-16)):

    fv = np.log(epsilon+f1(x)[0])
    grad = f1(x)[1]/(epsilon+f1(x)[0])

    return fv, grad


def fh(x, q=10**8):
    return (np.log(1+np.exp(-abs(q*x)))+max(q*x, 0)) / q


def f4(x: ndarray, q=10**8, result=0):
    for i in range(x.shape[0]):
        result += fh(x[i])+100*fh(-x[i])
    return result


def f5(x: ndarray, q=10**8, result=0):
    for i in range(x.shape[0]):
        result += fh(x[i])**2+100*fh(-x[i])**2
    return result
