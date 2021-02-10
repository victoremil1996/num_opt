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

    grad = []
    for i in range(x.shape[0]):
        grad.append(2 * x[i] * a ** (i / (x.shape[0] - 1)))

    hess = []
    for i in range(x.shape[0]):
        hess.append(2 * a ** (i / (x.shape[0] - 1)))

    return fv, grad, np.diagflat(np.diag(np.full((x.shape[0], x.shape[0]), hess)))


def f2(x: ndarray):
    if x.shape[0] != 2:
        print("Fail, x must be two-dimensional")
    else:
        fv = (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
        grad = np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
        hess = np.array([[-400*(x[1]-x[0]**2)+800*x[0]**2+2, -400*x[0]], [-400*x[0], 200]])

    return fv, grad, hess


def f3(x: ndarray, eps=10**(-16), a: ndarray = 2):
    fv = np.log(eps+f1(x, a)[0])
    grad = f1(x, a)[1]/(eps+f1(x, a)[0])
    hess = (f1(x, a)[2]*(eps+f1(x, a)[0])-np.multiply(f1(x, a)[1], f1(x, a)[1]))/(f1[x,a]**2+eps)
    return fv, grad, hess


def fh(x: ndarray, q=10**8):
    return (np.log(1+np.exp(-np.abs(q*x)))+np.maximum(q*x, 0)) / q


def f4(x: ndarray, q=10**8, result=0):
    for i in range(x.shape[0]):
        result += fh(x[i])+100*fh(-x[i])
    return result


def f5(x: ndarray, q=10**8, result=0):
    for i in range(x.shape[0]):
        result += np.power(fh(x[i]), 2)+100*np.power(fh(-x[i]), 2)
    return result


def A44(a, eps, x: ndarray):
    out = []
    for i in range(x.shape[0]):
        out.append(((x[i]+eps)**2 * a**(i / (x.shape[0]-1))-((x[i])**2 * a**(i / (x.shape[0]-1)))) / eps)
    return out


def A45(a, eps, x: ndarray):
    out = []
    for i in range(x.shape[0]):
        out.append((2 * (x[i]+eps) * a ** (i / (x.shape[0] - 1))-(2 * x[i] * a ** (i / (x.shape[0] - 1)))) / eps)
    return out
