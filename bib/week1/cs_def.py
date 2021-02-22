"""
Contains functions for numerical optimisation analysis
"""

from typing import Union
import numpy as np
from numpy import ndarray
import math


def fv1(x: ndarray, a: ndarray = np.array([1000])):
    fv = 0
    if x.shape[0]>=2:
        for i in range(x.shape[0]):
          fv += x[i]**2 * a**(i / (x.shape[0]-1))
    else:
        fv = x[0] ** 2
    return fv


def grad1(x: ndarray, a: ndarray = 1000):
        grad = []
        if x.shape[0] >= 2:
            for i in range(x.shape[0]):
                grad.append(2 * x[i] * a ** (i / (x.shape[0] - 1)))
        else:
            grad.append(2*x[0])
        return np.array(grad)


def hess1(x: ndarray, a: ndarray = np.array([1000])):
    hess = []
    if x.shape[0] >= 2:
        for i in range(x.shape[0]):
            hess.append(2 * a ** (i / (x.shape[0] - 1)))
    else:
        hess = 2
    return np.diagflat(np.diag(np.full((x.shape[0], x.shape[0]), hess)))


def fv2(x: ndarray):
    fv = []
    if x.shape[0] != 2:
        print("Fail, x must be two-dimensional")
    else:
        fv = (1 - x[0]) ** 2 + 100 * (x[1] - (x[0] ** 2)) ** 2
    return fv


def grad2(x: ndarray):
    if x.shape[0] != 2:
        print("Fail, x must be two-dimensional")
    else:
        grad = [-2 * (1 - x[0]) - 400 * x[0] * (x[1] - (x[0] ** 2))
                , 200 * (x[1] - (x[0] ** 2))]
    return grad


def hess2(x: ndarray):
    if x.shape[0] != 2:
        print("Fail, x must be two-dimensional")
    else:
        hess = np.array([[-400 * (x[1] - x[0] ** 2) + 800 * x[0] ** 2 + 2
                        , -400 * x[0]], [-400 * x[0], 200]])
    return hess


def fv3(x: ndarray, eps=10**(-16), a: ndarray = 2):
    fv = np.log(eps + fv1(x=x,a=a))
    return fv


def grad3(x: ndarray, eps=10**(-16), a: ndarray = 2):
    grad = grad1(x=x, a=a) / np.array([(eps + fv1(x=x,a=a))])
    return grad


def hess3(x: ndarray, eps=10**(-16), a: ndarray = 2):
    hess = (np.matrix(hess1(x=x, a=a))*(eps+fv1(x=x,a=a))
            - np.dot(np.matrix(grad1(x=x,a=a)).T,
                     np.matrix(grad1(x=x,a=a))))/(eps + fv1(x=x,a=a))**2
    return np.array(hess)


def fvh(x: ndarray, q=10**8):
    fv = (np.log(1+np.exp(-np.abs(q*x))) + max(q*x, 0)) / q
    return fv


def gradhm(x: ndarray, q=10**8):
    if x >= 0:
        return -(np.exp(-q * x) / (1 + np.exp(-q * x)))
    else:
        return -(1 / (1 + np.exp(q * x)))


def gradhp(x: ndarray, q=10**8):
    if x >= 0:
        return 1 / (1 + np.exp(-q * x))
    else:
        return np.exp(q * x) / (1 + np.exp(q * x))


def hessh(x: ndarray, q=10**8):
    if x >= 0:
        return (q*np.exp(-q*x))/((1+np.exp(-q*x))**2)
    else:
        (q * np.exp(q * x)) / ((1 + np.exp(q * x)) ** 2)


def hess4(x: ndarray, q=10**8):
    hess = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                hess[i, j] = 0
            else:
                hess[i, j] = hessh(x[i], q)**2 + 100 * gradhm(x[i], q)**2
    return hess


def hess5(x: ndarray, q=10**8):
    hess = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                hess[i, j] = 0
            else:
                hess[i, j] = 2 * gradhp(x[i], q)**2 + 2*fvh(x[i], q) * hessh(x[i], q)
                + 200 * gradhm(x[i], q)**2 + 200 * fvh(-x[i], q) * hessh(x[i], q)
    return hess


def fv4(x: ndarray, q=10**8):
    s = 0
    for i in range(len(x)):
        s +=fvh(x[i], q) + 100 * fvh(-x[i], q)
    return s


def grad4(x: ndarray, q=10**8):
    grad = []
    for i in range(len(x)):
        grad.append(gradhp(x[i], q) + 100 * gradhm(x[i], q))
    return np.array(grad)


def fv5(x: ndarray, q=10**8):
    s = 0
    for i in range(len(x)):
        s += fvh(x[i], q)**2 + 100 * fvh(-x[i], q)**2
    return s


def grad5(x: ndarray, q=10**8):
    grad = []
    for i in range(len(x)):
        grad.append(2*fvh(x[i], q) * gradhp(x[i], q) + 200*fvh(-x[i], q) * gradhm(x[i], q))
    return np.array(grad)