"""
Contains functions for numerical optimisation analysis
"""

from typing import Union
import numpy as np
from numpy import ndarray
import math


class F1:
    """
    Class that computes function value, gradient and hessian for F1
    """

    def __init__(self, x: ndarray, a: ndarray = np.array([1000])):
        self.x = x
        self.a = a

    def fv(self):
        fv = 0
        for i in range(self.x.shape[0]):
            fv += self.x[i]**2 * self.a**(i / (self.x.shape[0]-1))

        return fv

    def grad(self):
        grad = []
        for i in range(self.x.shape[0]):
            grad.append(2 * self.x[i] * self.a ** (i / (self.x.shape[0] - 1)))

        return grad

    def hess(self):
        hess = []
        for i in range(self.x.shape[0]):
            hess.append(2 * self.a ** (i / (self.x.shape[0] - 1)))

        return np.diagflat(np.diag(np.full((self.x.shape[0], self.x.shape[0]), hess)))


class F2:
    """
    Class that computes function value, gradient and hessian for F2
    """

    def __init__(self, x: ndarray, a: ndarray = np.array([1000])):
        self.x = x
        self.a = a

    def fv(self):
        if self.x.shape[0] != 2:
            print("Fail, x must be two-dimensional")
        else:
            fv = (1 - self.x[0]) ** 2 + 100 * (self.x[1] - self.x[0] ** 2) ** 2

        return np.array([fv])

    def grad(self):
        if self.x.shape[0] != 2:
            print("Fail, x must be two-dimensional")
        else:
            grad = np.array([-2 * (1 - self.x[0]) - 400 * self.x[0] * (self.x[1] - self.x[0] ** 2)
                            , 200 * (self.x[1] - self.x[0] ** 2)])

        return grad

    def hess(self):
        if self.x.shape[0] != 2:
            print("Fail, x must be two-dimensional")
        else:
            hess = np.array([[-400 * (self.x[1] - self.x[0] ** 2) + 800 * self.x[0] ** 2 + 2
                             , -400 * self.x[0]], [-400 * self.x[0], 200]])

        return hess


class F3:
    """
    Class that computes function value, gradient and hessian for F3
    """

    def __init__(self, x: ndarray, eps=10**(-16), a: ndarray = 1000):
        self.x = x
        self.eps = eps
        self.a = a
        self.funk1 = F1(x=x, a=a)

    def fv(self):
        fv = np.log(self.eps + self.funk1.fv())
        return fv

    def grad(self):
        grad = self.funk1.grad() / (self.eps + self.funk1.fv())
        return grad

    def hess(self):
        hess = (np.matrix(self.funk1.hess())*(self.eps+self.funk1.fv())
                - np.dot(np.matrix(self.funk1.grad()).T,
                         np.matrix(self.funk1.grad())))/(self.eps + self.funk1.fv())**2
        return hess


class H:
    """
    Class that computes function value, gradient and hessian for F3
    """

    def __init__(self, x: ndarray, q=10**8):
        self.x = x
        self.q = q

    def fv(self):
        fv = (np.log(1+np.exp(-np.abs(self.q*self.x))) + np.maximum(self.q*self.x, 0)) / self.q
        return fv

    def gradm(self):
        if self.x >= 0:
            return -(np.exp(-self.q * self.x) / (1 + np.exp(-self.q * self.x)))
        else:
            return -(1 / (1 + np.exp(self.q * self.x)))

    def gradhp(self):
        if self.x >= 0:
            return 1 / (1 + np.exp(-self.q * self.x))
        else:
            return np.exp(self.q * self.x) / (1 + np.exp(self.q * self.x))

    def hess(self):
        hess = np.matmul(np.matrix(self.q*np.exp(-self.q*self.x)),
                         np.linalg.inv(np.matmul(np.matrix(np.exp(-self.q*self.x)+1), np.matrix(np.exp(-self.q*self.x)+1).T)))
        return hess


class F4():
    """
     Class that computes function value, gradient and hessian for F4
     """
    def __init__(self, x: ndarray, q=10 ** 8):
        self.x = x
        self.q = q

    def fv(self):
        grad = []
        for i in range(self.x.shape[0]):
            grad.append("")
        return print("Vi skal indhente gradhp og gradhm på en smart måde, hvor den kan indtage x[i]")

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
    grad = np.array(f1(x, a)[1])/(eps+f1(x, a)[0])
    hess = (np.multiply(f1(x, a)[2], (eps+f1(x, a)[0]))-np.dot(np.transpose(np.matrix(f1(np.array(x), a)[1])), np.matrix(f1(np.array(x), a)[1])))\
           / (f1(x, a)[0]**2+eps)
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
