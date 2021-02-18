"""
Contains functions for numerical optimisation analysis
"""

from typing import Union
import numpy as np
from numpy import ndarray
import math



def fv1(x: ndarray, a: ndarray = np.array([1000])):
    fv = 0
    for i in range(x.shape[0]):
      fv += x[i]**2 * a**(i / (x.shape[0]-1))
    return fv


def grad1(x: ndarray, a: ndarray = 1000):
        grad = []
        for i in range(x.shape[0]):
            grad.append(2 * x[i] * a ** (i / (x.shape[0] - 1)))
        return grad


def hess1(x: ndarray, a: ndarray = np.array([1000])):
    hess = []
    for i in range(x.shape[0]):
        hess.append(2 * a ** (i / (x.shape[0] - 1)))
    return np.diagflat(np.diag(np.full((x.shape[0], x.shape[0]), hess)))


def fv2(x: ndarray):
    if x.shape[0] != 2:
        print("Fail, x must be two-dimensional")
    else:
        fv = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    return np.array([fv])


def grad2(x: ndarray):
    if x.shape[0] != 2:
        print("Fail, x must be two-dimensional")
    else:
        grad = np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
                    , 200 * (x[1] - x[0] ** 2)])
    return grad


def hess2(x: ndarray):
    if x.shape[0] != 2:
        print("Fail, x must be two-dimensional")
    else:
        hess = np.array([[-400 * (x[1] - x[0] ** 2) + 800 * x[0] ** 2 + 2
                             , -400 * x[0]], [-400 * x[0], 200]])
    return hess



def fv3(x: ndarray, eps=10**(-16), a: ndarray = 1000):
    fv = np.log(eps + fv1(x=x,a=a))
    return fv

def grad3(x: ndarray, eps=10**(-16), a: ndarray = 1000):
    grad = grad1(x=x, a=a) / np.array([(eps + fv1(x=x,a=a))])
    return grad


def hess3(x: ndarray, eps=10**(-16), a: ndarray = 1000):
    hess = (np.matrix(hess1(x=x, a=a))*(eps+fv1(x=x,a=a))
            - np.dot(np.matrix(grad1(x=x,a=a)).T,
                     np.matrix(grad1(x=x,a=a))))/(eps + fv1(x=x,a=a))**2
    return hess


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


def hessh(x: ndarray, q=10**8): # FORKERT
    hess = np.matmul(np.matrix(q*np.exp(-q*x)),
                     np.linalg.inv(np.matmul(np.matrix(np.exp(-q*x)+1), np.matrix(np.exp(-q*x)+1).T)))
    return hess


#def fv():
#    grad = []
#    for i in range(x.shape[0]):
#        grad.append("")
#    return print("Vi skal indhente gradhp og gradhm på en smart måde, hvor den kan indtage x[i]")

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


def fv4(x: ndarray, q=10**8):
    s = 0
    for i in range(len(x)):
        s +=fvh(x[i], q) + 100 * fvh(-x[i], q)
    return s


def grad4(x: ndarray, q=10**8):
    grad = []
    for i in range(len(x)):
        grad.append(gradhp(x[i], q) + 100 * gradhm(x[i], q))
    return grad


def fv5(x: ndarray, q=10**8):
    s = 0
    for i in range(len(x)):
        s += fvh(x[i], q)**2 + 100 * fvh(-x[i], q)**2


def grad5(x: ndarray, q=10**8):
    grad = []
    for i in range(len(x)):
        grad.append(2*fvh(x[i], q) * gradhp(x[i], q) + 200*fvh(-x[i], q) * gradhm(x[i], q))
    return grad


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
