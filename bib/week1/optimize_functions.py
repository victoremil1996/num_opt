########## Optimize functions
#######

import scipy as spy
from scipy.optimize import minimize
from random import seed
from random import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from numpy import ndarray
from numpy import *
#from bib.week1.cs_functions import *
from bib.week1.cs_def import *


def testfunk(x: ndarray):
    sum = 0
    for i in range(len(x)):
        sum = sum + x[i]**2
    return sum


def testfunkg(x: ndarray):
    return np.array([2*x[0], 2*x[1]])


def backtracklsearch(x: ndarray, f, fg, pk):
    a = 1
    rho = 0.8
    c = 0.0001
    while f(x+a*pk) > f(x) + c * a * np.dot(fg(x), pk):
        a = rho*a
    return a


def steepest(x: ndarray, f, fg, eps=10**(-8), gnorm: bool=True):
    """
    If gnorm = True, stopping criteria = L2 norm of gradient
    If gnorm = False, stopping criteria = absolute difference in fct. value
    """
    newx = x
    iteration = 0
    stopping = 10
    x_points = []
    y_points = []
    while stopping > eps and iteration < 10000:
        alpha = backtracklsearch(newx, f, fg, -fg(newx))
        xtemp = newx
        newx = newx-fg(newx)*alpha
        fval = f(newx)
        x_points.append(newx)
        y_points.append(fval)
        iteration += 1
        if gnorm == True:
            stopping = np.linalg.norm(fg(newx))
        else:
            stopping = np.abs(f(xtemp)-f(newx))
        print(fval, newx, fg(newx), alpha)
    return fval, x_points, y_points, iteration


xtest = np.array([20, 4])
steepest(xtest, fv5, grad5, eps=0.000000001, gnorm=True)

""" NEWTONS METHOD """
def added_multi_identity(A):
    """ Algorithm 3.3 from book """
    beta = 10**-3
    dia = np.zeros(len(A))
    for i in range(len(A)):
        dia[i] = A[i,i]
    if np.min(dia) > 0:
        tau = 0
        return tau
    else:
        tau = -np.min(dia) + beta
    while True:
        try:
            np.linalg.cholesky(A+tau*np.eye(len(A)))
            return tau
        except:
            tau = max(2*tau, beta)


def newton_method(x, f, fg, fh, eps=10**(-8)):
    """ Algorithm 3.2 from book """
    y_points = [f(x)]
    x_points = [x]
    max_run = 1000000
    run = 0
    # Change to stopping criteria instead of while True
    while np.linalg.norm(fg(x)) > eps and run < 1000000:
        f_hess = fh(x)
        new_hess = f_hess + added_multi_identity(f_hess) * np.eye(len(x))
        p = - np.linalg.inv(new_hess) @ fg(x)
        stepsize = backtracklsearch(x, f, fg, p)
        x = x + stepsize * p
        y = f(x)
        x_points.append(x)
        y_points.append(y)
        run += 1
        print(run, stepsize, x, f(x))
        if run > max_run:
            return x_points, y_points

    return x_points, y_points

#newton_method(xtest, fv4, grad4, hess4)