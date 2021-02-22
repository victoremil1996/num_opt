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
    rho = 0.5
    c = 0.001
    while f(x+a*pk) > f(x) + c * a * np.dot(fg(x), pk):
        a = rho*a
    return a


def steepest(x: ndarray, f, fg, eps = 10**(-8)):
    newx = x
    iteration = 0
    while np.linalg.norm(fg(newx)) > eps and iteration < 1000000:
        alpha = backtracklsearch(newx, f, fg, -fg(newx))
        newx = newx-fg(newx)*alpha
        fval = f(newx)
        iteration += 1
        print(fval, newx, fg(newx), alpha)
    return fval, newx, iteration


xtest = np.array([2,4])
steepest(xtest, fv1, grad1, eps=0.0001)
