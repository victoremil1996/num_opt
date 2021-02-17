########## TO TEST
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
from bib.week1.cs_functions import f1, f2, f3, fh, f4, f5, A44

def f1ToTest(x: ndarray, a: ndarray = 1000):
    fv = 0
    for i in range(x.shape[0]):
        fv += x[i]**2 * a**(i / (x.shape[0]-1))
    return fv


def f2ToTest(x: ndarray):
    fv = 0
    if x.shape[0] != 2:
        print("Fail, x must be two-dimensional")
    else:
        fv = (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    return fv


x0 = np.array([0.2, 0.4])
res = minimize(f1ToTest, x0, method='nelder-mead',options={'xatol': 1e-8, 'disp': True})

f1ToTest(x0)
f1(np.array([0,0]))

