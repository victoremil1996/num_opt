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


def f3ToTest(x: ndarray, eps=10**(-16), a: ndarray = 2):
    fv = np.log(eps+f1(x, a)[0])
    return fv

x0 = np.array([7, 3])
history = [np.linalg.norm(f1(x0)[1])]
def callback(x):
    fobj = np.linalg.norm(f1(x)[1])
    history.append(fobj)

result = minimize(f1ToTest, x0, method='BFGS', tol=1e-8, callback=callback)
print(history)

plt.plot(np.log10(history))
np.log(history)
result = minimize(f4, x0, method='BFGS', tol=1e-8)
