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

def backtracklsearch(x: ndarray, f, fg, pk):
    """
    Higher c will take less iterations but might incur problems
    """
    a = 1
    rho = 0.5
    c = 0.0001
    while f(x+a*pk) > f(x) + c * a * np.dot(fg(x), pk):
        a = rho*a
    return a


def steepest(x: ndarray, f, fg, eps=10**(-8), gnorm: bool = True):
    """
    If gnorm = True, stopping criteria = L2 norm of gradient
    If gnorm = False, stopping criteria = absolute difference in fct. value
    """
    newx = x
    succes = True
    iteration = 0
    stopping = 10
    x_points = []
    y_points = []
    while stopping > eps and iteration < 1000000:
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
        if iteration >= (1000000-1):
            succes = False
#        print(iteration, fval, newx, fg(newx), alpha)
    return x_points, y_points, succes, iteration


xtest = np.array([20, 4])
steepest(xtest, fv5, grad5, eps=0.000000001, gnorm=True)

""" NEWTONS METHOD """
def added_multi_identity(A):
    """ Algorithm 3.3 from book """
    beta = 10**-3
    dia = np.zeros(len(A))
    for i in range(len(A)):
        dia[i] = A[i, i]
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


def newton_method(x, f, fg, fh, eps=10**(-8), gnorm: bool = True):
    """ Algorithm 3.2 from book """
    y_points = [f(x)]
    x_points = [x]
    max_run = 1000000
    run = 0
    stopping = 10
    succes = True
    # Change to stopping criteria instead of while True
    while stopping > eps and run < 1000000:
        f_hess = fh(x)
        new_hess = f_hess + added_multi_identity(f_hess) * np.eye(len(x))
        p = - np.linalg.inv(new_hess) @ fg(x)
        stepsize = backtracklsearch(x, f, fg, p)
        xtemp = x
        x = x + stepsize * p
        y = f(x)
        x_points.append(x)
        y_points.append(y)
        run += 1
        if run>=(1000000-1):
            succes = False
        if gnorm == True:
            stopping = np.linalg.norm(fg(x))
        else:
            stopping = np.abs(f(xtemp)-f(x))
        print(run, stepsize, x, f(x))
        if run > max_run:
            return x_points, y_points, succes, run

    return x_points, y_points
nmf1 = [range(100)]
for i in range(100):
    xtest = np.array([2, 3])
    nmf1 = newton_method(xtest, fv1, grad1, hess1)
    nmf2 = newton_method(xtest, fv2, grad2, hess2)
    nmf3 = newton_method(xtest, fv3, grad3, hess3)
    nmf4 = newton_method(xtest, fv4, grad4, hess4, gnorm=False)
    nmf5 = newton_method(xtest, fv5, grad5, hess5, gnorm=False)
    sdf1 = steepest(xtest, fv1, grad1, gnorm=True)
    sdf2 = steepest(xtest, fv2, grad2, eps=10**(-12), gnorm=True)
    sdf3 = steepest(xtest, fv3, grad3, gnorm=False)
    sdf4 = steepest(xtest, fv4, grad4, gnorm=True)
    sdf5 = steepest(xtest, fv5, grad5, gnorm=False)

testarr = []
norm_x = []
nomi = []
denomi = []
for i in range(np.array(sdf1[1]).shape[0]-1):
    #nominator = np.linalg.norm(nmf3[0][i+1]-np.array([0, 0]), ord=2)
    denominator = np.linalg.norm(sdf1[1][i]-np.array([0, 0, 0, 0, 0]))
    #nomi.append(nominator)
#    denomi.append(np.log(denominator))
    norm_x.append(np.log(denominator))

norm_x3 = norm_x
#plt.plot(nmf2[0])
plt2 = plt.plot(norm_x3, 'o')
plt3 = plt.plot(np.log(nomi), np.log(denomi))
plt.ylabel('$||x_k-x*||_2$')
plt.xlabel('Iterations')
plt.title('$f_1$, Steepest Descent')
plt.legend(['α = 100', 'α = 10'])
