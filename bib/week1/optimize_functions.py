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
import time
import cProfile
import io
import pstats
import functools


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
    start = time.time()
    newx = x
    succes = True
    iteration = 0
    stopping = 10
    x_points = []
    y_points = []
    while stopping > eps and iteration < 60000:
        alpha = backtracklsearch(newx, f, fg, -fg(newx))
        xtemp = newx
        newx = newx-fg(newx)*alpha
        fval = f(newx)
        x_points.append(newx)
        #y_points.append(fval)
        iteration += 1
        if gnorm == True:
            stopping = np.linalg.norm(fg(newx))
        else:
            stopping = np.abs(f(xtemp)-f(newx))
        if iteration >= (60000-1):
            succes = False
           #print(iteration, fval, newx, fg(newx), alpha)
    elapsed_time = (time.time() - start)
    return x_points, succes, iteration, elapsed_time


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
    start = time.time()
    y_points = [f(x)]
    x_points = [x]
    max_run = 100000
    run = 0
    stopping = 10
    succes = True
    # Change to stopping criteria instead of while True
    while stopping > eps and run < 100000:
        f_hess = fh(x)
        new_hess = f_hess + added_multi_identity(f_hess) * np.eye(len(x))
        p = - np.linalg.inv(new_hess) @ fg(x)
        stepsize = backtracklsearch(x, f, fg, p)
        xtemp = x
        x = x + stepsize * p
        y = f(x)
        x_points.append(x)
        #y_points.append(y)
        run += 1
        if run >= (1000000-1):
            succes = False
        if gnorm == True:
            stopping = np.linalg.norm(fg(x))
        else:
            stopping = np.abs(f(xtemp)-f(x))
        print(run, stepsize, x, f(x))
        if run > max_run:
            return x_points, y_points, succes, run
    elapsed_time = (time.time() - start)
    return x_points, succes, run, elapsed_time


nmf1, nmf2, nmf3, nmf4, nmf5, sdf1, sdf2, sdf3, sdf4, sdf5 = [0]*100,[0]*100,[0]*100,[0]*100,[0]*100,[0]*100,[0]*100,[0]*100,[0]*100,[0]*100
all_x = [0]*100
pr = cProfile.Profile()
pr.enable()
random.seed(2342)
for i in range(100):
    xtest = np.array(random.randint(-10, 10, size=2))
    all_x[i] = xtest
    nmf1[i] = newton_method(xtest, fv1, grad1, hess1, eps = 10**(-10), gnorm=False)
    print(i)
    nmf2[i] = newton_method(xtest, fv2, grad2, hess2, eps = 10**(-10), gnorm=False)
    print(i)
    nmf3[i] = newton_method(xtest, fv3, grad3, hess3, eps = 10**(-10), gnorm=False)
    print(i)
    nmf4[i] = newton_method(xtest, fv4, grad4, hess4, eps = 10**(-10), gnorm=False)
    print(i)
    nmf5[i] = newton_method(xtest, fv5, grad5, hess5, eps = 10**(-10), gnorm=False)
    print(i)
    sdf1[i] = steepest(xtest, fv1, grad1, eps = 10**(-10), gnorm=False)
    print(i)
    sdf2[i] = steepest(xtest, fv2, grad2, eps = 10**(-10), gnorm=False)
    print(i)
    sdf3[i] = steepest(xtest, fv3, grad3, eps = 10**(-10), gnorm=False)
    print(i)
    sdf4[i] = steepest(xtest, fv4, grad4, eps = 10**(-10), gnorm=False)
    print(i)
    sdf5[i] = steepest(xtest, fv5, grad5, eps = 10**(-10), gnorm=False)
    print(i)
pr.disable()
s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
ps.print_stats()
with open('test.txt', 'w+') as f:
    f.write(s.getvalue())

sumite = []
tempsumite = 0
tempsumtid = 0
tempsumsuccess = 0
tempdistance = 0
for i in range(100):
    if len(sdf4[i][0]) >= 60000:
        tempsumsuccess = tempsumsuccess+1
    else:
        tempdistance = tempdistance + np.linalg.norm(sdf4[i][0][len(sdf4[i][0])-1] - np.array([0, 0]))
    print(tempdistance, i)
#    tempsumite = tempsumite + sdf5[i][2]
#    tempsumtid = tempsumtid + sdf5[i][3]
#    tempsumsuccess = tempsumsuccess + sdf5[i][1]

nmf1[i] = newton_method(xtest, fv1, grad1, hess1, eps=10 ** (-10), gnorm=False)
nmf2[i] = newton_method(xtest, fv2, grad2, hess2, eps=10 ** (-10), gnorm=False)
nmf3[i] = newton_method(xtest, fv3, grad3, hess3, eps=10 ** (-10), gnorm=False)
nmf4[i] = newton_method(xtest, fv4, grad4, hess4, eps=10 ** (-10), gnorm=False)
nmf5[i] = newton_method(xtest, fv5, grad5, hess5, eps=10 ** (-10), gnorm=False)
sdf1[i] = steepest(xtest, fv1, grad1, eps=10 ** (-10), gnorm=False)
sdf2[i] = steepest(xtest, fv2, grad2, eps=10 ** (-10), gnorm=False)
sdf3[i] = steepest(xtest, fv3, grad3, eps=10 ** (-10), gnorm=False)
sdf4[i] = steepest(xtest, fv4, grad4, eps=10 ** (-10), gnorm=False)
sdf5[i] = steepest(xtest, fv5, grad5, eps=10 ** (-10), gnorm=False)
norm_x, norm_x5, norm_x10 = [], [], []
xtest5 = np.array([2, 3, -4, 9, -6])
xtest10 = np.array([7, 5, 2, -8, -6, 3, 1, -3, 8, -2])
xtest5 = np.array([1, 2, 3, 4, 5])
xtest10 = np.array([7,3,1,-3,6,1,-9])
sdf15 = steepest(xtest5, fv5, grad5, eps=10**(-10), gnorm=False)
sdf110 = steepest(xtest10, fv5, grad5, eps=10**(-10), gnorm=False)
nmf15 = newton_method(xtest5, fv5, grad5, hess5, eps=10**(-10), gnorm=False)
nmf110 = newton_method(xtest10, fv5, grad5, hess5, eps=10**(-10), gnorm=False)
for i in range(len(nmf5[0][0])-1):
    denominator = np.linalg.norm(nmf5[0][0][i]-np.array([0, 0]))
    norm_x.append(np.log(denominator))
for i in range(len(nmf15[0])-1):
    denominator5 = np.linalg.norm(nmf15[0][i]-np.array([0, 0, 0, 0, 0]))
    norm_x5.append(np.log(denominator5+2**(-64)))
for i in range(len(nmf110[0])-1):
    denominator10 = np.linalg.norm(nmf110[0][i]-np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    norm_x10.append(np.log(denominator10))

norm_x3 = norm_x
plt2 = plt.plot(norm_x, 'o')
plt2 = plt.plot(norm_x5, 'o')
plt2 = plt.plot(norm_x10, 'o')
#plt3 = plt.plot(np.log(nomi), np.log(denomi))
plt.ylabel('$||x_k-x*||_2$')
plt.xlabel('Iterations')
plt.title('$f_5$, Newtons Method')
plt.legend(['x = [0, 7]', 'x = [2, 3, -4, 9, -6]', 'x = [7, 5, 2, -8, -6, 3, 1, -3, 8, -2]'])
plt.legend(['x = [0, 7]', 'x = [-1, -3]', 'x = [-1, 2]'])
