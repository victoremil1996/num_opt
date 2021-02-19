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
#from bib.week1.cs_functions import *
from bib.week1.cs_def import *

def callback(x):
    fobj = np.linalg.norm(grad5(x))
    history.append(fobj)
    history_fv.append(fv5(x))


x0 = np.array(random.randint(1, 10, size=2))
x0 = np.array([0.1, 0.1])
#x0 = np.array([1,2])
history = [np.linalg.norm(grad5(x0))]
history_fv = [fv5(x0)]
fv4(x0)
#BFGS
result_BFGS = minimize(fv4, x0, method='BFGS', tol=10**(-8), callback=callback, jac=grad4)
#Newton-CG
result_NM = minimize(fv5, x0, method='Newton-CG', tol=10**(-15), callback=callback, jac=grad5)
#Trust-NCG
result_TNCG = minimize(fv5, x0, method='trust-ncg', tol=10**(-8), callback=callback, jac=grad5,
                       hess=hess5)
#Neal
result_Nelder = minimize(fv4, x0, method='Nelder-Mead', tol=10**(-8), callback=callback)
#plt.plot((np.log(history)))
plt.semilogy(history)
#plt.plot(history_fv)
plt.legend(['BFGS', 'Newton-CG', 'trust-ncg'])
plt.title('Optimization Algorithms - Attractive-Sector 2, d=50')
#plt.ylabel('log$||∇f||_2$')
plt.ylabel('$f$')
plt.xlabel('Iterations')
help(result_NM)
result_BFGS.nfev
fv4(result_BFGS.x)
result_NM.nfev
fv5(result_NM.x)
result_TNCG.maxcv
fv5(result_TNCG.x)

fv4(result_Nelder.x)
grad3(result_Nelder.x)