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
    fobj = np.linalg.norm(grad1(x))
    history.append(fobj)


x0 = np.array(random.randint(-100, 100, size=50))
history = [np.linalg.norm(grad1(x0))]

#BFGS
result_BFGS = minimize(fv1, x0, method='BFGS', tol=10**(-8), callback=callback, jac=grad1)
#Newton-CG
result_NM = minimize(fv1, x0, method='Newton-CG', tol=10**(-8), callback=callback,
                     jac=grad1, hess=hess1)
#Trust-NCG
result_TNCG = minimize(fv1, x0, method='trust-ncg', tol=10**(-8), callback=callback,
                       jac=grad1, hess=hess1)
#plt.plot((np.log(history)))
plt.semilogy(history)
plt.legend(['BFGS', 'Newton-CG', 'trust-ncg'])
plt.title('Optimization Algorithms with d=50')
plt.ylabel('log$||f||_1$')
plt.xlabel('Iterations')

fv1(result_BFGS.x)
result_BFGS.x
result_NM.x
result_TNCG.x
np.log(history)
result = minimize(f4, x0, method='BFGS', tol=1e-8)
