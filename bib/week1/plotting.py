""""
Contains plotting function
"""

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from numpy import ndarray
from bib.week1.cs_functions import f1, f2, f3, fh

fig = plt.figure()
ax = plt.axes(projection='3d')
# Data for a three-dimensional line
zline = np.array(np.linspace(0, 15, 1000))
xline = np.sin(zline)
yline = np.cos(zline)
xy = np.array([[yline], [xline]])
ax.plot3D(yline, xline, f1(xy[0:], np.array(100))[0], 'grey')