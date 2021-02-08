""""
Contains plotting function
"""

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from numpy import ndarray
from bib.week1.cs_functions import f1, f2, f3, fh,f4,f5

# Data for a three-dimensional line
# zline = np.array(np.linspace(0, 15, 1000))
xline = np.array(range(-30, 30))
yline = np.array(range(-30, 30))
X, Y = np.meshgrid(xline, yline)
xy = np.array([[X], [Y]])
fig = plt.figure()
ax = plt.axes(projection='3d')
# f1
# ax.plot3D(yline, xline, f1(xy[0:], np.array(100))[0], 'grey')
# ax.contour3D(X, Y, f1(xy[0:], np.array(1))[0:2][0], 50, cmap='binary')
ax.plot_surface(X, Y, f1(xy[0:], np.array(1))[0:2][0], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Ellipsoid')
# f2
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, f2(xy[0:])[0:2][0], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Banana')
# f3
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, f3(xy[0:])[0:2][0], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
f4(X)