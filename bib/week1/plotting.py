""""
Contains plotting function
"""

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from numpy import ndarray
from bib.week1.cs_functions import f1, f2, f3, fh, f4, f5, A44

xline = np.array(np.linspace(-2.5, 4, 55))
yline = np.array(np.linspace(-2.5, 4, 55))
zline = np.array(np.linspace(-2.5, 4, 55))
X, Y = np.meshgrid(xline, yline)
xy = np.array([X, Y])

# Alternative method to populate functions:
Res = np.empty((xline.shape[0], yline.shape[0]))
for i, x in enumerate(xline):
    for j, y in enumerate(yline):
        Res[i, j] = f4(np.array([x, y]))

# Plots
fig = plt.figure()
ax = plt.axes(projection='3d')
# f1
ax.plot_surface(X, Y, f1(xy[0:], np.array(2))[0], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Ellipsoid')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f_1 value')

# f2
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, f2(xy[0:])[0], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Banana')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f_2 value')

# f3
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, f3(xy[0:], 2)[0], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Log-Ellipsoid')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f_3 value')

# f4
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, f4(xy), rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Attractive Sector Function 1')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f_4 value')

# xy.shape
# f3(xy[0:],2)[0:2][0].shape
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, f5(xy), rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Attractive Sector Function 2')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f_5 value')

# TESTS
# Eksempel på test af gradient for f_1
A44(1000, 10**(-8), np.array([0,-1.3, 987,-323, 2.32]))
f1(np.array([0,-1.3, 987,-323, 2.32]))[2]
np.multiply(f1(np.array([0,-1.3, 987,-323, 2.32]))[1],(np.array([0,-1.3, 987,-323, 2.32])[1]))

f3(np.array([1,2]),10**(-8))

A45(1000, 10**(-8), np.array([0,-1.3, 987,-323, 2.32]))