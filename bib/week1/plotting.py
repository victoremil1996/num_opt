""""
Contains plotting function
"""

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from numpy import ndarray
from bib.week1.cs_functions import f1, f2, f3, fh, f4, f5, A44

xline = np.array(np.linspace(-1, 20, 55))
yline = np.array(np.linspace(-1, 20, 55))
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
ax.set_title('$Ellipsoid, α = 2$')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f_1(x_1,x_2)$')

# f2
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, f2(xy[0:])[0], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Banana')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f_2(x_1,x_2)$')
#Til contourplot
fig, ax = plt.subplots()
CS = plt.contourf(X,Y,f2(xy[0:])[0],levels=[0,1,10,25,50,75,100],cmap='RdGy')
ax.set_title('Banana')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
plt.colorbar()

# f3
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, f3(xy)[0], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Log-Ellipsoid, α = 2')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f_3(x_1,x_2)$')
f3(np.array([1, 2]))

# f4
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, f4(xy), rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Attractive Sector Function 1')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f_4(x_1,x_2)$')
# xy.shape
# f3(xy[0:],2)[0:2][0].shape
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, f5(xy), rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Attractive Sector Function 2')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f_5(x_1,x_2)$')
# TESTS
# Eksempel på test af gradient for f_1
A44(1000, 10**(-8), np.array([0,-1.3, 987,-323, 2.32]))
f1(np.array([0,-1.3, 987,-323, 2.32]))[1]
np.multiply(f1(np.array([0,-1.3, 987,-323, 2.32]))[1],(np.array([0,-1.3, 987,-323, 2.32])[1]))
xt = np.array([-3.2, 7.7])
f2(xt)[2]
f3(np.array([1,2]),10**(-8))[2]
eps = 10**(-8)
# test banan
f2(xt)[2]
(f2(np.array([-3.2, 7.7+eps]))[0]-f2(np.array([-3.2, 7.7]))[0])/eps
# test banan hesse
(f2(np.array([-3.2+eps, 7.7]))[1]-f2(np.array([-3.2, 7.7]))[1])/eps
1280-1279.99999791
# Test af f_3
f3(xt,10**(-16),2)[0]
(f3(np.array([-3.2+eps, 7.7]),10**(-16),2)[1]-f3(np.array([-3.2, 7.7]),10**(-16),2)[1])/eps
(f3(np.array([-3.2, 7.7+eps]))[0]-f3(np.array([-3.2, 7.7]))[0])/eps

# Test af f_4
(f4(np.array([-3.2+eps, 7.7]))-f4(np.array([-3.2, 7.7])))/eps