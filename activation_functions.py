import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define activation functions
def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def elu(x, alpha=1):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



x = np.linspace(-6, 6, 100)
sigmoid_x = np.linspace(-6, 6, 100)

fig = plt.figure(figsize=(12, 10))

# Linear plot
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x, linear(x), color="navy", alpha=0.5, linewidth=3, label = r"Linear, $\alpha = 1.0$")
ax1.set_xlim([-6, 6])
ax1.set_ylim([-6, 6])
ax1.set_xlabel('z')
ax1.set_ylabel('linear(z)')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.legend()

# Sigmoid plot
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(sigmoid_x, sigmoid(sigmoid_x), color="orange", alpha=0.5, linewidth=3, label = "Sigmoid")
ax2.set_xlim([-6, 6])
ax2.set_ylim([0, 1])
ax2.set_xlabel('z')
ax2.set_ylabel('$\sigma$(z)')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.legend()


# ReLU plot
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(x, relu(x), color="firebrick", alpha=0.5, linewidth=3, label="ReLU")
ax3.plot(x, leaky_relu(x, alpha = 0.1), color="blue", alpha=0.5, linewidth=3, label=r"LeakyReLU, $\alpha = 0.1$ ")
ax3.plot(x, elu(x), color="green", alpha=0.5, linewidth=3, label=r"ELU, $\alpha = 1.0$ ")
ax3.set_xlim([-6, 6])
ax3.set_ylim([-2, 6])
ax3.set_xlabel('z')
ax3.set_ylabel('f(z)')
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.legend()

# Softmax example
_x = np.linspace(-6, 6, 100)
_y = np.linspace(-6, 6, 100)[::-1]
_xv, _yv = np.meshgrid(_x, _y)

# Compute softmax for each point on the grid
_z1 = np.array([softmax(np.array([x,y]))[0] for x, y in zip(np.ravel(_xv), np.ravel(_yv))])
_z2 = np.array([softmax(np.array([x,y]))[1] for x, y in zip(np.ravel(_xv), np.ravel(_yv))])
_z1 = _z1.reshape(_xv.shape)
_z2 = _z2.reshape(_yv.shape)

ax4 = fig.add_subplot(2, 2, 4, projection='3d')
surf1 = ax4.plot_surface(_xv, _yv, _z1, color='b', alpha=0.5, label=r"$ softmax(\vec{z})_1 $")
surf2 = ax4.plot_surface(_xv, _yv, _z2, color='r', alpha=0.5, label=r"$ softmax(\vec{z})_2 $")
ax4.set_xlabel('$z_{1}$')
ax4.set_ylabel('$z_{2}$')
ax4.invert_yaxis()
ax4.set_zlabel(r'$ softmax(\vec{z})_i $')
ax4.view_init(elev=25, azim=-60)  # Set the viewing angle

surf1._facecolors2d = surf1._facecolor3d
surf1._edgecolors2d = surf1._edgecolor3d

surf2._facecolors2d = surf2._facecolor3d
surf2._edgecolors2d = surf2._edgecolor3d

ax4.legend()


# Show the resulting plot
plt.savefig("activation_functions.png")
