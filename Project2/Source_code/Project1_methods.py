import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    """
    Franke function.
    Input/return is ndarray or float/int.
    """

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4

def frankeData(n=50):
    """
    Franke generated data with noise
    """

    l = np.linspace(0, 1, n)
    x, y = np.meshgrid(l, l)
    x, y = np.ravel(x), np.ravel(y)

    np.random.seed(1)                             #produce same N(0, 1) each run
    eps = np.random.normal(0, 1, n**2)            #n^2 points of N(0,1)
    z = f(x, y) + eps                             #Data with N(0,1) noise

    X = np.zeros((n**2, 2))
    X[:, 0], X[:, 1] = np.ravel(x), np.ravel(y)   #(n^2, 2) matrix
    z = np.ravel(z).reshape(-1, 1)                #(n^2, 1) vector

    return X, z

def MyPlot(x, y, z, title="", shrink=0.5, aspect=4):
    """
    3D plotting method with features fitting our situation
    """

    #Set up a 3d plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

    labels = (ax.set_xlabel("x-axis"), ax.set_ylabel("y-axis"),
                ax.set_zlabel("z-axis"), ax.set_title(title))

    fig.colorbar(surf, shrink=shrink, aspect=aspect)

    return True

if __name__ == '__main__':
    #Noised franke plot
    X, z = frankeData()
    z = np.ravel(z).reshape(50, 50)
    x, y = X[:, 0].reshape(50, 50), X[:, 1].reshape(50, 50)

    MyPlot(x, y, z, title="Noised franke plot")
    plt.show()
