import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from Part_a import FrankeFunction, DesignMatrix, Polynom, MSE, R2, MyPlot

def Ridge(X, y, Lambda = 1e-2):
    """
    """

    XTX = X.T.dot(X)
    I = np.identity(len(XTX))
    beta = np.linalg.inv(XTX + Lambda*I).dot(X.T).dot(y)

    return beta

if __name__ == '__main__':
    #Make data.
    x = np.linspace(0, 1, 25)
    y = np.linspace(0, 1, 25)
    x, y = np.meshgrid(x, y)

    #Ridge regression of degree 1...n with plots
    n = 5
    Noise = False
    for n in range(1, n+1):
        X = DesignMatrix(np.ravel(x), np.ravel(y), n=n)
        z = FrankeFunction(np.ravel(x), np.ravel(y), Noise=Noise)
        beta = Ridge(X, z)
        z_Ridge = Polynom(x, y, beta, n=n)

        #Set up a plot
        MyPlot(x, y, z_Ridge)
        plt.show()
        #plt.savefig("OLS_n%d_N%s.png" % (n, Noise))
