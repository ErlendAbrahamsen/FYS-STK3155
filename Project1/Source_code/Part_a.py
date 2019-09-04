import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def FrankeFunction(x, y, Noise=False):
    """
    Franke function with optional normal distrubited noise.
    Noise is boolean parameter and adds normal distrubited
    noise if set to True.
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    z = term1 + term2 + term3 + term4

    if Noise:
        return z + np.random.randn()
    else:
        return z

def DesignMatrix(x, y, n=2):
    """
    Returns design matrix with rows on the form [1, x, y, x^2, xy, y^2, ...].
    x and y are 1darrays.
    n is optional int specifying degree of fitting polynom.
    """

    col = int((n+1)*(n+2)/2)                #Number of needed columns
    X = np.ones((len(x), col))              #Design matrix

    for i in range(1, n+1):
        s = int(i*(i+1)/2)
        for j in range(i+1):
            X[:, s+j] = x**(i-j)*y**(j)     #Fitting the column vectors

    return X

def OLS(X, z_data):
    """
    Returns the coefficients beta of polynomial fitting the
    least square reggresion.
    Inputs design matrix X and column vector z_data.
    """

    #Solving the least squares lin. alg. problem
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z_data)

    return beta

def Polynom(x, y, coeff, n=2):
    """
    Returns ndarray of points from our fitted continouse polynom.
    Inputs x and y points (ndarray), 1darray of coefficients, and
    optional int argument n for the polynom degree.
    """

    z = coeff[0]                            #Constant term
    for i in range(1, n+1):
        s = int(i*(i+1)/2)
        for j in range(i+1):
            z += coeff[s+j]*x**(i-j)*y**(j) #Other terms

    return z

def MSE(z_true, z_pred):
    """
    Inputs two data sets and returns mean squared error
    """

    return 1/z_true.size*np.sum((z_true - z_pred)**2)

def R2(z_true, z_pred):
    """
    Inputs two data sets and returns the R^2 score
    """

    avg = np.mean(z_true)
    s1 = np.sum((z_true - z_pred)**2)
    s2 = np.sum((z_true - avg)**2)

    return 1 - s1/s2

if __name__ == '__main__':
    #Make data.
    x = np.linspace(0, 1, 25)
    y = np.linspace(0, 1, 25)
    x, y = np.meshgrid(x, y)


    #OLS regression of degree 1...n with plots
    n = 5
    Noise = False
    for n in range(1, n+1):
        X = DesignMatrix(np.ravel(x), np.ravel(y), n=n)
        z = FrankeFunction(np.ravel(x), np.ravel(y), Noise=Noise)
        beta = OLS(X, z)
        z_OLS = Polynom(x, y, beta, n=n)

        #Set up a plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x, y, z_OLS, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)

        ax.set_zlim(-0.10, 1.40)
        labels = (ax.set_xlabel("x-axis"), ax.set_ylabel("y-axis"),
                    ax.set_zlabel("z-axis"), ax.set_title("n=%d Franke-OLS, Noise=%s" % (n, str(Noise))))

        fig.colorbar(surf, shrink=0.5, aspect=4)
        plt.savefig("OLS_n%d_N%s.png" % (n, Noise))

    #FrankeFunction plot
    z_F = FrankeFunction(x, y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z_F, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_zlim(-0.10, 1.40)
    labels = (ax.set_xlabel("x-axis"), ax.set_ylabel("y-axis"),
                ax.set_zlabel("z-axis"), ax.set_title("FrankeFunction"))

    fig.colorbar(surf, shrink=0.5, aspect=4)
    plt.savefig("FrankeFunction.png")

    #print(MSE(z_F, z_OLS))
    #print(mean_squared_error(z_F, z_OLS))
    #print(R2(z_F[0], z_OLS[0]))
    #print(r2_score(z_F[0],z_OLS[0]))
    #print(np.sum((z_F-0.1)**2))

    #Confidence intervall of beta
