import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

def FrankeFunction(x, y):
    """
    Franke function
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4

def DesignMatrix(x, y, n=5):
    """
    Returns design matrix with rows on the form
    [1, x, y, x^2, xy, y^2, ..., x^n, x^(n-1)y, x^(n-2)y^2, ..., y^n].
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

def Polynom(x, y, coeff, n=5):
    """
    Returns ndarray of points from our fitted continouse polynom.
    Inputs x and y points (ndarray), 1darray of coefficients, and
    optional int argument n for the polynom degree.
    Matches coefficient setup of DesignMatrix!
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

    z_true, z_pred = np.ravel(z_true), np.ravel(z_pred) #Ravel to be sure

    return np.mean((z_true - z_pred)**2)

def R2(z_true, z_pred):
    """
    Inputs two data sets and returns the R^2 score
    """

    z_true, z_pred = np.ravel(z_true), np.ravel(z_pred)

    avg = np.mean(z_true)
    s1 = np.sum((z_true - z_pred)**2)
    s2 = np.sum((z_true - avg)**2)

    return 1 - s1/s2

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

#Make data importable to other parts
n = 50
l = np.linspace(0, 1, n)
x, y = np.meshgrid(l, l)

np.random.seed(1)                    #produce same N(0, 1) each run
eps = np.random.normal(0, 1, (n, n)) #(n, n) matrix of N(0,1)
f = FrankeFunction(x, y)             #Real continouse realationship
z = f + eps                          #Data with N(0,1) noise

if __name__ == '__main__':
    ##OLS regression of degree 1...n with plots
    n = 5 #Polynom degree
    mse, r2 = [], []
    for n in range(1, n+1):
        X = DesignMatrix(np.ravel(x), np.ravel(y), n=n)
        beta = OLS(X, np.ravel(z))
        z_OLS = np.dot(X, beta) #Prediciton

        mse.append(MSE(f, z_OLS)), r2.append(R2(f, z_OLS))

        ##OLS plot
        title = "n=%d Franke-OLS, Noise=True" % n
        MyPlot(x, y, z_OLS.reshape((len(x), len(x))), title=title)
        #plt.savefig("OLS_n%d_Noise.png" % n)
        plt.show()

    ##FrankeFunction plot
    MyPlot(x, y, f, "FrankeFunction")
    #plt.savefig("Franke.png")
    plt.show()

    ##Confidence intervall of beta (n=5)
    z_ = 1.96              #95% CI
    n = np.sqrt(z.size)    #root of sample size
    CI = [(np.round(coeff-z_*np.std(beta)/n, decimals=2),
           np.round(coeff+z_*np.std(beta)/n, decimals=2)) for coeff in beta]
    CI_table = pd.DataFrame({"betas": np.arange(len(beta)),
                             "95% CI": CI})

    print(CI_table, "\n\n")

    n = 5
    for n in range(1, n+1):
        print("n=%d: MSE=%.4g, R2=%.4g " % (n, mse[n-1], r2[n-1]))
