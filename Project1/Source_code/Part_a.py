import numpy as np
import matplotlib.pyplot as plt

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

def Least_Square_Regression(x_data, y_data, DEG=2):
    """
    Returns coefficients of polynomial fitting the
    least square reggresion.
    DEG is int parameter for choosing the polynomial degree.
    """

    #Fitting x_data, y_data to the vandermonde matrix (X)
    X = np.zeros((len(x_data), DEG))
    X[:,0] = 1
    for n in range(1, deg):
        X[:,n] = [xn for xn in x_data**n]

    #Solving
    coeff = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_data)

    return coeff



if __name__ == '__main__':
    print(FrankeFunction(1,1, Noise=True))
    x_data = np.linspace(0, 1, 100)
    y_data = FrankeFunction()
    print(Least_Square_Regression())
