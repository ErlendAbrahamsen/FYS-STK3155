from Part_a import X, y
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from scipy.stats import kde

def derivative(X, y, beta):
    """
    Returns the hessian and gradient of log function
    """

    p = logfunc(X, beta)
    W = np.diagflat(p*(1-p))
    H = X.T @ W @ X

    Grad = np.dot(X.T, p-y)

    return H, Grad

def newtonsMethod(X, y, k=0.4):
    """
    Newtons method with optional step size k.
    """

    #np.random.seed(1)
    p = np.shape(X)[1]
    beta = np.random.normal(0, 10, (p, 1)) #Random guess
    max_iter = int(1e2)
    for i in range(max_iter):
        H, Grad = derivative(X, y, beta)
        beta = beta - k*np.dot(np.linalg.inv(H), Grad)

    return beta

def logfunc(X, beta):
    """
    Probility mass function
    """

    Xb = np.array(np.dot(X, beta), dtype=np.float64)

    return 1/(1 + np.exp(-Xb))

def accuracy(p, y):
    """
    Expectance of correctly guessed / total data
    """

    p, y = np.ravel(p), np.ravel(y)
    total, score = len(p), 0
    for p, y in zip(p, y):
        if y == 1:
            score += p
        else:
            score += 1 - p

    return score/total

if __name__ == '__main__':
    #beta = np.linspace(0,1,24).reshape(-1,1)
    X_max = np.max(X)
    #X = preprocessing.MinMaxScaler().fit_transform(X)
    X = X/X_max

    beta = newtonsMethod(X, y)
    p = logfunc(X, beta)
    accuracy = accuracy(p, y)

    #Plot
    p, y = np.ravel(p), np.ravel(y)
    plt.suptitle("Accuracy = %.3f" % accuracy)
    plt.hist2d(p, y, bins=(30, 30), cmap=plt.cm.Greys)
    plt.xlabel("Predicted probability"), plt.ylabel("y: (0=No, 1=Yes)")
    plt.colorbar().ax.set_title("Number of predictions")
    #plt.show()
