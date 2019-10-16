import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    np.random.seed(1)                       #produce same N(0, 1) each run
    eps = np.random.normal(0, 1, (n, n))    #(n, n) matrix of N(0,1)                #Real continouse realationship
    z = f(x, y) + eps                             #Data with N(0,1) noise

    return x, y, z
