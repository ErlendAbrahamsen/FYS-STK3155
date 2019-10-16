import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Project1_methods import frankeData

def creditDataReduction(creditData):
    """
    Check for wrong values and reduce
    data set.
    """

    #Check X2, X3, X4, X6-X11
    X = ["X2", "X3", "X4"]
    ExpectedOut = [[1, 2], [1, 2, 3, 4], [1, 2, 3]]
    for i in range(6,12):
        X.append("X%d" % i) #X6-X11
        exp = np.concatenate([ [-1], np.arange(1,10) ] ) #[-1,1,2,...,9]
        ExpectedOut.append(exp)

    FAILS = []  #List for containing index with outlier
    for Xi, Yi in zip(X, ExpectedOut):
        test = np.isin(creditData[Xi][1:], Yi)
        fails = np.where(test == False)[0]

        if len(fails)>0:
            FAILS.append(fails)


    #Check creditData[X1, X5, X12-X17, X18-X23] >= 0
    i = np.concatenate([ [1, 5], np.arange(12, 18), np.arange(18, 24) ] )
    X = ["X%d" % i for i in i]
    for Xi in X:
        j = 0
        for Yi in creditData[Xi][1:]:
            if Yi < 0:
                FAILS.append([j])
            j += 1

    FAILS = np.concatenate(FAILS)
    ids = FAILS + 1
    creditData = creditData.drop(ids)

    return creditData

#CLS credit card data mapping X -> y
creditData = pd.read_excel("credit.xls")
creditData = creditDataReduction(creditData) #Remove outliers

X = np.array(creditData)[1:, :]              #Convert pandas dataframe to ndarray
y = X[:,-1].reshape(-1,1)                    #y vector
X = X[:, :-1]                                #X as (n, 23) matrix. (X1,X2,...,X23)

#Franke data mapping (xf, yf) -> fData
xf, yf, fData = frankeData()
