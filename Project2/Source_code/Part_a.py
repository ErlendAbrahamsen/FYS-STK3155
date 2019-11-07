import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Project1_methods import frankeData

def creditDataReduction(creditData):
    """
    Check for wrong values and reduce
    data set.
    """

    #Check if X2, X3, X4, X6-X11 data is in Expected
    X = ["X2", "X3", "X4"]
    Expected = [[1, 2], [1, 2, 3, 4], [1, 2, 3]]
    for i in range(6,12):
        X.append("X%d" % i) #X6-X11
        exp = np.concatenate([ [-1], np.arange(1,10) ] ) #[-1,1,2,...,9]
        Expected.append(exp)

    FAILS = []  #List for containing index with outlier
    for Xi, Yi in zip(X, Expected):
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
    creditData = creditData.drop(ids)   #Dropping id's with outlier

    return creditData



#CLS credit card data mapping X -> y
creditData = pd.read_excel("credit.xls")
creditData = creditDataReduction(creditData) #Remove outliers

Data = np.array(creditData)[1:, :]           #Excluding row 0 with strings.
row, col = np.shape(Data)
X = np.ones((row, col))

y = Data[:,-1].reshape(-1,1)                 #y column vector
X[:, 1:] = Data[:, :-1]                      #Design matrix (n, 24)

if __name__ == '__main__':
    #Test of y values in range (0 or 1)
    sum = np.sum(y==0) + np.sum(y==1)
    if sum != y.size: print("y not in range")

    #X, y shapes
    print("y shape:", np.shape(y))
    print("X shape:", np.shape(X), "\n")

    #Precentage of creditable and non-creditable costumers
    cred = np.sum(y==0)/y.size
    non_cred = np.sum(y==1)/y.size
    print("Creditable: %.3f %%" % cred)
    print("Non-creditable: %.3f %%" % non_cred)
