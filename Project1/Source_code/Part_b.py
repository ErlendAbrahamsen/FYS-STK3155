import numpy as np
from sklearn.model_selection import train_test_split
from Part_a import FrankeFunction, DesignMatrix, OLS, Polynom, MSE, R2

def KFold(x, y, z, k=10):
    """
    Divide training data into k-1 folds and test data into the remaining fold.
    Go trough all partitions and store them in lists of ndarrays.
    """

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = [], [], [], [], [], []

    n = len(x)
    p = int(n/k)  #Length of each partition

    for i in range(k):
        x_test, y_test, z_test = x[p*i:p*(i+1)], y[p*i:p*(i+1)], z[p*i:p*(i+1)]  # p rows
        x_train = np.append(x[:p*i], x[p*(i+1):], axis=0)                        # n - p rows
        y_train = np.append(y[:p*i], y[p*(i+1):], axis=0)
        z_train = np.append(z[:p*i], z[p*(i+1):], axis=0)

        X_train.append(x_train), X_test.append(x_test)
        Y_train.append(y_train), Y_test.append(y_test)
        Z_train.append(z_train), Z_test.append(z_test)

    return X_train, X_test, Y_train, Y_test, Z_train, Z_test

def CrossVal(Z_test, Z_pred):
    """
    Crossvalidate our data.
    Returns avrage MSE and R2 score of data with k different partitions.
    """

    MSE_sum, R2_sum, n = 0, 0, len(Z_test)
    for i in range(n):
        z_test, z_pred = Z_test[i], Z_pred[i]
        MSE_sum += MSE(z_test, z_pred)
        R2_sum += R2(z_test, z_pred)

    return 1/n*MSE_sum, 1/n*R2_sum

if __name__ == '__main__':
    #Make data.
    n = 25
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)

    #Unit test ????
    #print(x.shape)
    #t = np.arange(10)
    #n = 4
    #p = int(t.size/n)
    #print(p)
    #for i in range(n):
    #    print(t[p*i:p*(i+1)])
    #    print(np.append(t[:p*i], t[p*(i+1):], axis=0))


    X_train, X_test, Y_train, Y_test, Z_train, Z_test = KFold(x,y,z)

    #Fitting all partitions to OLS model
    n = 5 #Polynom degree
    X, Z, BETA = [], [], []
    for xr, yr in zip(X_train, Y_train):
        xr, yr = np.ravel(xr), np.ravel(yr)
        X.append(DesignMatrix(xr, yr, n=n))
        Z.append(FrankeFunction(xr, yr))

    BETA = [OLS(x, z) for x, z in zip(X, Z)]
    Z_pred = [Polynom(x, y, beta) for x, y, beta in zip(X_test, Y_test, BETA)]

    #Partition scores
    avg_MSE, avg_R2 = CrossVal(Z_test, Z_pred)
    print(avg_MSE, avg_R2)

    #x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)
