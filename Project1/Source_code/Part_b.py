from Part_a import *

def KFoldCross(x, y, z, f, k=5):
    """
    Divide training data into k-1 folds and test data into the remaining fold.
    Go trough all partitions and store them in lists of ndarrays.
    """

    #Shuffle the data
    p = int(x.size/k)                           #Length of each fold
    r = np.arange(x.size)                       #Indices of x/y/z
    np.random.shuffle(r)                        #Random shuffle of indices
    x = np.array([np.ravel(x)[i] for i in r])   #Shuffle same indices on x, y and z
    y = np.array([np.ravel(y)[i] for i in r])
    z = np.array([np.ravel(z)[i] for i in r])
    f = np.array([np.ravel(f)[i] for i in r])

    X_train, X_test, Y_train, Y_test, Z_train, Z_test, F_train, F_test = (
    [], [], [], [], [], [], [], [] )

    #Cycle testing data trough k different folds. Add to lists of 1darry's
    for i in range(k):
        x_test, y_test = x[p*i:p*(i+1)], y[p*i:p*(i+1)]  # p points
        z_test, f_test = z[p*i:p*(i+1)], f[p*i:p*(i+1)]

        x_train = np.append(x[:p*i], x[p*(i+1):], axis=0)                # n - p points
        y_train = np.append(y[:p*i], y[p*(i+1):], axis=0)
        z_train = np.append(z[:p*i], z[p*(i+1):], axis=0)
        f_train = np.append(f[:p*i], f[p*(i+1):], axis=0)

        X_train.append(x_train), X_test.append(x_test)
        Y_train.append(y_train), Y_test.append(y_test)
        Z_train.append(z_train), Z_test.append(z_test)
        F_train.append(f_train), F_test.append(f_test)

    return X_train, X_test, Y_train, Y_test, Z_train, Z_test, F_train, F_test

def Metrics(Z_test, Z_pred):
    """
    Returns average MSE and R2 score from list of data arrays.
    """

    MSE_sum, R2_sum, n = 0, 0, len(Z_test)
    for i in range(n):
        z_test, z_pred = np.ravel(Z_test[i]), np.ravel(Z_pred[i])
        MSE_sum += MSE(z_test, z_pred)
        R2_sum += R2(z_test, z_pred)

    return 1/n*MSE_sum, 1/n*R2_sum

if __name__ == '__main__':
    #Fitting all partitions to OLS model
    N, k = np.arange(1, 11), 3 #pol. degree and num. of folds
    X_train, X_test, Y_train, Y_test, Z_train, Z_test, F_train, F_test = (
    KFoldCross(x, y, z, f, k=k) )

    for n in N:
        X = [DesignMatrix(x, y, n=n) for x, y in zip(X_train, Y_train)]
        B = [OLS(x, z) for x, z in zip(X, Z_train)]
        Z_pred = [Polynom(x, y, beta, n=n) for x, y, beta in zip(X_test, Y_test, B)]

        #Partition scores
        avg_MSE, avg_R2 = Metrics(F_test, Z_pred)
        print("(n,k)=(%d,%d): avg_MSE=%.4g, avg_R2=%.4g " % (n, k, avg_MSE, avg_R2))
