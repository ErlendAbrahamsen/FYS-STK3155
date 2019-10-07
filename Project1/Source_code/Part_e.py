from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from Part_a import *

def LassoCV(x, y, z, f, kf, n=5, alpha=1e-1):
    """
    Returns (float) average mse, bias, variance on test and train set from Kfold algorithm.
    Inputs: x,y,z,f arraylike with sklearn Kfold object (kf).
            n and alpha are optional arguments.
    """

    x, y, z, f = (np.ravel(x), np.ravel(y),
                  np.ravel(z), np.ravel(f))

    m = 0                                                 #loop length
    MSE_train, var_train, bias_train = 0, 0, 0
    MSE_test, var_test, bias_test = 0, 0, 0
    true_MSE = 0
    for train_index, test_index in kf.split(x):           #Looping trough Kfolds
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        z_train, z_test = z[train_index], z[test_index]
        f_train, f_test = f[train_index], f[test_index]

        X = DesignMatrix(np.ravel(x_train),
                         np.ravel(y_train), n=n)

        reg = Lasso(alpha=alpha, max_iter=1e4, fit_intercept=False)
        coef = reg.fit(X, z_train).coef_
        z_train_pred = np.dot(X, coef)
        z_test_pred = np.ravel(Polynom(x_test, y_test, coef, n=n))

        m += 1
        train_mean = np.full(len(z_train), np.mean(z_train_pred)) #fix dimensions for scikit MSE
        MSE_train += mean_squared_error(z_train, z_train_pred)
        var_train += np.var(z_train_pred)
        bias_train += mean_squared_error(f_train, train_mean)

        test_mean = np.full(len(z_test), np.mean(z_test_pred))
        MSE_test += mean_squared_error(z_test, z_test_pred)
        var_test += np.var(z_test_pred)
        bias_test += mean_squared_error(z_test, test_mean)

        true_MSE += mean_squared_error(f_test, z_test_pred)

    return_ = (1/m*MSE_train, 1/m*MSE_test,
               1/m*var_train, 1/m*var_test,
               1/m*bias_train, 1/m*bias_test,
               1/m*true_MSE)

    return return_

if __name__ == '__main__':
    #Resampling and validating for a range of alpha values
    start, points = 1e-3, 5
    A = np.array([start*10**i for i in range(points)]) #[1e-3, ..., 1e+1]

    N, k = np.arange(1, 12), 5

    dim = (len(N), len(A))
    avg_MSE_train, avg_MSE_test =  np.zeros(dim), np.zeros(dim)
    avg_var_train, avg_var_test = np.zeros(dim), np.zeros(dim)
    avg_bias_train, avg_bias_test = np.zeros(dim), np.zeros(dim)
    avg_true_MSE = np.zeros(dim)

    kf = KFold(n_splits=k, shuffle=True)
    for i in range(len(N)):
        n = N[i]
        for j in range(len(A)):
            alpha = A[j]

            metrics = LassoCV(x, y, z, f, kf, n=n, alpha=alpha)
            avg_MSE_train[i][j], avg_MSE_test[i][j] = metrics[:2]
            avg_var_train[i][j], avg_var_test[i][j] = metrics[2:4]
            avg_bias_train[i][j], avg_bias_test[i][j] = metrics[4:6]
            avg_true_MSE[i][j] = metrics[-1]

i = 0 # alpha index
plt.suptitle("True Complexity-MSE tradeoff")
plt.plot(N, np.log10(avg_true_MSE[:,i]), label="mse(f, z_test_prediction), alpha=%.1e" % A[i])
plt.plot(N, np.log10(avg_true_MSE[:,i+1]), label="mse(f, z_test_prediction), alpha=%.1e" % A[i+1])
plt.xlabel("Polynomial degree (Complexity)"), plt.ylabel("log10(MSE)")
plt.grid(), plt.legend()
plt.show()

#Most optimal true MSE
n = 8
print(avg_true_MSE[n-1, 0])

plt.suptitle("Test bias")
plt.plot(N, np.log10(avg_bias_test[:,i]), label="Bias alpha=%.1e" % A[i])
plt.plot(N, np.log10(avg_bias_test[:,i+1]), label="Bias alpha=%.1e" % A[i+1])
plt.xlabel("Polynomial degree (Complexity)"), plt.ylabel("log10()")
plt.grid(), plt.legend()
plt.show()

plt.suptitle("Test Variance")
plt.plot(N, np.log10(avg_var_test[:,i]), label="Bias /w alpha=%.1e" % A[i])
plt.plot(N, np.log10(avg_var_test[:,i+1]), label="Bias /w alpha=%.1e" % A[i+1])
plt.xlabel("Polynomial degree (Complexity)"), plt.ylabel("log10()")
plt.grid(), plt.legend()
plt.show()

n = 4 #Polynom start degree
plt.suptitle("MSE alpha dependence")
plt.plot(np.log10(A), np.log10(avg_MSE_test[n-1,:]), label="Test-data n=%d" % int(n))
plt.plot(np.log10(A), np.log10(avg_MSE_test[n,:]), label="Test-data n=%d" % int(n+1))
plt.plot(np.log10(A), np.log10(avg_MSE_test[n+1,:]), label="Test-data n=%d" % int(n+2))
plt.plot(np.log10(A), np.log10(avg_MSE_test[n+5,:]), label="Test-data n=%d" % int(n+6))
plt.xlabel("log10(alpha)"), plt.ylabel("log10(MSE)")
plt.grid(), plt.legend()
plt.show()

#Verify specific graph
print(avg_MSE_test[3,0])
