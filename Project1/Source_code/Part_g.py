from Part_a import *
from Part_d import Ridge
from imageio import imread
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import scale
from time import time

def CV(x, y, z, kf, n=5, alpha=1e-2):
    """
    Return average mse, bias, variance from OLS, Lasso, Ridge method with Kfold
    crossvalidation (sklearn).
    Input: x, y, z arraylike, Kfold object (kf) and optional arguments
           n, alpha (alpha in Lasso, Ridge).
    """

    m = 0                                                     #loop length
    MSE_OLS, var_OLS, bias_OLS = 0, 0, 0
    MSE_Ridge, var_Ridge, bias_Ridge = 0, 0, 0
    MSE_Lasso, var_Lasso, bias_Lasso = 0, 0, 0

    for train_index, test_index in kf.split(x):               #Looping trough Kfolds
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            z_train, z_test = z[train_index], z[test_index]

            X = DesignMatrix(x_train, y_train, n=n)[:,1:]

            model1 = LinearRegression()
            model2 = Ridge(alpha=alpha, max_iter=1e3)
            model3 = Lasso(alpha=alpha, max_iter=1e3)

            coef = [model1.fit(X, z_train).coef_,
                    model2.fit(X, z_train).coef_,
                    model3.fit(X, z_train).coef_]

            #Predictions on test data
            b_OLS, b_Ridge, b_Lasso = coef
            z_OLS, z_Ridge, z_Lasso = model1.intercept_, model2.intercept_, model3.intercept_
            for i in range(1, n+1):
                s = int(i*(i+1)/2) - 1
                for j in range(i+1):
                    z_OLS +=   b_OLS[s+j]*x_test**(i-j)*y_test**(j)
                    z_Ridge += b_Ridge[s+j]*x_test**(i-j)*y_test**(j)
                    z_Lasso += b_Lasso[s+j]*x_test**(i-j)*y_test**(j)

            #Metrics
            m += 1
            mean_OLS = np.full(len(z_test), np.mean(z_OLS))
            MSE_OLS += mean_squared_error(z_test, z_OLS)
            var_OLS += np.var(z_OLS)
            bias_OLS += mean_squared_error(z_test, mean_OLS)

            mean_Ridge = np.full(len(z_test), np.mean(z_OLS))
            mean_Ridge = np.full(len(z_test), np.mean(z_Ridge))
            MSE_Ridge += mean_squared_error(z_test, z_Ridge)
            var_Ridge += np.var(z_Ridge)
            bias_Ridge += mean_squared_error(z_test, mean_Ridge)

            mean_Lasso = np.full(len(z_test), np.mean(z_Lasso))
            mean_Lasso = np.full(len(z_test), np.mean(z_Lasso))
            MSE_Lasso += mean_squared_error(z_test, z_Lasso)
            var_Lasso += np.var(z_Lasso)
            bias_Lasso += mean_squared_error(z_test, mean_Lasso)

    return_ = (1/m*MSE_OLS, 1/m*MSE_Ridge, 1/m*MSE_Lasso,
               1/m*var_OLS, 1/m*var_Ridge, 1/m*var_Lasso,
               1/m*bias_OLS, 1/m*bias_Ridge, 1/m*bias_Lasso)

    return return_

def RandomReduce(rows, cols, x, y, z, p=int(1e+2)):
    """
    Returns 1darray of p random points from arraylike data sets x, y, z.
    """

    #Reduce the terrain data to a random sub-set of p points
    np.random.seed(1)
    r = np.arange(rows*cols)
    np.random.shuffle(r)
    r = r[:p]
    x = np.array([np.ravel(x)[i] for i in r])   #Shuffle same indices on x, y and z
    y = np.array([np.ravel(y)[i] for i in r])
    z = np.array([np.ravel(z)[i] for i in r])

    return x, y, z

if __name__ == '__main__':
    #Load real terrain data
    z = imread("SRTM_data_Norway_1.tif")
    rows, cols = np.shape(z)
    x, y = np.linspace(0, cols, cols), np.linspace(0, rows, rows)
    x, y = np.meshgrid(x, y)

    x, y, z = RandomReduce(rows, cols, x, y, z, p=int(1e3))
    #x, y, z = 1/1000*x, 1/1000*y, 1/1000*z #Scaling data for convergence purposes
    x, y, z = scale(x), scale(y), scale(z)  #Scaling data for convergence purposes

    alpha, N, k = 1e-2, np.arange(1, 11), 5
    kf = KFold(n_splits=k, shuffle=True)

    MSE_ols, MSE_ridge, MSE_lasso = [], [], []
    var_ols, var_ridge, var_lasso = [], [], []
    bias_ols, bias_ridge, bias_lasso = [], [], []


    for n in N:
        CV(x, y, z, kf, n=n, alpha=alpha)
        (MSE_OLS, MSE_Ridge, MSE_Lasso,
         var_OLS, var_Ridge, var_Lasso,
         bias_OLS, bias_Ridge, bias_Lasso) = CV(x, y, z, kf, n=n, alpha=alpha)

        MSE_ols.append(MSE_OLS)
        MSE_ridge.append(MSE_Ridge)
        MSE_lasso.append(MSE_Lasso)

        var_ols.append(var_OLS)
        var_ridge.append(var_Ridge)
        var_lasso.append(var_Lasso)

        bias_ols.append(bias_OLS)
        bias_ridge.append(bias_Ridge)
        bias_lasso.append(bias_Lasso)

    print(MSE_ols)
    plt.suptitle("Complexity-MSE tradeoff")
    plt.title("alpha=%.1e" % alpha)
    plt.plot(N, np.log10(MSE_ols), label = "OLS")
    plt.plot(N, np.log10(MSE_ridge), label = "Ridge")
    plt.plot(N, np.log10(MSE_lasso), label = "Lasso")
    plt.xlabel("Polynomial degree (Complexity)"), plt.ylabel("log10(MSE)")
    plt.grid(), plt.legend()
    plt.show()

    plt.suptitle("Bias")
    plt.title("alpha=%.1e" % alpha)
    plt.plot(N, np.log10(bias_ols), label = "OLS")
    plt.plot(N, np.log10(bias_ridge), label = "Ridge")
    plt.plot(N, np.log10(bias_lasso), label = "Lasso")
    plt.xlabel("Polynomial degree (Complexity)"), plt.ylabel("log10(MSE)")
    plt.grid(), plt.legend()
    plt.show()

    plt.suptitle("Variance")
    plt.title("alpha=%.1e" % alpha)
    plt.plot(N, np.log10(var_ols), label = "OLS")
    plt.plot(N, np.log10(var_ridge), label = "Ridge")
    plt.plot(N, np.log10(var_lasso), label = "Lasso")
    plt.xlabel("Polynomial degree (Complexity)"), plt.ylabel("log10(MSE)")
    plt.grid(), plt.legend()
    plt.show()
