from Part_a import *
from Part_b import KFoldCross, Metrics
from sklearn.metrics import mean_squared_error

def CrossPredict(method, X_train, X_test, Y_train, Y_test, Z_train, n=5, Lambda=1e-2):
    """
    For all groups:
    Train on training data, predict on both train and test data.

    method: Name of method. Options are "OLS", "Ridge", "Lasso"
    *_train: List of training arrays (groups)
    *_test: List of testing arrays (groups)
    n: optional Polynomial degree with defualt 5
    Lambda: optional with default 1e-2
    """

    Z_pred_train, Z_pred_test = [], []
    for x_train, x_test, y_train, y_test, z_train in zip(X_train, X_test, Y_train, Y_test, Z_train):
        X = DesignMatrix(x_train, y_train, n=n)
        X_ = DesignMatrix(x_test, y_test, n=n)

        if method == "OLS":
            beta = OLS(X, z_train)

        elif method == "Ridge":
            from Part_d import Ridge
            beta = Ridge(X, z_train, Lambda=Lambda)

        Z_pred_train.append(np.dot(X, beta))
        Z_pred_test.append(Polynom(x_test, y_test, beta, n=n))

    return Z_pred_train, Z_pred_test

def TradeoffMetrics(Z_train, Z_pred_train, Z_test, Z_pred_test, F_train, F_test):
    """
    Return avg- MSE, variance and bias over all groups for both
    training and testing predictions.
    Also return MSE between true f and the test prediction.

    *_pred_train: List of array's predictions on training data
    *_pred_test: List of array's predictions on testing data
    F_train, F_test: List of array's from true data realationship
    """

    MSE_train = np.mean([MSE(z, z_) for z, z_ in zip(Z_train, Z_pred_train)])
    var_train = np.mean([np.var(z_) for z_ in Z_pred_train])
    bias_train = np.mean([MSE(f, np.mean(z_)) for f, z_ in zip(F_train, Z_pred_train)])

    MSE_test = np.mean([MSE(z, z_) for z, z_ in zip(Z_test, Z_pred_test)])
    var_test = np.mean([np.var(z_) for z_ in Z_pred_test])
    bias_test = np.mean([MSE(f, np.mean(z_)) for f, z_ in zip(F_test, Z_pred_test)])

    true_MSE = np.mean([MSE(f, z_) for f, z_ in zip(F_test, Z_pred_test)])

    return_ = ( MSE_train, MSE_test, var_train, var_test,
                bias_train, bias_test, true_MSE )

    return return_

if __name__ == '__main__':
    ##OLS regression of degree 1...n with plot of model complexity vs MSE, bias and variance
    N, k = np.arange(1, 11), 5
    X_train, X_test, Y_train, Y_test, Z_train, Z_test, F_train, F_test = (
    KFoldCross(x, y, z, f, k=k) )

    avg_MSE_train, avg_MSE_test = [], []
    avg_var_train, avg_var_test = [], []
    avg_bias_train, avg_bias_test = [], []
    avg_true_MSE = []
    for n in N:
        Z_pred_train, Z_pred_test = CrossPredict("OLS", X_train, X_test, Y_train, Y_test, Z_train, n=n)

        MSE_train, MSE_test, var_train, var_test, bias_train, bias_test, true_MSE = (
        TradeoffMetrics(Z_train, Z_pred_train, Z_test, Z_pred_test, F_train, F_test) )

        avg_MSE_train.append(MSE_train), avg_MSE_test.append(MSE_test)
        avg_var_train.append(var_train), avg_var_test.append(var_test)
        avg_bias_train.append(bias_train), avg_bias_test.append(bias_test)
        avg_true_MSE.append(true_MSE)


    plt.title("Complexity-MSE tradeoff")
    plt.plot(N, np.log10(avg_MSE_train), label="Train-data")
    plt.plot(N, np.log10(avg_MSE_test), label="Test-data")
    plt.xlabel("Polynomial degree (Complexity)"), plt.ylabel("log10(MSE)")
    plt.grid(), plt.legend()
    plt.show()

    plt.title("Variances")
    plt.plot(N, np.log10(avg_var_train), label="Train-Variance")
    plt.plot(N, np.log10(avg_var_test), label="Test-Variance")
    plt.xlabel("Polynomial degree (Complexity)"), plt.ylabel("log10(var)")
    plt.grid(), plt.legend()
    plt.show()

    plt.title("Bias")
    plt.plot(N, np.log10(avg_bias_train), label="Train-Bias")
    plt.plot(N, np.log10(avg_bias_test), label="Test-Bias")
    plt.xlabel("Polynomial degree (Complexity)"), plt.ylabel("log10(bias)")
    plt.grid(), plt.legend()
    plt.show()

    plt.title("True Complexity-MSE tradeoff")
    plt.plot(N, np.log10(avg_true_MSE), label="mse(f, z_test_prediction)")
    plt.xlabel("Polynomial degree (Complexity)"), plt.ylabel("log10(MSE)")
    plt.grid(), plt.legend()
    plt.show()

    #Get best test mse from MSE tradeoff graph
    n = 4
    print(avg_MSE_test[n-1], "\n", avg_true_MSE[n-1])

    #Check MSE = bias^2 + var + sigma^2
    MSE = np.mean(avg_MSE_test)
    terms = np.mean(np.array(avg_bias_test)+np.array(avg_var_test)+1)
    print("MSE-terms: %.3f" % (MSE-terms))
