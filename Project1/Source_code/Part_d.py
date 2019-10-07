from Part_a import *
from Part_b import KFoldCross, Metrics
from Part_c import CrossPredict, TradeoffMetrics
import pandas as pd

def Ridge(X, z, Lambda=1e-2):
    """
    Return coefficients based on the design matrix X, and the lambda cost parameter
    """

    XTX = X.T.dot(X)
    I = np.identity(len(XTX))
    beta = np.linalg.inv(XTX + Lambda*I).dot(X.T).dot(z)

    return beta

if __name__ == '__main__':
    #Resampling and validating for a range of lambda values
    start, points = 1e-6, 8
    L = np.array([start*10**i for i in range(points)]) #[1e-6, ..., 1e+1]


    N, k = np.arange(1, 12), 5
    X_train, X_test, Y_train, Y_test, Z_train, Z_test, F_train, F_test = (
    KFoldCross(x, y, z, f, k=k) )

    dim = (len(N), len(L))
    avg_MSE, avg_R2 = np.zeros(dim), np.zeros(dim)

    avg_MSE_train, avg_MSE_test =  np.zeros(dim), np.zeros(dim)
    avg_var_train, avg_var_test = np.zeros(dim), np.zeros(dim)
    avg_bias_train, avg_bias_test = np.zeros(dim), np.zeros(dim)
    avg_true_MSE = np.zeros(dim)

    for i in range(len(N)):
        n = N[i]
        for j in range(len(L)):
            Lambda = L[j]

            Z_pred_train, Z_pred_test = (
            CrossPredict("Ridge", X_train, X_test, Y_train, Y_test, Z_train, n=n, Lambda=Lambda) )

            MSE_train, MSE_test, var_train, var_test, bias_train, bias_test, true_MSE = (
            TradeoffMetrics(Z_train, Z_pred_train, Z_test, Z_pred_test, F_train, F_test) )

            avg_MSE_train[i][j], avg_MSE_test[i][j] = MSE_train, MSE_test
            avg_var_train[i][j], avg_var_test[i][j] = var_train, var_test
            avg_bias_train[i][j], avg_bias_test[i][j] = bias_train, bias_test
            avg_true_MSE[i][j] = true_MSE

    i = 0 #Lambda start index
    plt.suptitle("Complexity-MSE tradeoff")
    plt.plot(N, np.log10(avg_MSE_train[:,i]), label="Train-data lambda=%.1e" % L[i])
    plt.plot(N, np.log10(avg_MSE_test[:,i]), label="Test-data lambda=%.1e" % L[i])
    plt.plot(N, np.log10(avg_MSE_train[:,i+2]), label="Train-data lambda=%.1e" % L[i+2])
    plt.plot(N, np.log10(avg_MSE_test[:,i+2]), label="Test-data lambda=%.1e" % L[i+2])
    plt.plot(N, np.log10(avg_MSE_train[:,i+3]), label="Train-data lambda=%.1e" % L[i+3])
    plt.plot(N, np.log10(avg_MSE_test[:,i+3]), label="Test-data lambda=%.1e" % L[i+3])
    plt.xlabel("Polynomial degree (Complexity)"), plt.ylabel("log10(MSE)")
    plt.grid(), plt.legend()
    plt.show()

    plt.suptitle("Test bias")
    plt.plot(N, np.log10(avg_bias_test[:,i]), label="Bias /w lambda=%.1e" % L[i])
    plt.plot(N, np.log10(avg_bias_test[:,i+3]), label="Bias /w lambda=%.1e" % L[i+3])
    plt.plot(N, np.log10(avg_bias_test[:,i+5]), label="Bias /w lambda=%.1e" % L[i+5])
    plt.xlabel("Polynomial degree (Complexity)"), plt.ylabel("log10()")
    plt.grid(), plt.legend()
    plt.show()

    plt.suptitle("Test Variance")
    plt.plot(N, np.log10(avg_var_test[:,i]), label="Bias /w lambda=%.1e" % L[i])
    plt.plot(N, np.log10(avg_var_test[:,i+3]), label="Bias /w lambda=%.1e" % L[i+3])
    plt.plot(N, np.log10(avg_var_test[:,i+5]), label="Bias /w lambda=%.1e" % L[i+5])
    plt.xlabel("Polynomial degree (Complexity)"), plt.ylabel("log10()")
    plt.grid(), plt.legend()
    plt.show()

    n = 4 #Polynom start degree
    plt.suptitle("MSE Lambda dependence")
    plt.plot(np.log10(L), np.log10(avg_MSE_test[n-1,:]), label="Test-data n=%d" % int(n))
    plt.plot(np.log10(L), np.log10(avg_MSE_test[n,:]), label="Test-data n=%d" % int(n+1))
    plt.plot(np.log10(L), np.log10(avg_MSE_test[n+1,:]), label="Test-data n=%d" % int(n+2))
    plt.plot(np.log10(L), np.log10(avg_MSE_test[n+5,:]), label="Test-data n=%d" % int(n+6))
    plt.xlabel("log10(Lambda)"), plt.ylabel("log10(MSE)")
    plt.grid(), plt.legend()
    plt.show()

    plt.title("True Complexity-MSE tradeoff")
    plt.plot(N, np.log10(avg_true_MSE[:,i+2]), label="mse(f, z_test_prediction), Lambda=%.1e" % L[i+2])
    plt.plot(N, np.log10(avg_true_MSE[:,i+3]), label="mse(f, z_test_prediction), Lambda=%.1e" % L[i+3])
    plt.plot(N, np.log10(avg_true_MSE[:,i+4]), label="mse(f, z_test_prediction), Lambda=%.1e" % L[i+4])
    plt.xlabel("Polynomial degree (Complexity)"), plt.ylabel("log10(MSE)")
    plt.grid(), plt.legend()
    plt.show()

    #Verify MSE Lambda dependence graph
    print("\n\n", N[4], L[3]) #n=5, L=1e-3
    print(avg_MSE_test[4][3])


    print("\n\n", N[3], L[3]) #n=4, L=1e-3
    print(avg_true_MSE[3][3])
