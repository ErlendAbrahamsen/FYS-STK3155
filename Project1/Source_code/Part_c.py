from Part_a import *
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    #Make data.
    n = 40
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    x, y = np.meshgrid(x, y)

    ##OLS regression of degree 1...n with plot of model complexity
    N, test_size = np.arange(1, 11), 0.2
    Noise = True
    MSE_train, MSE_test = [], []
    
    for n in N:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

        X_train, X_test = [DesignMatrix(np.ravel(x_train), np.ravel(y_train), n=n),
                            DesignMatrix(np.ravel(x_test), np.ravel(y_test), n=n)]

        z_train, z_test = [FrankeFunction(np.ravel(x_train), np.ravel(y_train), Noise=Noise),
                            FrankeFunction(np.ravel(x_test), np.ravel(y_test), Noise=Noise)]

        beta_train, beta_test = OLS(X_train, z_train), OLS(X_test, z_test)
        z_OLS_train, z_OLS_test = [Polynom(x_train, y_train, beta_train, n=n),
                                    Polynom(x_test, y_test, beta_test, n=n)]

        MSE_train.append(MSE(np.ravel(z_train), np.ravel(z_OLS_train)))
        MSE_test.append(MSE(np.ravel(z_test), np.ravel(z_OLS_test)))

    plt.plot(N, MSE_train, label="Training-set"), plt.plot(N, MSE_test, label="Testing-set")
    plt.title("Bias vs variance tradeoff")
    plt.xlabel("Polynomial degree"), plt.ylabel("MSE")
    plt.grid()
    plt.legend()
    plt.show()
