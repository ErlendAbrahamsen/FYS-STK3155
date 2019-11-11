from Part_c import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Project1_methods import frankeData, MyPlot, f

def regressor_grid_search(X, z, f, etas, lmbds, activation="sigmoid", cost="mse"):
    """
    Loops over different value combinations from etas and lmbds array.
    Returns ndarray of NeuralNetworks (NN) and mse's
    for each eta and lmbd combination.
    """

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, shuffle=True, random_state=1)

    NN = np.zeros((len(etas), len(lmbds)), dtype=object)
    test_mse = np.zeros((len(etas), len(lmbds)))
    train_mse = np.zeros((len(etas), len(lmbds)))

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
            nn = NeuralNetwork(X_train, z_train, type="regression", eta=eta, lmbd=lmbd, n_hidden_neurons=100,
                            n_hidden_layers=4, activation=activation, cost=cost, epochs=200)
            nn.train()
            NN[i, j] = nn

            z_test_pred = nn.predict(X_test)
            z_train_pred = nn.predict(X_train)

            #True fanke
            f_test = f(X_test[:, 0], X_test[:, 1])
            f_train = f(X_train[:, 0], X_train[:, 1])

            test_mse[i, j] = mean_squared_error(np.ravel(f_test), np.ravel(z_test_pred))
            train_mse[i, j] = mean_squared_error(np.ravel(f_train), np.ravel(z_train_pred))

    return NN, test_mse, train_mse

if __name__ == '__main__':
    #Sigmoid, mse, grid search /w plots
    X, z = frankeData()
    activation, cost = "sigmoid", "mse"
    etas, lmbds = np.logspace(-5, 0, 6), np.logspace(-5, 0, 6)
    NN, test_mse, train_mse = regressor_grid_search(X, z, f, etas, lmbds, activation=activation, cost=cost)

    fig, ax = plt.subplots()    #MSE(X_train, f_train)
    sns.heatmap(train_mse, annot=True, ax=ax, cmap="viridis", fmt=".3g")
    ax.set_title("%s with %s cost: Training MSE" % (activation, cost))
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    fig, ax = plt.subplots()    #MSE(X_test, f_test)
    sns.heatmap(test_mse, annot=True, ax=ax, cmap="viridis", fmt=".3g")
    ax.set_title("%s with %s cost: Testing MSE" % (activation, cost))
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    #Best result plot (tanh mse)
    X, z = frankeData()
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, shuffle=True, random_state=1)
    nn = NeuralNetwork(X_train, z_train, type="regression", eta=1e-4, lmbd=1e-1, n_hidden_neurons=50,
               n_hidden_layers=2, epochs=200, activation="tanh", cost="mse")
    nn.train()
    z_pred = nn.predict(X)

    x, y = X[:, 0].reshape(50, 50), X[:, 1].reshape(50, 50)
    z = np.ravel(z_pred).reshape(50, 50)

    MyPlot(x, y, z, title="NeuralNetwork regression")
    MyPlot(x, y, f(x, y), title="True franke")
    plt.show()
