import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn import preprocessing
from Part_a import X, y
from Part_c import NeuralNetwork, to_categorical_numpy, accuracy_score
from Project1_methods import frankeData, MyPlot, f

def NeuralNetwork_unit_test(n_hidden_neurons=50, n_hidden_layers=3, epochs=100, eta=1e-2, lmbd=0):
    """
    Test 100% accuracy on
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] in,
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] out.
    Prints nothing if passed!
    """

    X = np.zeros((10, 1))
    X[:5] = 1

    y_test = np.zeros((10, 1))
    y_test[5:] = 1
    y_test = to_categorical_numpy(np.array(np.ravel(y_test), dtype=np.int))

    nn = NeuralNetwork(X, y_test, n_hidden_neurons=n_hidden_neurons, n_hidden_layers=n_hidden_layers,
                                                     epochs=epochs, eta=eta, lmbd=lmbd, batch_size=5)
    nn.train()
    y_predict = nn.predict(X)
    y_test = np.ravel(y_test[:, -1])

    assert np.array_equal(y_test, y_predict)

def scikit_classifier_test(X, y, etas, lmbds, activation):
    """
    Test scikit mlp on credit data for comparison
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)
    h_layers = (100, 100, 100, 100)

    NN = np.zeros((len(etas), len(lmbds)), dtype=object)
    test_scores = np.zeros((len(etas), len(lmbds)))
    train_scores = np.zeros((len(etas), len(lmbds)))

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
            nn = MLPClassifier(hidden_layer_sizes=h_layers, activation=activation,
                                learning_rate_init=eta, alpha=lmbd, max_iter=200)
            nn.fit(X_train, y_train)
            NN[i, j] = nn

            test_scores[i, j] = nn.score(X_test, y_test)
            train_scores[i, j] = nn.score(X_train, y_train)

    return NN, test_scores, train_scores

def scikit_regressor_test(X, z, etas, lmbd, activation):
    """
    Test scikit mlp on frankeData for comparison
    """

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, shuffle=True, random_state=1)
    h_layers = (50, 50)

    NN = np.zeros((len(etas), len(lmbds)), dtype=object)
    test_mse = np.zeros((len(etas), len(lmbds)))
    train_mse = np.zeros((len(etas), len(lmbds)))

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
            nn = MLPRegressor(hidden_layer_sizes=h_layers, activation=activation, verbose=True,
                                learning_rate_init=eta, alpha=lmbd, max_iter=200, early_stopping= False)
            nn.fit(X_train, z_train)
            NN[i, j] = nn

            z_test_pred = nn.predict(X_test)
            z_train_pred = nn.predict(X_train)

            f_test = f(X_test[:, 0], X_test[:, 1])
            f_train = f(X_train[:, 0], X_train[:, 1])

            test_mse[i, j] = mean_squared_error(np.ravel(f_test), np.ravel(z_test_pred))
            train_mse[i, j] = mean_squared_error(np.ravel(f_train), np.ravel(z_train_pred))

    return NN, test_mse, train_mse

if __name__ == '__main__':
    NeuralNetwork_unit_test()

    X = X[:, 1:]
    X = preprocessing.StandardScaler().fit(X).transform(X)
    y = to_categorical_numpy(np.array(np.ravel(y), dtype=np.int))

    #scikit CLASSIFIER
    etas, lmbds = np.logspace(-5, 0, 6), np.logspace(-5, 0, 6)
    NN, test_scores, train_scores = scikit_classifier_test(X, y, etas, lmbds, "logistic")

    fig, ax = plt.subplots()    #Train acc plot
    sns.heatmap(train_scores, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Scikit Training accuracy scores")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    fig, ax = plt.subplots()    #Test acc plot
    sns.heatmap(test_scores, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Scikit Testing accuracy scores")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    #scikit REGRESSOR
    X, z = frankeData()
    NN, test_mse, train_mse = scikit_regressor_test(X, np.ravel(z), etas, lmbds, "tanh")

    fig, ax = plt.subplots()    #Train mse plot
    sns.heatmap(train_mse, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Scikit Training MSE")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    fig, ax = plt.subplots()    #Test mse plot
    sns.heatmap(test_mse, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Scikit Testing MSE")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()
