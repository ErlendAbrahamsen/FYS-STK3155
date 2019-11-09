import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor, MLPClassifier
from Part_a import X, y
from Part_c import NeuralNetwork, to_categorical_numpy, accuracy_score
from Project1_methods import frankeData, MyPlot

def NeuralNetwork_unit_test():
    """
    Test 100% accuracy on
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] in,
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] out.
    """

    X = np.zeros((10, 1))
    X[:5] = 1

    y_test = np.zeros((10, 1))
    y_test[5:] = 1
    y_test = to_categorical_numpy(np.array(np.ravel(y_test), dtype=np.int))

    nn = NeuralNetwork(X, y_test, n_hidden_neurons=50, n_hidden_layers=3, epochs=100, eta=0.01, lmbd=0, batch_size=5)
    nn.train()
    y_predict = nn.predict(X)
    y_test = np.ravel(y_test[:, -1])

    assert np.array_equal(y_test, y_predict)

def scikit_classifier_test(X, y, etas, lmbds, activation):
    """
    Test scikit mlp on credit data for comparison
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    h_layers = (50, 50)

    NN = np.zeros((len(etas), len(lmbds)), dtype=object)
    test_scores = np.zeros((len(etas), len(lmbds)))
    train_scores = np.zeros((len(etas), len(lmbds)))

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
            nn = MLPClassifier(hidden_layer_sizes=h_layers, activation=activation,
                                learning_rate_init=eta, alpha=lmbd, max_iter=300)
            nn.fit(X_train, y_train)
            NN[i, j] = nn

            test_scores[i, j] = nn.score(X_test, y_test)
            train_scores[i, j] = nn.score(X_train, y_train)

    return NN, test_scores, train_scores

def scikit_regressor_test(X, z, etas, lmbd, activation):
    """
    Test scikit mlp on frankeData for comparison
    """

    X_train, X_test, z_train, z_test = train_test_split(X, z)
    h_layers = (100, 100, 100, 100)
    #X_train, z_train = X, z

    NN = np.zeros((len(etas), len(lmbds)), dtype=object)
    test_scores = np.zeros((len(etas), len(lmbds)))
    train_scores = np.zeros((len(etas), len(lmbds)))

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
            nn = MLPRegressor(hidden_layer_sizes=h_layers, activation=activation, verbose=True,
                                learning_rate_init=eta, alpha=lmbd, max_iter=400, early_stopping= False)
            nn.fit(X_train, z_train)
            NN[i, j] = nn

            test_scores[i, j] = nn.score(X_test, z_test)
            train_scores[i, j] = nn.score(X_train, z_train)

    return NN, test_scores, train_scores

if __name__ == '__main__':
    NeuralNetwork_unit_test()
    #scikit_classifier_test(), scikit_regressor_test()

    X = X[:, 1:]
    X *= 1/np.max(X)
    y = to_categorical_numpy(np.array(np.ravel(y), dtype=np.int))

    eta, lmbd = 1e-3, 1e-2
    sci_nn = MLPClassifier(hidden_layer_sizes=(100, 100), activation="logistic",
                            learning_rate_init=eta, alpha=lmbd, max_iter=100, batch_size=80)
    sci_nn.out_activation_ = "softmax"

    nn = nn = NeuralNetwork(X, y, n_hidden_neurons=100, n_hidden_layers=2, epochs=100,
                            eta=eta, lmbd=lmbd, batch_size=80, activation="sigmoid", cost="log")

    sci_nn.fit(X, y)
    nn.train()

    sci_pred = sci_nn.predict(X)
    sci_pred = np.ravel(sci_pred[:, -1])

    pred = nn.predict(X)

    y = np.ravel(y[:, -1])
    print(accuracy_score(y, sci_pred))
    print(accuracy_score(y, pred))
    #diff_rate = np.sum(sci_pred != pred)/len(pred)
    #print(diff_rate)
    print(np.shape(sci_pred), np.shape(pred))

    #etas, lmbds = np.logspace(-5, 0, 6), np.logspace(-5, 0, 6)
    #NN, test_scores, train_scores = scikit_classifier_test(X, y, etas, lmbds, "relu")

    #X, z = frankeData()
    #NN, test_scores, train_scores = scikit_regressor_test(X, np.ravel(z), etas, lmbds, "tanh")
    #NN, train_scores = scikit_regressor_test(X, np.ravel(z), etas, lmbds, "tanh")
    #3, 2

    #x, y = X[:, 0].reshape(50, 50), X[:, 1].reshape(50, 50)
    #z = NN[i, j].predict(X).reshape(50, 50)

    #MyPlot(x, y, z, title="")
    #plt.show()

    #fig, ax = plt.subplots()
    #sns.heatmap(train_scores, annot=True, ax=ax, cmap="viridis")
    #ax.set_title("Training Accuracy")
    #ax.set_ylabel("$\eta$")
    #ax.set_xlabel("$\lambda$")
    #plt.show()

    #fig, ax = plt.subplots()
    #sns.heatmap(test_scores, annot=True, ax=ax, cmap="viridis")
    #ax.set_title("relu with mse cost: Test Accuracy")
    #ax.set_ylabel("$\eta$")
    #ax.set_xlabel("$\lambda$")
    #plt.show()
