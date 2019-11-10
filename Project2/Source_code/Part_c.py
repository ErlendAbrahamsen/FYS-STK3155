from Part_a import X, y
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class NeuralNetwork:
    """
    NeuralNetwork class capable of binary classification and continiouse regression.
    Only feed forward, can have multiple layers and activation functions.
    Has two types of output function. Softmax for classification and T(A)=1/max(A)*A for regression.

    ### Initializer and attributes ###
    X_data, Y_data: ndarray data inputs. X_data is (n_inputs, n_features).
                    Y_data is (n_inputs, n_categories).

    activation, type, cost: (strings) Options for activation are "sigmoid", "tanh", "relu".
                      type is either "classification" or "regression".
                      cost is "log" or "mse".

    n_hidden_layers, n_hidden_neurons: Number of hidden layers and hidden neurons.

    epochs, batch_size: number of training iterations and size of each mini-batch.

    eta, lmbd: learning rate in gradient descent and regularization parameter.

    ### Methods ###
    (See docstrings in method itself)
    create_biases_and_weights(self): Random initial guesses on all biases and weights.

    activ(self, t, derivative=False): Activation function or derivative

    out_func(self, t): output function in last layer

    feed_forward(self, X): Forwards pass in network

    backpropagation(self): Backwards pass in network for updating weights and biases

    train(self): trains network

    predict(self, X): Predictions on X
    """

    def __init__(
            self,
            X_data,
            Y_data,
            activation="sigmoid",
            type = "classification",
            cost = "log",
            n_hidden_layers=2,
            n_hidden_neurons=50,
            epochs=100,
            batch_size=80,
            eta=0.05,
            lmbd=0.5):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs, self.n_features = np.shape(X_data)
        self.activation = activation
        self.type = type
        self.cost = cost
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        if self.type == "classification":
            self.n_categories = 2

        elif self.type == "regression":
            self.n_categories = 1

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        """
        Initialize random normal distrbuted weights and biases for each layer
        """

        np.random.seed(1)

        #Set correct dimensions for weights
        self.weights = np.zeros(self.n_hidden_layers + 1, dtype=np.ndarray)
        self.weights[0]  = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.weights[-1] = np.random.randn(self.n_hidden_neurons, self.n_categories)

        for i in range(1, self.n_hidden_layers):
            self.weights[i] = np.random.randn(self.n_hidden_neurons, self.n_hidden_neurons)

        #Set correct dimensions for biases
        self.bias = np.zeros(self.n_hidden_layers + 1, dtype=np.ndarray)
        self.bias[-1] = np.zeros(self.n_categories) + 0.01

        for i in range(self.n_hidden_layers):
            self.bias[i] = np.zeros(self.n_hidden_neurons) + 0.01

    def activ(self, t, derivative=False):
        """
        Hidden activation function.
        Returns derivative in terms of activation ndarray if set to True,
        else it returns sigmoid(t), tanh(t) or relu(t).
        """

        try:
            t = np.array(t, dtype=np.float64)
        except:
            pass

        if self.activation == "sigmoid":
            if derivative: return t*(1-t) #Note that input t must be the activation (a[i])
            #return 1/(1+np.exp(-t))
            return 0.5*(np.tanh(0.5*t)+1)

        elif self.activation == "tanh":
            if derivative: return 1 - t**2
            return np.tanh(t)

        elif self.activation == "relu":
            if derivative:
                t[t > 0] = 1
                t[t <= 0] = 0
                return t
            return np.maximum(t, 0)

    def out_func(self, t):
        """
        Outputlayer function.
        Returns softmax(t) or t for self.type = "classification"
        or "regression".
        """

        try:
            t = np.array(t, dtype=np.float64)
        except:
            pass

        if self.type == "classification":
            exp_term = np.exp(t)
            return exp_term / np.sum(exp_term, axis=1, keepdims=True)

        elif self.type == "regression":
            return t/np.max(t)

    def feed_forward(self, X):
        """
        Feed forwards pass on the X data.
        a contains ndarrays of input and output of activations.
        """

        a = np.zeros(self.n_hidden_layers + 2, dtype=np.ndarray)

        a[0] = X
        for i in range(self.n_hidden_layers):
            z = np.dot(a[i], self.weights[i]) + self.bias[i]
            a[i+1] = self.activ(z)

        z_o = np.dot(a[-2], self.weights[-1]) + self.bias[-1]
        a[-1] = self.out_func(z_o)
        self.a = a

    def backpropagation(self):
        """
        Updates weights and biases by using gradient descent backwards.
        """

        delta = np.zeros(self.n_hidden_layers + 1, dtype=np.ndarray) #delta derivatives
        a = self.a
        aL = a[-1]

        #Initialize delta and gradients
        if self.cost == "log":  #log cost
            if self.activation == "sigmoid":
                delta[-1] = (aL - self.Y_data)
            else:
                delta[-1] = (aL - self.Y_data)/(aL*(1-aL)) * self.activ(aL, derivative=True)

        else: #mse cost
            delta[-1] = (aL - self.Y_data) * self.activ(aL, derivative=True)

        self.weights_grad = np.zeros(self.n_hidden_layers + 1, dtype=np.ndarray)
        self.weights_grad[-1] = np.dot(a[-2].T, delta[-1])

        self.bias_grad = np.zeros(self.n_hidden_layers + 1, dtype=np.ndarray)
        self.bias_grad[-1] = np.sum(delta[-1], axis=0)

        #Setting gradients backwards
        for i in range(-2, -self.n_hidden_layers - 2, -1):
            delta[i] = np.dot(delta[i+1], self.weights[i+1].T) * self.activ(a[i], derivative=True)
            self.weights_grad[i] = np.dot(a[i-1].T, delta[i])
            self.bias_grad[i] = np.sum(delta[i], axis=0)

        #Gradient descent backwards for updating weights and biases
        for i in range(-1, -self.n_hidden_layers - 2, -1):
            if self.lmbd > 0.0: self.weights_grad[i] += self.lmbd * self.weights[i] #Regularization

            self.weights[i] = self.weights[i] - self.eta * self.weights_grad[i]     #Gradient descent
            self.bias[i] = self.bias[i] - self.eta * self.bias_grad[i]

    def train(self):
        """
        Train network using random mini-batches and
        backpropagation.
        """

        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward(self.X_data)
                self.backpropagation()

    def predict(self, X):
        """
        return prediction output given input X (n_inputs, n_features).
        """

        self.feed_forward(X)

        if self.type == "classification":
            #Chooses greatest probabilty as prediction
            return np.argmax(self.a[-1], axis=1)

        elif self.type == "regression":
            return self.a[-1]

def grid_search(X, y, etas, lmbds, activation="sigmoid", cost="log"):
    """
    Loops over different value combinations from etas and lmbds array.
    Returns ndarray of NeuralNetworks (NN) and scores (scores)
    for each eta and lmbd combination.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

    NN = np.zeros((len(etas), len(lmbds)), dtype=object)
    test_scores = np.zeros((len(etas), len(lmbds)))
    train_scores = np.zeros((len(etas), len(lmbds)))

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
            nn = NeuralNetwork(X_train, y_train, eta=eta, lmbd=lmbd, n_hidden_neurons=50,
                               n_hidden_layers=2, activation=activation, cost=cost, epochs=100)
            nn.train()
            NN[i, j] = nn

            y_test_pred = nn.predict(X_test)
            y_train_pred = nn.predict(X_train)

            test_scores[i, j] = accuracy_score(y_test[:, -1], y_test_pred)
            train_scores[i, j] = accuracy_score(y_train[:, -1], y_train_pred)

    return NN, test_scores, train_scores

def accuracy_score(y_test, y_pred):
    """
    Return corect- to total predictions ratio
    """

    y_test, y_pred = np.ravel(y_test), np.ravel(y_pred)

    return np.sum(y_test == y_pred)/len(y_test)

def to_categorical_numpy(integer_vector):
    """
    Change integer array to ndarray in one-hot notation
    """

    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector

if __name__ == '__main__':
    X = X[:, 1:]
    #Standardize features
    X = preprocessing.StandardScaler().fit(X).transform(X)
    y = to_categorical_numpy(np.array(np.ravel(y), dtype=np.int))

    #Grid search (tanh with mse cost)
    etas, lmbds = np.logspace(-5, 0, 6), np.logspace(-5, 0, 6)
    activation, cost = "tanh", "mse"
    NN, test_scores, train_scores = grid_search(X, y, etas, lmbds, activation=activation, cost=cost)

    fig, ax = plt.subplots()    #Train acc plot
    sns.heatmap(train_scores, annot=True, ax=ax, cmap="viridis")
    ax.set_title("%s with %s cost: Training Accuracy" % (activation, cost))
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    fig, ax = plt.subplots()    #Test acc plot
    sns.heatmap(test_scores, annot=True, ax=ax, cmap="viridis")
    ax.set_title("%s with %s cost: Test Accuracy" % (activation, cost))
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    #Best false positive/negative rates (sigmoid logcost)
    eta, lmbd = 1e-3, 1e-1
    activation, cost = "sigmoid", "log"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

    nn = NeuralNetwork(X_train, y_train, eta=eta, lmbd=lmbd, n_hidden_neurons=100,
                       n_hidden_layers=4, activation=activation, cost=cost, epochs=100)
    nn.train()
    y_pred = np.ravel(nn.predict(X_test))
    y_test = np.ravel(y_test[:, 1])

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    false_pos_rate = fp/(fp+tn)
    false_neg_rate = fn/(fn+tp)
    print("False-neg. rate: %.3f" % false_neg_rate)
    print("False-pos. rate: %.3f" % false_pos_rate)
