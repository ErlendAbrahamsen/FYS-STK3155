from Part_a import X, y
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    """
    NeuralNetwork class capable of binary classification and continiouse regression.
    Only feed forward, can have multiple layers and activation functions.
    Has two types of output function. Softmax for classification and f(x)=x for regression.

    ### Initializer and attributes ###
    X_data, Y_data: ndarray data inputs. X_data is (n_inputs, n_features).
                    Y_data is (n_inputs, n_categories).

    activation, type: (strings) Options for activation are "sigmoid", "tanh", "relu".
                      type is either "classification" or "regression".

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
            n_hidden_layers=2,
            n_hidden_neurons=50,
            epochs=20,
            batch_size=100,
            eta=0.05,
            lmbd=0.5):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs, self.n_features = np.shape(X_data)
        self.type = type
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.activation = activation

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
        self.weights = np.zeros(self.n_hidden_layers + 2, dtype=np.ndarray)
        self.weights[0]  = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.weights[-1] = np.random.randn(self.n_hidden_neurons, self.n_categories)
        for i in range(1, self.n_hidden_layers + 1):
            self.weights[i] = np.random.randn(self.n_hidden_neurons, self.n_hidden_neurons)

        self.bias = np.zeros(self.n_hidden_layers + 2, dtype=np.ndarray)
        self.bias[-1] = np.zeros(self.n_categories) + 0.01
        for i in range(self.n_hidden_layers + 1):
            self.bias[i] = np.zeros(self.n_hidden_neurons) + 0.01

    def activ(self, t, derivative=False):
        """
        Hidden activation function.
        Returns derivative if set to True, else
        it returns sigmoid(t), tanh(t) or relu(t).
        """

        try:
            t = np.array(t, dtype=np.float64)
        except:
            pass

        if self.activation == "sigmoid":
            if derivative: return np.exp(-t)/(1+np.exp(-t))**2
            return 1/(1+np.exp(-t))

        elif self.activation == "tanh":
            if derivative: return 1/(np.cosh(t))**2
            return np.tanh(t)

        elif self.activation == "relu":
            if derivative:
                t[t > 0] = 1
                t[t < 0] = 0
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
            return t

    def feed_forward(self, X):
        """
        Feed forwards pass on the X data.
        z, a contains ndarrays of input and output of activations.
        returns z, a.
        """

        z = np.zeros(self.n_hidden_layers + 1, dtype=np.ndarray)
        a = np.zeros(self.n_hidden_layers + 1, dtype=np.ndarray)

        z[0], a[0] = 0, X
        for i in range(self.n_hidden_layers):
            z[i+1] = np.dot(a[i], self.weights[i]) + self.bias[i]
            a[i+1] = self.activ(z[i+1])

        z_o = np.dot(a[-1], self.weights[-1]) + self.bias[-1]
        self.out = self.out_func(z_o)

        return z, a

    def backpropagation(self, z, a):
        """
        Updates weights and biases by using gradient descent backwards.
        """

        delta = np.zeros(self.n_hidden_layers + 2, dtype=np.ndarray) #delta derivatives
        aL = self.out #Output

        #Initialize delta and gradients
        delta[-1] = aL - self.Y_data ###

        self.weights_grad = np.zeros(self.n_hidden_layers + 2, dtype=np.ndarray)
        self.weights_grad[-1] = np.dot(a[-1].T, delta[-1]) ###

        self.bias_grad = np.zeros(self.n_hidden_layers + 2, dtype=np.ndarray)
        self.bias_grad[-1] = np.sum(delta[-1], axis=0)

        #Setting gradients backwards
        for i in range(-2, -self.n_hidden_layers-1, -1):
            delta[i] = np.dot(delta[i+1], self.weights[i+1].T) * a[i] * (1 - a[i]) ###
            self.weights_grad[i] = np.dot(a[i].T, delta[i])
            self.bias_grad[i] = np.sum(delta[i], axis=0)

        #Gradient descent backwards for updating weights and biases
        for i in range(-1, -self.n_hidden_layers-3, -1):
            if self.lmbd > 0.0: self.weights_grad[i] += self.lmbd * self.weights[i]

            self.weights[i] = self.weights[i] - self.eta * self.weights_grad[i]
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

                z, a = self.feed_forward(self.X_data)
                self.backpropagation(z, a)

    def predict(self, X):
        """
        return prediction output given input X (n_inputs, n_features).
        """

        self.feed_forward(X)

        if type == "classification":
            #Chooses greatest probabilty as prediction
            return np.argmax(self.out, axis=1)

        elif type == "regression":
            return self.out

def grid_search(etas, lmbds, X_train, X_test, y_train, y_test):
    """
    Loops over different value combinations from etas and lmbds array.
    Returns ndarray of NeuralNetworks (NN) and scores (scores)
    for each eta and lmbd combination.
    """

    NN = np.zeros((len(etas), len(lmbds)), dtype=object)
    scores = np.zeros((len(etas), len(lmbds)))

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
            nn = NeuralNetwork(X_train, y, eta=eta, lmbd=lmbd, n_hidden_neurons=20, n_hidden_layers=4)
            nn.train()
            NN[i, j] = nn

            y_pred = nn.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            scores[i, j] = score

    return NN, scores

def accuracy_score(y_test, y_pred):
    """
    Return corect- to total predictions ratio
    """

    y_test, y_pred = np.ravel(y_test), np.ravel(y_pred)

    return np.sum(y_test == y_pred)/len(y_test)

def to_categorical_numpy(integer_vector):
    """
    Convert integer array to onehot notation array
    """

    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector

if __name__ == '__main__':
    X = X[:, 1:]
    X *= 1/np.max(X)
    #y = to_categorical_numpy(np.array(np.ravel(y), dtype=np.int))

    #X, y = X[:1000], y[:1000]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    nn = NeuralNetwork(X_train, y_train, n_categories=2, n_hidden_neurons=50, n_hidden_layers=4,
                       eta=1e-4, lmbd=0.1, batch_size=80, activation="sigmoid")
    nn.train()
    y_pred = nn.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(y_pred[:15])
    print(np.ravel(y_test[:15]))

    #test = np.random.randn(2, 5, 2)
    #print(test[1])
    #t = np.zeros((2, ))

    #etas, lmbds = np.logspace(-5, 1, 7), np.logspace(-5, 1, 7)
    #NN, scores = grid_search(etas, lmbds, X_train, X_test, y_train, y_test)
    #best_score = np.max(scores)
    #print(best_score)
    #print(np.where(scores == best_score))

    #print("\n\n", np.shape(p))
    #print(p[:10])
    #print(y[:10])
