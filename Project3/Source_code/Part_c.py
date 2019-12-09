from Part_b import u_true, MyPlot
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import tensorflow as tf
from tensorflow.losses import mean_squared_error
from tensorflow import keras

class HeatLearner:
    """
    Class designed to solve 1d heat PDE using tensorflow functionality.
    Uses Adam optimizer.

    Different activations (from tensorflow) can be set in init.
    Can input custom made loss function custom_loss in init.:
    """

    def __init__(self, x, t, layer_dimensions, activation=tf.nn.sigmoid, custom_loss=False):
        """
        x, t is meshgrid and/or even lengths.
        layer_dimensions is list of dimensions (int) [#input, #hidden, #out].
        activation = tf.nn.sigmoid / tf.nn.relu etc.
        custom_loss to replace loss().
        custom_loss must input tensorflow objects x, t, u_hat.
        """

        self.x = tf.convert_to_tensor(x.reshape(-1, 1), dtype=tf.float64)
        self.t = tf.convert_to_tensor(t.reshape(-1, 1), dtype=tf.float64)

        self.layer_dimensions = layer_dimensions
        self.activation = activation
        self.create_weights_biases(eps=1e-2)

        if custom_loss:
            self.loss = custom_loss
        else:
            self.custom_loss = False

        self.LOSS = self.loss(self.x, self.t, self.u_hat)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.LOSS)
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def create_weights_biases(self, eps=1e-2):
        """
        Initializes weights and biases as lists of tensor variables.
        Weight initilized as random normal distrubuted. Mean 0.0, STD=1.0 .
        bias initiliazed to constant eps.
        """

        layer_dimensions = self.layer_dimensions

        weights, biases = [], []
        num_layers = len(layer_dimensions)
        for l in range(num_layers-1):
            W = tf.Variable(tf.random_normal([layer_dimensions[l], layer_dimensions[l+1]], dtype=tf.float64))
            b = np.zeros((1, layer_dimensions[l+1])) + eps
            b = tf.Variable(tf.convert_to_tensor(b, dtype=tf.float64))
            weights.append(W), biases.append(b)

        self.weights, self.biases = weights, biases

    def u_hat(self, x, t):
        """
        Feed forward for u(x, t) approximation.
        Output function set to sigmoid for now.
        """

        num_layers = len(self.layer_dimensions)
        a = tf.concat([x, t], 1)
        for l in range(num_layers-2):
            W, b = self.weights[l], self.biases[l]
            a = self.activation(tf.add(tf.matmul(a, W), b))

        W, b = self.weights[-1], self.biases[-1]
        self.dnn_out = tf.nn.sigmoid(tf.add(tf.matmul(a, W), b))
        #self.dnn_out = tf.add(tf.matmul(a, W), b)

        return self.dnn_out

    def loss(self, x, t, u_hat):
        """
        Returns mse sums catching properties of 1d heat eq.
        """

        #Parital derivatives
        u = u_hat(x, t)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_t = tf.gradients(u, t)[0]

        #Initial and boundary conditions
        n = tf.size(x)
        zeros, ones = tf.zeros([n, 1], dtype=tf.float64), tf.ones([n, 1], dtype=tf.float64)
        input_values = tf.cast(tf.reshape(tf.linspace(0.0, 1.0, n), [-1, 1]), dtype=tf.float64)

        u_t0 = u_hat(input_values, zeros)     #t=0
        u_x0 = u_hat(zeros, input_values)     #x=0
        u_x1 = u_hat(ones, input_values)      #x=1

        return (mean_squared_error(u_t, u_xx) + mean_squared_error(u_t0, tf.sin(np.pi*input_values))
                + mean_squared_error(u_x0, zeros) + mean_squared_error(u_x1, zeros))

    def train(self, num_iter=100):
        """
        Training over num_iter with optimizer set to AdamOptimizer
        in __init__.
        """

        t0 = time.time()
        print("Initial training loss:", self.sess.run(self.LOSS))

        for i in range(num_iter):
            self.sess.run(self.optimizer)

        t1 = time.time()
        t = t1 - t0
        print("Final training loss:", self.sess.run(self.LOSS))
        print("Training time: %f s" % t)

    def predict(self, x, t):
        """
        Returns Feed forward pass of u_hat
        """

        x = tf.convert_to_tensor(x.reshape(-1, 1), dtype=tf.float64)
        t = tf.convert_to_tensor(t.reshape(-1, 1), dtype=tf.float64)

        return self.sess.run(self.u_hat(x, t))

if __name__ == '__main__':
    ### Training ###
    tf.set_random_seed(1)
    n_train = 10
    x = np.linspace(0, 1, n_train)
    t = np.linspace(0, 2, n_train)
    x, t = np.meshgrid(x, t)

    #Layer architecture
    layer_dimensions = [2, 200, 200, 200, 200, 200, 1]
    dnn = HeatLearner(x, t, layer_dimensions, activation=tf.nn.sigmoid)
    dnn.train(num_iter=200)

    ### Predicting and plotting ###
    n = 100
    x = np.linspace(0, 1, n)
    t = np.linspace(0, 1, n)
    x, t = np.meshgrid(x, t)
    u_hat = dnn.predict(x, t)

    u_true = u_true(x.reshape(-1, 1), t.reshape(-1, 1))
    true_MSE = skl.metrics.mean_squared_error(np.ravel(u_true), np.ravel(u_hat))
    print("True MSE:", true_MSE)

    title = "%d Hidden layers trained on %dX%d grid. \n $MSE_{total}=%.3e$" % (len(layer_dimensions)-2, n_train, n_train, true_MSE)
    MyPlot(x.reshape((n, n)), t.reshape((n, n)), u_hat.reshape((n, n)), title=title)
    plt.show()
