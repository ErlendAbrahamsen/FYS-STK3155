from Part_b import u
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras

class HeatLearner:
    def __init__(self, x, t, layer_dimensions):
        self.x = tf.convert_to_tensor(x.reshape(-1, 1))
        self.t = tf.convert_to_tensor(t.reshape(-1, 1))
        self.X = tf.concat([x, t], 1)
        self.layer_dimensions = layer_dimensions
        self.create_weights_biases(eps=1e-2)

    def create_weights_biases(self, eps=1e-2):
        """
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

    def feed_forward(self, x, t):
        """
        """

        num_layers = len(self.layer_dimensions)
        a = tf.concat([x, t], 1)
        for l in range(num_layers-2):
            W, b = self.weights[l], self.biases[l]
            a = tf.nn.sigmoid(tf.add(tf.matmul(a, W), b))

        W, b = self.weights[-1], self.biases[-1]
        self.dnn_out = tf.add(tf.matmul(a, W), b)

        return self.dnn_out

    def loss(self):
        """
        """

        u_hat = self.feed_forward(self.x, self.t)**1.0

        #Parital derivatives
        u_x = tf.gradients(u_hat, self.x)[0]
        u_xx = tf.gradients(u_x, self.x)[0]
        u_t = tf.gradients(u_hat, self.t)[0]

        #Initial and boundary conditions
        zeros = tf.zeros(tf.shape(self.x), dtype=tf.float64)
        u_t0 = self.feed_forward(self.x, zeros)     #t=0
        u_x0 = self.feed_forward(zeros, self.t)     #x=0
        u_x1 = self.feed_forward(zeros+1, self.t)   #x=1

        terms = [u_t-u_xx, u_t0-tf.sin(np.pi*self.x), u_x0, u_x1]
        return 1/36*(tf.reduce_sum(tf.square(terms[0])) + tf.reduce_sum(tf.square(terms[1])) + tf.reduce_sum(tf.square(terms[2])) + tf.reduce_sum(tf.square(terms[3])))

        #diff = u_t0 - tf.sin(np.pi*self.x[:6])

        #return 1/36*(tf.reduce_sum(tf.square(u_t-u_xx)))# + tf.reduce_sum(tf.square(diff)))
        #zeros = tf.zeros(tf.shape(x), dtype=tf.float64)
        #return keras.losses.MSE(zeros, u_t-u_xx) #+ keras.losses.MSE(zeros, diff)
        #print(tf.shape(zeros), tf.shape(u_t-u_xx))

    def train(self, num_iter=100):
        """
        """

        loss = self.loss()
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            print("Initial loss:", self.loss().eval())
            for i in range(num_iter):
                sess.run(optimizer)

            print("Final loss:", self.loss().eval())
            return self.dnn_out.eval()

    #def predict(self, )

if __name__ == '__main__':
    tf.set_random_seed(1)
    n = 6
    x = np.linspace(0, 1, n)
    x, t = np.meshgrid(x, x)
    layer_dimensions = [2, 100, 100, 100, 1]

    dnn = HeatLearner(x, t, layer_dimensions)
    out = dnn.train(num_iter=300)
    true = u(x.reshape(-1, 1), t.reshape(-1, 1))
    print(mean_squared_error(np.ravel(true), np.ravel(out)))
