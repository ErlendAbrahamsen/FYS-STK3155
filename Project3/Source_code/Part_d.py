from Part_c import HeatLearner
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import tensorflow as tf
from tensorflow.losses import mean_squared_error
from tensorflow import keras

def eigen_ODE_loss(x0, t, x_hat, l=300):
    """
    Custom loss function to be used in HeatLearner class.
    ( HeatLearner(custom_loss=eigen_ODE_loss) )

    Uses a loop to compute f(x(t)) because of shape requirments in f transform.
    x(t) output is shape (n*m, 1).
    x(t=t_m) is shape (n, 1).
    """

    np.random.seed(1) ###
    n = 6
    A = np.random.normal(0, 1, (n, n))
    A = (A.T + A)/2
    A = tf.convert_to_tensor(A, dtype=tf.float64) #(n, n) symmetrical matrix

    x = x_hat(x0, t) #First forward pass/prediction

    #Loops over each (n, 1) at each time m in (n*m, 1)
    Fx = []
    for i in range(int(l/n)):               #l/n = m
        x_vec = x[n*i:n*(i+1), 0]           #x vector at time dictated by rows
        x_vec = tf.reshape(x_vec, [-1, 1])  #Shape (n, 1)

        #f(x(t))
        xT = tf.transpose(x_vec)
        m1 = tf.matmul(xT, x_vec)*A
        m2 = (1 - tf.matmul(xT, tf.matmul(A, x_vec)))
        fx = tf.matmul(m1+m2, x_vec)

        Fx.append(fx)

    fx = tf.reshape(Fx, [-1, 1])            #Reshape to (n*m, 1)
    x_t = tf.gradients(x_hat(x0, t), t)[0]  #Gradient (n*m, 1)

    return mean_squared_error(x_t, fx-x)

if __name__ == '__main__':
    tf.set_random_seed(1), np.random.seed(1) ###
    n, m = 6, 50
    x0 = np.random.normal(0, 1, n)  #x0(t) guess
    t = np.linspace(0, 100, m)      #times
    x0, t = np.meshgrid(x0, t)

    #Random symmetrical (n, n) (normal distb.)
    A = np.random.normal(0, 1, (n, n))
    A = (A.T + A)/2
    eig_val, eig_vec = np.linalg.eig(A)

    #Layer architecture and training
    layer_dimensions = [2, 200, 200, 1]
    dnn = HeatLearner(x0, t, layer_dimensions, activation=tf.nn.sigmoid, custom_loss=eigen_ODE_loss)
    dnn.train(num_iter=200)
    x = dnn.predict(x0, t)

    ### Compute lambda at each timepoint t_m ###
    Lmbd = []
    for i in range(m):
        x_vec=x[n*i:n*(i+1)].reshape(-1, 1)
        lmbd = x_vec.T@A@x_vec/(x_vec.T@x_vec)
        Lmbd.append(lmbd[0][0])

    ### Final plotting of each true eigenval and predicted eigenval. ###
    abs_end_error = min([abs(Lmbd[-1]-eig) for eig in eig_val])
    plt.title("ODE predicted $\lambda$ vs Numpy computed $\lambda$")
    plt.plot(Lmbd, label="Predicted $\lambda$")
    plt.plot(eig_val, "o", c="r", label="True $\lambda$")
    plt.text(7.5, 1, ("$Error_{end}=%.2e$" % abs_end_error))
    plt.legend(), plt.show()
