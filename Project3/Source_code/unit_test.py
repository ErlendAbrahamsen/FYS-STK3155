from Part_c import *

def test_cost(x, t, u_hat):
    """
    Solve PDE with solution u(x, 0) = sin(x)
    Properties: u_x = cos(x), u(0, 0) = 0.
    """

    u = u_hat(x, t)
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    zeros = tf.zeros(tf.shape(x), dtype=tf.float64)

    return mean_squared_error(u_x, tf.cos(x)) + mean_squared_error(u_hat(zeros, zeros), zeros)

def HeatLearner_test(tol=1e-3, n=10):
    """
    AssertionError if not passed.
    """

    tf.set_random_seed(1)
    x = np.linspace(0, 1, n)
    x, t = np.meshgrid(x, x)
    layer_dimensions = [2, 500, 1]

    dnn = HeatLearner(x, t, layer_dimensions, custom_loss=test_cost)
    dnn.train(num_iter=200)
    x, t = x.reshape(-1, 1), t.reshape(-1, 1)
    u_hat = dnn.predict(x, t)
    mse = skl.metrics.mean_squared_error(np.sin(x[:n]), u_hat[:n])
    print(mse)

    assert abs(mse) < tol

if __name__ == '__main__':
    HeatLearner_test()
