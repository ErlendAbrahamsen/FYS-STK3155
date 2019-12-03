import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error

def MyPlot(x, y, z, title="", shrink=0.5, aspect=4):
    """
    3D plotting method with features fitting our situation
    """

    #Set up a 3d plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

    labels = (ax.set_xlabel("x-axis"), ax.set_ylabel("y-axis"),
                ax.set_zlabel("z-axis"), ax.set_title(title))

    fig.colorbar(surf, shrink=shrink, aspect=aspect)

    return True

def finit_diff(dx=1e-1):
    """
    Forward time centered space finit differences of
    simple 1d heat eq /w given conditions
    """

    x_stop, t_stop = 1, 5
    dt = dx**2/2
    d = dt/dx**2
    d2 = 1 - 2*d

    x = np.linspace(0, x_stop, int(1+x_stop/dx))
    t = np.linspace(0, t_stop, int(1+x_stop/dt))

    u = np.zeros((x.size, t.size))
    u[:, 0] = np.sin(np.pi*x)

    for j in range(len(t)-1):
        for i in range(1, len(x)-1):
            u[i, j+1] = d*u[i+1, j] + d2*u[i, j] + d*u[i-1, j]

    return x, t, u

def u(x, t):
    """
    Continouse analytic solution of 1d heat
    with given conditions
    """

    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)

if __name__ == '__main__':
    x, t, u_approx = finit_diff(dx=1e-2)
    x_, t_ = np.meshgrid(x, t)
    u_true = u(x_, t_).T

    mse = mean_squared_error(np.ravel(u_true), np.ravel(u_approx))

    t_ = 0.2
    plt.title("Time t=%.1f" % t_)
    plt.plot(x, u_true[:, np.where(t==t_)[0]], label="analytic-solution")
    plt.plot(x, u_approx[:, np.where(t==t_)[0]], label="finit-difference")
    plt.xlabel("x-axis"), plt.ylabel("u-axis"), plt.text(0.8, 0.8, ("mse=%.3e" % mse))
    plt.legend(), plt.show()

    ##print(np.shape(x))
    #x_, t_ = x[:11, :], t[:11, :]
    #u_true_, u_approx_ = u_true[:11, :11], u_approx[:11, :11]
    #print(np.shape(u_true_))
    #MyPlot(x_, t_, u_true_, title="analytic-solution")
    #MyPlot(x_, t_, u_approx_, title="approx-solution")
    #plt.xlabel("x-position"), plt.ylabel("t-time")
    #plt.show()
