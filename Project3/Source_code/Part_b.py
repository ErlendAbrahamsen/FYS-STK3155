import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error

def MyPlot(x, y, z, title="", shrink=0.5, aspect=4):
    """
    Simple 3d plot method for analytic solution of 1d-Heat eq.
    """

    #Set up a 3d plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

    labels = (ax.set_xlabel("Pos. (x)"), ax.set_ylabel("Time. (t)"),
                ax.set_zlabel("Temp. (u)"), ax.set_title(title))

    fig.colorbar(surf, shrink=shrink, aspect=aspect)

    return True

def finit_diff(dx=1e-1, ratio=0.5):
    """
    Forward time centered space finit differences of
    simple 1d heat eq /w given conditions.
    Returns 1d-arrays x, y and 2d-array u.
    """

    x_stop, t_stop = 1, 2
    dt = ratio*dx**2
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

def stability_ratio_error_analysis(n=5, max_ratio=0.5, dx=1e-1):
    """
    Computes MSE(analytic, finit_diff) for
    ratio values [max_ratio/n, ..., max_ratio].
    Returns list ratios and MSE.
    """

    MSE = []
    ratios = np.linspace(max_ratio/n, max_ratio, n)
    for ratio in ratios:
        x, t, u_approx = finit_diff(dx=dx, ratio=ratio)
        x_, t_ = np.meshgrid(x, t)
        u_analytic = u_true(x_, t_).T

        MSE.append(mean_squared_error(np.ravel(u_analytic), np.ravel(u_approx)))

    return ratios, MSE

def u_true(x, t):
    """
    Continouse analytic solution of 1d heat
    with given conditions.
    """

    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)

if __name__ == '__main__':
    ### MSE for different ratios (dt/dx^2 = ratio) ###
    dx1, dx2 = 1e-1, 1e-2
    ratios, MSE1 = stability_ratio_error_analysis(n=5, dx=dx1)
    plt.title("$MSE$ errors for different ratios")
    plt.plot(ratios, 1e3*np.array(MSE1), label="$\Delta x = %.1e$" % dx1)
    plt.xlabel("$=\Delta t / \Delta x^2$"), plt.ylabel("$10^{-3}$ $MSE$")
    plt.show()

    ratios, MSE2 = stability_ratio_error_analysis(max_ratio=0.6, n=6, dx=dx1)
    plt.title("Breaking stability criterion")
    plt.plot(ratios, np.array(MSE2), label="$\Delta x = %.1e$" % dx1)
    plt.xlabel("$=\Delta t / \Delta x^2$"), plt.ylabel("$MSE$")
    plt.show()

    ### Analytic and finit diff start state (t=0) ###
    dx = dx1
    x, t, u_approx = finit_diff(dx=dx)
    x_, t_ = np.meshgrid(x, t)
    u_analytic = u_true(x_, t_).T

    mse = mean_squared_error(np.ravel(u_analytic), np.ravel(u_approx))

    t_index = 0
    plt.title("Start-state $(t=%.1f)$ with $\Delta x = %.1e$" % (t[t_index], dx))
    plt.plot(x, u_approx[:, t_index], label="finit-diff")
    plt.plot(x, u_analytic[:, t_index], "--", color="r", label="analytic-solution")
    plt.xlabel("x-axis"), plt.ylabel("u-axis"), plt.text(0.35, 0, ("$MSE_{total}=%.2e$" % mse))
    plt.xlim(-0.05, 1.05), plt.ylim(-0.05, 1.05)
    plt.legend(), plt.show()

    ### Analytic vs finit-diff at fixed time###
    t_index = 5
    plt.title("$t=%.2f$ with $\Delta x = %.1e$" % (t[t_index], dx))
    plt.plot(x, u_approx[:, t_index], label="Finit-diff")
    plt.plot(x, u_analytic[:, t_index], "--", color="r", label="True-value")
    plt.xlabel("x-axis"), plt.ylabel("u-axis"), plt.text(0.35, 0.3, ("$MSE_{total}=%.2e$" % mse))
    plt.xlim(-0.05, 1.05), plt.ylim(-0.05, 1.05)
    plt.legend(), plt.show()

    ### Analytic solution plot ###
    x = np.linspace(0, 1, 50)
    x, t = np.meshgrid(x, x)
    u = u_true(x, t)
    MyPlot(x, t, u, title="Analytic 1D-Heat solution")
    plt.show()
