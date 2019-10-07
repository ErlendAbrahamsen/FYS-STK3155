from Part_a import *
from imageio import imread
from sklearn.preprocessing import normalize

if __name__ == '__main__':
    z = imread("SRTM_data_Norway_1.tif") #Load data
    rows, cols = np.shape(z)
    x, y = np.linspace(0, cols, cols), np.linspace(0, rows, rows)
    x, y = np.meshgrid(x, y)

    ##Plot
    plt.title("Terrain over Norway 1")
    plt.imshow(z, cmap=cm.coolwarm)
    plt.xlabel("x"), plt.ylabel("y")
    plt.colorbar()

    ##n=5 OLS plot
    X = DesignMatrix(np.ravel(x), np.ravel(y), n=5)
    beta = OLS(X, np.ravel(z))
    z_OLS = np.dot(X, beta).reshape(rows, cols)
    MyPlot(x, y, z_OLS, title="n=5 OLS of real data")
    plt.show()
