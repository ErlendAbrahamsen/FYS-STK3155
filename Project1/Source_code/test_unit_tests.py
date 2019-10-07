import time
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from Part_a import DesignMatrix, OLS, Polynom, MSE, R2
from Part_b import KFoldCross

"""
Tests are passed unless an assertion error is raised.
"""

def test_DesignMatrix(n=15, m=300, eps=1e-12):
    """
    Check that length of last row and value is correct!
    Also benchmark with timing of DesignMatrix().
    n: polynomial degree. m: arraysize. eps: error tolerance.
    """

    true_val = 1.0                      #y[-1]**n
    true_length = int((n+1)*(n+2)/2)
    l = np.linspace(0, 1, m)            #Interval
    x, y = np.meshgrid(l, l)            #mxm meshgrid
    x, y = np.ravel(x), np.ravel(y)
    t0 = time.time()
    X = DesignMatrix(x, y, n=n)
    t1 = time.time()
    passed = True
    if len(X[-1]) != true_length or abs(X[-1][-1]-true_val) > eps:
        passed = false

    print("(m,m) = (%d,%d) DesignMatrix() benchmark: %f s." % (m, m, t1-t0))
    assert passed, "Dimensions and/or last value is wrong"

def test_Polynom(m=300, eps=1e-12):
    """
    Check that length of last row and value is correct!
    Also benchmark with timing of Polynom().
    m: arraysize. eps: error tolerance.
    """

    n = 5 #polynom degree
    num = int((n+1)*(n+2)/2)
    coeff = np.random.rand(num)
    l = np.linspace(0, 1, m)
    x, y = np.meshgrid(l, l)
    t0 = time.time()
    z = Polynom(x, y, coeff, n=n)
    t1 = time.time()
    print("(m,m) = (%d,%d) Polynom() benchmark: %f s." % (m, m, t1-t0))

    #n=5 polynom test value
    r = np.random.randint(m, size=4)
    x, y = x[r[0]][r[1]], y[r[2]][r[3]]
    terms = [1, x, y, x**2, x*y, y**2,
            x**3, x**2*y, x*y**2, y**3,
            x**4, x**3*y, x**2*y**2, x*y**3, y**4,
            x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5]
    true_val = np.dot(coeff, terms)

    assert abs(Polynom(x, y, coeff, n=n)-true_val) < eps

def test_MSE(m=300, eps=1e-12):
    """
    Confirm results with professional library Scikit learn.
    Using two sets of mxm N(0,1) points.
    Also benchmark vs Scikit.
    m: arraysize. eps: error tolerance.
    """

    z, z_ = np.random.randn(m, m), np.random.randn(m, m)
    z, z_ = np.ravel(z), np.ravel(z_)

    t0 = time.time()
    test = MSE(z, z_)
    t1 = time.time()
    T = t1 - t0

    t0_ = time.time()
    true = mean_squared_error(z, z_)
    t1_ = time.time()
    T_ = t1_ - t0_

    print("(m,m) = (%d,%d) MSE() benchmark VS Sklearn: %f s (Time diff)." % (m, m, abs(T_-T)))
    assert abs(test-true) < eps

def test_R2(m=300, eps=1e-12):
    """
    Confirm results with professional library Scikit learn.
    Using two sets of mxm N(0,1) points.
    Also benchmark vs Scikit.
    m: arraysize. eps: error tolerance.
    """

    z, z_ = np.random.randn(m, m), np.random.randn(m, m)
    z, z_ = np.ravel(z), np.ravel(z_)

    t0 = time.time()
    test = R2(z, z_)
    t1 = time.time()
    T = t1 - t0

    t0_ = time.time()
    true = r2_score(z, z_)
    t1_ = time.time()
    T_ = t1_ - t0_

    print("(m,m) = (%d,%d) R2() benchmark VS Sklearn: %f s (Time diff)." % (m, m, abs(T_-T)))
    assert abs(test-true) < eps

def test_KFoldCross(m=300, k=15, eps=1e-4):
    """
    Test that mapping (x, y) -> z is preserved after shuffling in KFold.
    Check that there are k folds of equal length.
    Also benchmark KFold time.
    m: arraysize. k: #folds. eps: error tolerance.
    """

    x = np.arange(m**2)
    y = x
    z = x + y #mapping (x, y) -> x + y
    t0 = time.time()
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = KFoldCross(x, y, z, z, k=k)[:-2]
    t1 = time.time()

    print("(m,m) = (%d,%d) KFoldCross() benchmark: %f s." % (m, m, t1-t0))

    #Check that x_train has k-1 folds and x_test has 1 fold
    passed = True
    for x_train, x_test in zip(X_train, X_test):
        ratio = len(x_test)/len(x_train)
        if abs(ratio-1/(k-1)) > eps:
            passed = False

    #Check that mapping is preserved after shuffle
    for x, y, z in zip(X_train, Y_train, Z_train):
        if abs((x[-1]+y[-1])-z[-1]) > eps:
            passed = False

    assert passed

if __name__ == '__main__':
    test_DesignMatrix(), test_Polynom(), test_MSE(), test_R2(), test_KFoldCross()
