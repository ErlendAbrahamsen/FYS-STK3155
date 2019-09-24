import time
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from Part_a import DesignMatrix, OLS, Polynom, MSE, R2

def test_DesignMatrix(n=15, m=300, eps=1e-16):
    """
    Check that length of last row and value is correct!
    Also benchmark with timing of DesignMatrix().
    """

    true_val = 1.0 #y[-1]**n
    true_length = int((n+1)*(n+2)/2)
    l = np.linspace(0, 1, m) #Interval
    x, y = np.meshgrid(l, l)
    x, y = np.ravel(x), np.ravel(y)
    t0 = time.time()
    X = DesignMatrix(x, y, n=n)
    t1 = time.time()
    passed = True
    if len(X[-1])!=true_length or abs(X[-1][-1]-true_val)>eps:
        passed = false

    print("(m,m) = (%d,%d) DesignMatrix() benchmark: %f s." % (m, m, t1-t0))
    assert passed, "Dimensions and/or last value is wrong"

def test_Polynom(n=15, m=300, eps=1e-16):
    """
    Check that length of last row and value is correct!
    Also benchmark with timing of Polynom().
    """

    num = int((n+1)*(n+2)/2)
    coeff = np.random.rand(num)
    l = np.linspace(0, 1, m)         #Interval
    x, y = np.meshgrid(l, l)
    x, y = np.ravel(x), np.ravel(y)  #mxm points
    t0 = time.time()
    z = Polynom(x, y, coeff, n=n)
    t1 = time.time()
    print("(m,m) = (%d,%d) Polynom() benchmark: %f s." % (m, m, t1-t0))

def test_MSE(eps=1e-16):
    """
    """

def test_R2():
    """
    """

def test_KFold_Cross():
    """
    Test correct intervall lengths
    """
    #Unit test ????
    #print(x.shape)
    #t = np.arange(10)
    #n = 4
    #p = int(t.size/n)
    #print(p)
    #for i in range(n):
    #    print(t[p*i:p*(i+1)])
    #    print(np.append(t[:p*i], t[p*(i+1):], axis=0))
    ####
    #print(x)
    #r = np.arange(x.size)
    #np.random.shuffle(r)
    #x = [np.ravel(x)[i] for i in r]
    #x = np.array(x).reshape(25,25)

if __name__ == '__main__':
    test_DesignMatrix(), test_Polynom()
