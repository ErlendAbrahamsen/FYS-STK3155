from Part_a import X, y
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scikitplot.metrics import plot_cumulative_gain

def dlogcost(X, y, beta):
    """
    Returns the hessian and gradient of the cross entropy cost
    """

    p = logfunc(X, beta)
    W = np.diagflat(p*(1-p))
    H = X.T @ W @ X

    Grad = np.dot(X.T, p-y)

    return H, Grad

def newtonsMethod(X, y, k=0.4, max_iter=1e2, tol=1e-3):
    """
    Newtons method with optional step size k.
    """

    #Random seed to avoid singular matrix
    np.random.seed(2)
    p = np.shape(X)[1]
    beta = np.random.normal(0, 10, (p, 1)) #Random guess
    for i in range(int(max_iter)):
        H, Grad = dlogcost(X, y, beta)     #Hessian and gradient
        prev_beta = beta
        beta = beta - k*np.dot(np.linalg.inv(H), Grad)  #Update beta
        next_beta = beta

        #Using max absolute difference as tolerance condition
        if np.max(abs(next_beta - prev_beta)) < tol:
            break

    #If tolerance condition not met
    if i == int(max_iter-1):
        print("Convergence warning: max iterations reached")
        print("max_iter = %.2e, tol = %.2e" % (max_iter, tol))

    return beta

def logfunc(X, beta):
    """
    Sigmoid/log function, input/output is scaler or arraylike.
    """

    Xb = np.array(np.dot(X, beta), dtype=np.float32)

    return 1/(1 + np.exp(-Xb))

def Eaccuracy(p, y):
    """
    Expectance of correctly guessed / total data.
    I.e. if probability of 1 is 0.6 it predicts 1 0.6 times.
    """

    p, y = np.ravel(p), np.ravel(y)
    total, score = len(p), 0
    for p, y in zip(p, y):
        if y == 1:
            score += p
        else:
            score += 1 - p

    return score/total

def split_accuracy(y_test, y_pred, p=0.5):
    """
    Predict 1 if probabilty of 1 is >= p.
    Return accuracy.
    """

    y_test, y_pred = np.ravel(y_test), np.ravel(y_pred)
    y_pred[y_pred >= p] = 1
    y_pred[y_pred < p] = 0

    return np.sum(y_test == y_pred)/len(y_test)

if __name__ == '__main__':
    X_max = np.max(X)
    X = X/X_max #Scaling [0,1]

    #Average prediction over n runs with prob >= split predicts 1.
    n, split = 10, 0.5
    Train_acc, Test_acc = [], []
    for i in range(n):
        #Split into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=i)

        #Training beta weights and getting probabilty predictions
        beta = newtonsMethod(X_train, y_train)
        p_train, p_test = logfunc(X_train, beta), logfunc(X_test, beta)

        #Appending accuracy's
        Train_acc.append(split_accuracy(y_train, p_train, p=split))
        Test_acc.append(split_accuracy(y_test, p_test, p=split))

    #Average accuracy's
    train_acc, test_acc = np.mean(Train_acc), np.mean(Test_acc)

    #Accuracy tables and plots
    print("Average Accuracy with split threshold = %.3f" % split)
    print("Training: %.3f" % train_acc)
    print("Testing: %.3f \n" % test_acc)

    plt.title("Prediction accuracy on train and test set split=%.3f" % split)
    plt.plot(np.arange(n), Train_acc, label="train", color="b")
    plt.plot([0, n], [train_acc, train_acc], label="avg-train", linestyle=":", color="b")
    plt.plot(np.arange(n), Test_acc, label="test", color="y")
    plt.plot([0, n], [test_acc, test_acc], label="avg-test", linestyle=":", color="y")
    plt.xlabel("n"), plt.ylabel("accuracy %")
    #plt.legend(), plt.grid(), plt.show()

    #ROC curves
    p_test = np.ravel(p_test)
    y_probas = np.zeros((len(p_test), 2))
    y_probas[:, 0] = 1 - p_test
    y_probas[:, 1] = p_test

    plot_cumulative_gain(y_true=y_test, y_probas=y_probas)
    #plt.show()
