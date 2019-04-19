from __future__ import division
from collections import OrderedDict
import numpy as np
import scipy.optimize

# **********************
# Example to show working LOGISTIC REGRESSION FUNCTION to solve CLASSIFICATION PROBLEM
# using SUPERVISED LEARNING from provided TRAINED SET, and show total cost changes.
# Using GRADIENT DESCENT ALGORITHM. This example doesn't have TEST SET.

# **********************
# This example using REGULARIZED prefix to our LOSS FUNCTION and LEARNING ALGORITHM

# **********************
# And our HYPOTHESIS "sigmoid(f(x)) = sigmoid(a0 * x0 + a1 * x1 + a2 * x3 + an * xn)"
# Outputs respectively to the previous Inputs. Vector 14 rows * 1 columns with BIAS.


def prediction(x, theta):
    return sigmoid(np.dot(x, np.transpose(theta)))


def sigmoid(m):
    return 1 / (1 + np.exp(-m))


def cost(theta, x, y, lam):
    theta = theta.reshape(1, len(theta))
    predictions = sigmoid(np.dot(x, np.transpose(theta))).reshape(len(x), 1)

    regularization = (lam / (len(x) * 2)) * np.sum(np.square(np.delete(theta, 0, 1)))

    complete = np.multiply((-y), np.log(predictions)) - \
               np.multiply((1 - y), np.log(1 - predictions))
    avg = np.sum(complete) / len(x)
    return avg + regularization


def gradient(theta, x, y, lam):
    theta_len = len(theta)
    theta = theta.reshape(1, theta_len)

    predictions = sigmoid(np.dot(x, np.transpose(theta))).reshape(len(x), 1)

    theta_wo_bias = theta.copy()
    theta_wo_bias[0, 0] = 0

    assert (theta_wo_bias.shape == theta.shape)
    regularization = np.squeeze(
        ((lam / len(x)) * theta_wo_bias).reshape(theta_len, 1))

    return np.sum(np.multiply((predictions - y), x), 0) / len(x) + regularization


def train(x, y, variations, lamb):
    _, size = np.shape(x)
    thetas = np.zeros([1, size])

    for variant in variations:
        initial = np.zeros(size)
        _y = np.array([int(item == variant) for item in y])\
            .reshape(len(y), 1)

        theta = scipy.optimize.fmin_bfgs(
            cost, initial,fprime=gradient, args=(x, _y, lamb))

        thetas = np.vstack([thetas, theta])
    return np.delete(thetas, 0, 0)


def predict(thetas, x):
    _p, _class = -1, -1
    for theta in thetas:
        value = prediction(x, theta)
        if value > _p:
            _p, _class = value, \
                         np.where(thetas == theta)[0][0]
    return _class


# **********************
# In this income data, we assume that last column (rank) will be Y VALUE,
# and first 3 columns represent X DATA, to train our LOGISTIC REGRESSION.
X_LOADED = np.loadtxt('../resource/iris.data', delimiter=',',
                      dtype=np.float64, usecols=range(4))
Y_LOADED = np.loadtxt('../resource/iris.data', delimiter=',',
                      dtype=np.str_, usecols=4)

THETAS = train(X_LOADED, Y_LOADED,
               list(OrderedDict.fromkeys(Y_LOADED)), 1)

print("Trained multi classification. Working Item : {} \n"
      "Predicted class (from data set order): {} "
      .format(X_LOADED[110], predict(THETAS, X_LOADED[110])))
