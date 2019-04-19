from __future__ import division
from collections import OrderedDict
import numpy as np


# **********************
# Example to show working NEURAL NETWORK to solve CLASSIFICATION PROBLEM
# using SUPERVISED LEARNING from provided TRAINED SET, and show total cost changes.
# Using BACKPROPAGATION ALGORITHM. This example doesn't have TEST SET.

# **********************
# This example using REGULARIZED prefix to our LOSS FUNCTION and LEARNING ALGORITHM

# **********************
# Our model it's INPUT LAYER with 4 units(Features) w/o Bias HIDDEN LAYER with 8
# and OUTPUT LAYER with 3 Units.
import scipy.optimize


def sigmoid(m):
    return 1 / (1 + np.exp(-m))


def sigmoid_gradient(m):
    _sigmoid = sigmoid(m)
    return _sigmoid * (1 - _sigmoid)


def feed_forward(x, theta1, theta2):
    hidden_dot = np.dot(add_bias(x), np.transpose(theta1))
    hidden_p = add_bias(sigmoid(hidden_dot))

    p = sigmoid(np.dot(hidden_p, np.transpose(theta2)))
    return hidden_dot, hidden_p, p


def cost(thetas, x, y, hidden, lam):
    theta1, theta2 = get_theta_from(thetas, x, y, hidden)
    _, _, p = feed_forward(x, theta1, theta2)

    regularization = (lam / (len(x) * 2)) * (
        np.sum(np.square(np.delete(theta1, 0, 1)))
        + np.sum(np.square(np.delete(theta2, 0, 1))))

    complete = np.nan_to_num(np.multiply((-y), np.log(
        p)) - np.multiply((1 - y), np.log(1 - p)))
    avg = np.sum(complete) / len(x)
    return avg + regularization


def vector(z):
    # noinspection PyUnresolvedReferences
    return np.reshape(z, (np.shape(z)[0], 1))


def add_bias(z):
    return np.append(np.ones((z.shape[0], 1)), z, axis=1)


def gradient(thetas, x, y, hidden, lam):
    theta1, theta2 = get_theta_from(thetas, x, y, hidden)
    hidden_dot, hidden_p, p = feed_forward(x, theta1, theta2)

    error_o = p - y
    error_h = np.multiply(np.dot(
        error_o, theta2),
        sigmoid_gradient(add_bias(hidden_dot)))

    x = add_bias(x)
    error_h = np.delete(error_h, 0, 1)

    theta1_grad, theta2_grad = \
        np.zeros(theta1.shape[::-1]), np.zeros(theta2.shape[::-1])
    records = y.shape[0]

    for i in range(records):
        theta1_grad = theta1_grad + np.dot(
            vector(x[i]), np.transpose(vector(error_h[i])))
        theta2_grad = theta2_grad + np.dot(
            vector(hidden_p[i]), np.transpose(vector(error_o[i])))

    reg_theta1 = theta1.copy()
    reg_theta1[:, 0] = 0

    theta1_grad = np.transpose(
        theta1_grad / records) + ((lam / records) * reg_theta1)

    reg_theta2 = theta2.copy()
    reg_theta2[:, 0] = 0

    theta2_grad = np.transpose(
        theta2_grad / records) + ((lam / records) * reg_theta2)

    return np.append(
        theta1_grad, theta2_grad)


def get_theta_shapes(x, y, hidden):
    return (hidden, x.shape[1] + 1), \
           (y.shape[1], hidden + 1)


def get_theta_from(thetas, x, y, hidden):
    t1_s, t2_s = get_theta_shapes(x, y, hidden)
    split = t1_s[0] * t1_s[1]

    theta1 = np.reshape(thetas[:split], t1_s)
    theta2 = np.reshape(thetas[split:], t2_s)
    return theta1, theta2


def rand_init(l_out, l_in):
    epsilon_init = (np.math.sqrt(6)) / (np.math.sqrt(l_in + l_out))
    return np.random.rand(l_out, l_in) * 2 * epsilon_init - epsilon_init


def get_binary_y(y):
    variants = list(OrderedDict.fromkeys(Y_LOADED))

    return np.array(list(map(
        lambda x: [int(item == x) for item in variants], y)))


def train(x, y, hidden_size, lam):
    y = get_binary_y(y)

    t1_s, t2_s = get_theta_shapes(x, y, hidden_size)
    thetas = np.append(
        rand_init(t1_s[0], t1_s[1]),
        rand_init(t2_s[0], t2_s[1]))

    initial_cost = cost(thetas, x, y, hidden_size, lam)
    print("Starting Loss: " + str(initial_cost))

    check_grad_val = scipy.optimize.check_grad(
        cost, gradient, thetas, x, y, hidden_size, lam)
    print("Check gradient: " + str(check_grad_val))

    trained_theta = scipy.optimize.fmin_bfgs(
        cost, thetas, fprime=gradient, args=(x, y, hidden_size, lam))

    print("Trained Loss: " +
          str(cost(trained_theta, x, y, hidden_size, lam)))
    return None


# **********************
# In this income data, we assume that last column (rank) will be Y VALUE,
# and first 3 columns represent X DATA, to train our NN
X_LOADED = np.loadtxt('../resource/iris.data', delimiter=',',
                      dtype=np.float64, usecols=range(4))
Y_LOADED = np.loadtxt('../resource/iris.data', delimiter=',',
                      dtype=np.str_, usecols=4)
train(X_LOADED, Y_LOADED, 8, 0.5)
