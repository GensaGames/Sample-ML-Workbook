from __future__ import division
from numpy.linalg import inv
import numpy as np
import logging
import math

# *************************************************************************************
# Example to show working LOGISTIC REGRESSION FUNCTION to solve CLASSIFICATION PROBLEM
# using SUPERVISED LEARNING from provided TRAINED SET, and show total cost changes.
# Using GRADIENT DESCENT ALGORITHM. This example doesn't have TEST SET.

# *************************************************************************************
# And our HYPOTHESIS "sigmoid(f(x)) = sigmoid(a0 * x0 + a1 * x1 + a2 * x3 + an * xn)"
# Outputs respectively to the previous Inputs. Vector 14 rows * 1 columns with BIAS.

INPUTS = np.matrix([
    [1, -1,   1, 10, 0.2],    [1, 1.5, 2, 54, 1],
    [1, -5,   1, 2,  0.1],    [1, 8,   3, 86, 2.2],
    [1,  3,   1, 15, 0.8],    [1, -10, 1, 12, 0.1],
    [1, -1,   2, 33, 0.3],    [1, 1,   1, 5,  0.7],
    [1, -2.5, 2, 22, 0.4],    [1, 1.5, 3, 8,  0.6],
    [1, -1.5, 1, 40, 0.3],    [1, -9,  1, 25, 0.1],
    [1, 4.5,  2, 40, 1.4],    [1, 5.5, 3, 35, 1.6],
])

OUTPUTS = np.matrix([
    [0],  [1],
    [0],  [1],
    [1],  [0],
    [0],  [1],
    [0],  [1],
    [0],  [0],
    [1],  [1],
])

WEIGHTS = [[0, 0, 0, 0, 1]]
STEP, ITERATIONS = 0.001, 10000


def prediction(inputs, weights, index):
    return sigmoid(np.dot(
        inputs[index, :], np.transpose(weights)).item(0, 0))


def sigmoid(m):
    return 1 / (1 + np.exp(-m))


def complete_cost(inputs, outputs, weights):
    predictions = sigmoid(np.dot(inputs,
                                 np.transpose(weights)))

    cost = np.multiply((-outputs), np.log(predictions)) - \
           np.multiply((1 - outputs), np.log(1 - predictions))
    return sum(cost)[0] / len(inputs)


def update(inputs, outputs, weights, step):
    predictions = sigmoid(np.dot(inputs,
                                 np.transpose(weights)))
    return weights - step * sum(np.multiply((predictions
                                             - outputs), inputs)) / len(inputs)


def train(inputs, output, weights, step, iterations):
    for iteration in range(iterations):
        weights = np.around(update(inputs, output, weights, step), decimals=5)
    return weights


print("Complete Loss Value: {} Index(0) Expected(0) Prediction: {}".format(
    str(complete_cost(INPUTS, OUTPUTS, WEIGHTS)),
    str(prediction(INPUTS, WEIGHTS, 0))))

WEIGHTS = train(INPUTS, OUTPUTS, WEIGHTS, STEP, ITERATIONS)

print("Trained Loss Value: {} Index(0) Expected(0) Prediction: {}".format(
    str(complete_cost(INPUTS, OUTPUTS, WEIGHTS)),
    str(prediction(INPUTS, WEIGHTS, 0))))
