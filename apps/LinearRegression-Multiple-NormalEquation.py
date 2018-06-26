from __future__ import division
from numpy.linalg import inv
import numpy as np
import logging


# *************************************************************************************
# Example to show working LINEAR REGRESSION FUNCTION to solve REGRESSION PROBLEM
# using SUPERVISED LEARNING from provided TRAINED SET, and show total cost changes.
# Using NORMAL EQUATION ALGORITHM. This example doesn't have TEST SET.


# *************************************************************************************
# Training INPUTS. (Where for ex. 1. Square meters area
# 2. Number of floors 3. Building age 4. Far from Center)
# Note! MATRIX 14 rows * 4 columns.

# *************************************************************************************
# OUTPUTS respectively to the previous Inputs,
# (Where for ex.  ech value it's thousands of $)
# Note! Vector 14 rows * 1 columns.

# *************************************************************************************
# We have several inputs and continuous(regression) single output.
# And our HYPOTHESIS "f(x) = a0 * x0 + a1 * x1 + a2 * x3 + an * xn"

INPUTS = np.matrix([
    [1, 80,  1, 10, 0.2],   [1, 120, 2, 54, 1],
    [1, 55,  1, 2,  0.1],   [1, 220, 3, 86, 2.2],
    [1, 150, 1, 15, 0.8],   [1, 120, 1, 12, 0.1],
    [1, 180, 2, 33, 0.3],   [1, 75,  1, 5,  0.7],
    [1, 135, 2, 22, 0.4],   [1, 155, 3, 8,  0.6],
    [1, 70,  1, 40, 0.3],   [1, 110, 1, 25, 0.1],
    [1, 250, 2, 40, 1.4],   [1, 185, 3, 35, 1.6],
])

OUTPUTS = np.matrix([
    [90],   [174],
    [57],   [306],
    [165],  [132],
    [213],  [80],
    [157],  [163],
    [110],  [135],
    [290],  [220],
])

WEIGHTS = np.random.uniform(low=-1, high=1, size=5)


def prediction(data, weights, index):
    return np.dot(data[index, :], np.transpose(weights)).item(0, 0)


def complete_cost(data, outputs, weights):
    _c = 0
    for i in range(len(data)):
        _c += (prediction(data, weights, i)
               - outputs[i][0]) ** 2
    return _c / (2 * len(data))


def train(data, output):
    weights = np.dot(np.dot(inv(np.dot(np.transpose(data), data)),
                            np.transpose(data)), output)
    return np.transpose(weights)


print("Start Loss Value: {} Index Expected(80) Prediction Value(0): {}".format(
             str(complete_cost(INPUTS, OUTPUTS, WEIGHTS)),
             str(prediction(INPUTS, WEIGHTS, 0))))

WEIGHTS = train(INPUTS, OUTPUTS)

print("Trained Loss Value: {} Index Expected(80) Prediction Value(0): {}".format(
             str(complete_cost(INPUTS, OUTPUTS, WEIGHTS)),
             str(prediction(INPUTS, WEIGHTS, 0))))
