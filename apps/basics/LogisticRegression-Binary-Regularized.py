from __future__ import division
import numpy as np

# *************************************************************************************
# Example to show working LOGISTIC REGRESSION FUNCTION to solve CLASSIFICATION PROBLEM
# using SUPERVISED LEARNING from provided TRAINED SET, and show total cost changes.
# Using GRADIENT DESCENT ALGORITHM. This example doesn't have TEST SET.

# *************************************************************************************
# This example using REGULARIZED prefix to our LOSS FUNCTION and LEARNING ALGORITHM

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
STEP, ITERATIONS, LAM = 0.001, 10000, 1


def prediction(inputs, weights, index):
    return sigmoid(np.dot(inputs[index, :],
                          np.transpose(weights)).item(0, 0))


def sigmoid(m):
    return 1 / (1 + np.exp(-m))


def regularization_cost(inputs, weights, lam):
    return (lam / len(inputs) * 2) * np.sum(np.square(weights))


def complete_cost(inputs, outputs, weights, lam):
    predictions = sigmoid(np.dot(inputs,
                                 np.transpose(weights)))
    regularized = regularization_cost(inputs, weights, lam)

    complete = np.multiply((-outputs), np.log(predictions)) - \
               np.multiply((1 - outputs), np.log(1 - predictions))
    avg = np.sum(complete) / len(inputs)
    return avg + regularized


def update(inputs, outputs, weights, step, lam):
    predictions = sigmoid(np.dot(inputs,
                                 np.transpose(weights)))
    regularization = (lam / len(inputs)) * np.sum(weights)

    return weights - step * (np.sum(
        np.multiply((predictions - outputs), inputs), 0) / len(inputs)
                             + regularization)


def train(inputs, output, weights, step, iterations, lam):
    for iteration in range(iterations):
        weights = np.around(update(inputs, output,
                                   weights, step, lam), decimals=5)
    return weights


print("Complete Loss Value: {} \n Index(0) Expected(0) Prediction: {}"
      .format(str(complete_cost(INPUTS, OUTPUTS, WEIGHTS, LAM)),
              str(prediction(INPUTS, WEIGHTS, 0))))

WEIGHTS = train(INPUTS, OUTPUTS, WEIGHTS, STEP, ITERATIONS, LAM)

print("Trained Loss Value:  {} \n Index(0) Expected(0) Prediction: {}"
      .format(str(complete_cost(INPUTS, OUTPUTS, WEIGHTS, LAM)),
              str(prediction(INPUTS, WEIGHTS, 0))))
