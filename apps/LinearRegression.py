from __future__ import division
import numpy as np
import logging

# *************************************************************************************
# Example to show working LINEAR REGRESSION FUNCTION to solve simple REGRESSION PROBLEM
# using SUPERVISED LEARNING from provided TRAINED SET, and show total cost changes.
# Using GRADIENT DESCENT ALGORITHM. This example doesn't have TEST SET.

# *************************************************************************************
# We have single input and continuous(regression) single output.
# And our HYPOTHESIS "f(x) = a0 + a1 * x" (linear function)

INPUTS = [[1, 2],
          [1, 3],
          [1, 6]]
OUTPUTS, STEP, ITERATIONS = [[3], [5], [11]], 0.1, 200
WEIGHTS = [[1, 1]]


def complete_cost(inputs, outputs, weights):
    return sum((np.dot(inputs, np.transpose(weights))
                - outputs) ** 2)[0] / 2 * len(inputs)


def update(inputs, outputs, weights, step):
    return weights - step * sum(np.multiply((np.dot(
        inputs, np.transpose(weights)) - outputs), inputs)) / len(inputs)


def work(iterations, inputs, outputs, weights, step):
    for iteration in range(iterations):
        weights = np.around(update(inputs, outputs, weights, step), decimals=5)
        if iteration % 100:
            print("Trained value WEIGHTS: {} Total Loss Value: {} "
                  .format(str(weights), str(complete_cost(inputs, outputs, weights))))
    return weights


WEIGHTS = work(ITERATIONS, INPUTS, OUTPUTS, WEIGHTS, STEP)

print("Trained value WEIGHTS: {}   Total Loss Value: {} "
      .format(str(WEIGHTS), str(complete_cost(INPUTS, OUTPUTS, WEIGHTS))))
