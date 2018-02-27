# Perceptron Learning

import pandas as pd
import numpy as np
from random import random

# global parameters
learn_rate = 0.1
n_epochs = 100


# perceptron threshold function
def sgn(z):
    if z > 0:
        return 1
    else:
        return -1


# import data set
df = pd.read_csv('d-100.csv')
n_rows, n_cols = df.shape

# randomly initialize weights + bias
weights = np.array([random() for w in range(n_cols)])

for epoch in range(n_epochs):

    errors = 0.0
    for i, row in df.iterrows():

        # appending additional input with value (1.0) for bias
        xs = np.append(row.values[:-1], 1.0)

        # classification label for training example
        target = row.values[-1]

        # summation for linear unit
        z = np.dot(weights, xs)

        # output from perceptron
        out = sgn(z)

        # Check for misclassification and increment errors
        if target != out:
            errors += 1.0

        delta = learn_rate * (target - out) * xs

        # update weights
        weights += delta

    # Display summary
    print("epoch {}: {} errors".format(epoch, errors))

