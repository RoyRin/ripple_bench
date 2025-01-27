import numpy as np 


def invert_logistic(y):
    """ Inverts the logistic function for a given output y (0 < y < 1). """
    if y <= 0 or y >= 1:
        raise ValueError("y must be between 0 and 1, exclusive.")
    return -np.log((1 / y) - 1)