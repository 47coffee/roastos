import numpy as np

# evaluate flavor error

def flavor_cost(predicted, target):

    p = np.array(list(predicted.values()))
    t = np.array(list(target.values()))

    return np.sum((p - t) ** 2)