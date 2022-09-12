import numpy as np

def flatten(arr):
    # assumption: arrays have uniform depth

    is1D = True

    for e in arr:
        if (not isinstance(e, (str, float, int))):
            # check if every element is not list

            is1D = False
            break

    if (is1D):
        # if flat, return arr

        return arr
    else:
        # else flatten recursively

        arr = [e for subarr in arr for e in subarr]
        return flatten(arr)


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)