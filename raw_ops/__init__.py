import numpy as np

def UniqueV2(x, axis=0):
    return np.unique(x, return_inverse=True, axis=axis)
