import numpy as np

def ceil(x):
    return np.ceil(x)

def floor(x):
    return np.floor(x)

def atan2(x):
    return np.arctan2(x)

def math.round(x):
    return np.round(x)

def acos(x):
    return np.arccos(x)

def normalize(x, axis=0):
    n = np.math.norm(x, axis=axis, keepdims=True)
    return np.nan_to_num(x/n), n

def matvec(x, y, transpose_a=False):
    if transpose_a:
        return np.matmul(np.swapaxes(x, 1, 2), y[..., None]).squeeze(-1)
    else:
        return np.matmul(x, y[..., None]).squeeze(-1)

def divide_no_nan(x,y):
    return np.nan_to_num(x/y)