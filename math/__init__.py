import numpy as np

def ceil(x):
    return np.ceil(x)

def floor(x):
    return np.floor(x)

def atan2(x):
    return np.arctan2(x)

def round(x):
    return np.round(x)

def acos(x):
    return np.arccos(x)

def normalize(x, axis=0):
    n = np.math.norm(x, axis=axis, keepdims=True)
    return np.nan_to_num(x/n), n

def divide_no_nan(x,y):
    return np.nan_to_num(x/y)