import numpy as np

def ceil(*args, **kwargs):
    return np.ceil(*args, **kwargs)

def floor(*args, **kwargs):
    return np.floor(*args, **kwargs)

def atan2(*args, **kwargs):
    return np.arctan2(*args, **kwargs)

def round(*args, **kwargs):
    return np.round(*args, **kwargs)

def acos(*args, **kwargs):
    return np.arccos(*args, **kwargs)

def divide_no_nan(x,y):
    return np.nan_to_num(x/y)
