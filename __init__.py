import numpy as np
from numpy import *

from . import math
from . import raw_ops
from . import linalg

def rank(x):
    return len(x.shape)

def fill(dims, value):
    return np.full(dims, value)

def unique_with_counts(x):
    return np.unique(x, return_counts=True, return_inverse=True)


def where(*args, **kwargs):
    ret = np.where(*args, **kwargs)
    if isinstance(ret, tuple):
        return np.stack(ret, axis=-1)
    else:
        return ret

def unique(x):
    return np.unique(x, return_index=True)

def complex(x, y):
    return x+1j * y

def tensor_scatter_nd_update(tensor, indices, updates):
    t = tensor.copy()
    idx = np.array(indices, int)
    inds = tuple(idx.reshape(-1, idx.shape[-1]).T)
    t[inds] = updates
    # buggy when updates shape not broadcastable
    return t

def gather(params, indices, axis=0, batch_dims=None):
    params = np.array(params)
    indices = np.array(indices)
    isp = list(indices.shape)
    psp = list(params.shape)
    ibsp = list(isp[batch_dims:])
    pbsp = list(psp[batch_dims:])
    resultsp = psp[:axis] + ibsp + psp[axis+1:]
    if batch_dims:
        baxis = axis - batch_dims
    else:
        baxis = axis
    n_batches = np.prod(psp[:axis], dtype=int)
    pbs = params.reshape([n_batches] + pbsp)
    ibs = indices.reshape([n_batches] + ibsp)
    res = []
    for bp, bi in zip(pbs, ibs):
        taked = np.take(bp, bi.flatten(), axis=baxis)
        res.append(taked)
    res = np.stack(res, axis=0).reshape(resultsp)
    return res

def cast(x, dtype=np.float32):
    return np.array(x, dtype=dtype)

def reduce_sum(*args, **kwargs):
    return np.sum(*args, **kwargs)

def reduce_any(*args, **kwargs):
    return np.any(*args, **kwargs)

def reduce_mean(*args, **kwargs):
    return np.mean(*args, **kwargs)

def reduce_prod(*args, **kwargs):
    return np.prod(*args, **kwargs)

def reduce_max(*args, **kwargs):
    return np.max(*args, **kwargs)

def concat(*args, **kwargs):
    return np.concatenate(*args, **kwargs)

def maximum(*args, **kwargs):
    return np.max(*args, **kwargs)

def Variable(*args, **kwargs):
    return np.array(*args, **kwargs)

def range(*args, **kwargs):
    return np.arange(*args, **kwargs)

def constant(*args, **kwargs):
    return np.array(*args, **kwargs)

def clip_by_value(t, clip_value_min, clip_value_max):
    return np.clip(t, clip_value_min, clip_value_max)

def stop_gradient(x):
    return x

def dtype(x):
    return x.dtype
