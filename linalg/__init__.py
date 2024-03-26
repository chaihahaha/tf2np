def matvec(x, y, transpose_a=False):
    if transpose_a:
        return np.matmul(np.swapaxes(x, 1, 2), y[..., None]).squeeze(-1)
    else:
        return np.matmul(x, y[..., None]).squeeze(-1)