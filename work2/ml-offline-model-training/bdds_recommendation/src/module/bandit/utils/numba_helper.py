import numpy as np
from numba import jit


@jit(nopython=True)
def numba_zeros(shape):

    return np.zeros(shape)


@jit(nopython=True)
def numba_identity(shape):

    return np.identity(shape)


@jit(nopython=True)
def numba_transpose(x, axis=None):

    return np.transpose(x, axis)


@jit(nopython=True)
def numba_dot(x, y):

    return x.dot(y)


@jit(nopython=True)
def numba_sqrt(x):

    return np.sqrt(x)


@jit(nopython=True)
def numba_linalg_inv(x):

    return np.linalg.inv(x)


@jit(nopython=True)
def numba_outer(x, y):

    return np.outer(x, y)


@jit(nopython=True)
def numba_addition(x, y):

    return x + y


@jit(nopython=True)
def numba_multiplication(x, y):

    return x * y
