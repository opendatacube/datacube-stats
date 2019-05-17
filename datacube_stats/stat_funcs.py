"""
Functions for performing statistical data analysis.
"""
import collections
from functools import reduce as reduce_
from operator import mul as mul_op

import numpy as np

try:
    from bottleneck import anynan, nansum
except ImportError:
    nansum = np.nansum

    def anynan(x, axis=None):
        return np.isnan(x).any(axis=axis)


def argnanmedoid(x, axis=1):
    """
    Return the indices of the medoid

    :param x: input array
    :param axis: axis to medoid along
    :return: indices of the medoid
    """
    if axis == 0:
        x = x.T

    invalid = anynan(x, axis=0)
    band, time = x.shape
    diff = x.reshape(band, time, 1) - x.reshape(band, 1, time)
    dist = np.sqrt(np.sum(diff * diff, axis=0))  # dist = np.linalg.norm(diff, axis=0) is slower somehow...
    dist_sum = nansum(dist, axis=0)
    dist_sum[invalid] = np.inf
    i = np.argmin(dist_sum)

    return i


def medoid_indices(arr, invalid=None):
    """
    The indices of the medoid.

    :arg arr: input array
    :arg invalid: mask for invalid data containing NaNs
    """
    # vectorized version of `argnanmedoid`
    bands, times, ys, xs = arr.shape

    diff = (arr.reshape(bands, times, 1, ys, xs) -
            arr.reshape(bands, 1, times, ys, xs))

    dist = np.linalg.norm(diff, axis=0)
    dist_sum = nansum(dist, axis=0)

    if invalid is None:
        # compute it in case it's not already available
        invalid = anynan(arr, axis=0)

    dist_sum[invalid] = np.inf
    return np.argmin(dist_sum, axis=0)


def prod(a):
    """Product of a sequence"""
    return reduce_(mul_op, a, 1)


def _blah(shape, step=1, dtype=None):
    return np.arange(0, prod(shape) * step, step, dtype=dtype).reshape(shape)


def axisindex(a, index, axis=0):
    """
    Index array 'a' using 'index' as depth along 'axis'
    """
    shape = np.array(index.shape, dtype=np.int32)
    idx = np.zeros(a.ndim, dtype=np.int32)
    idx[axis] = 1
    idx[idx==0] = shape
    # require numpy >= 1.15.0
    return np.take_along_axis(a, index.reshape(idx), axis=axis).reshape(shape)


def section_by_index(array, index, axis=0):
    """
    Take the slice of `array` indexed by entries of `index`
    along the specified `axis`.
    """
    # alternative `axisindex` implementation

    # that avoids the index arithmetic
    # uses `numpy` fancy indexing instead

    # possible index values for each dimension represented
    # as `numpy` arrays all having the shape of `index`
    indices = np.ix_(*[np.arange(dim) for dim in index.shape])

    # the slice is taken along `axis`
    # except for the array `index` itself, the other indices
    # do nothing except trigger `numpy` fancy indexing
    fancy_index = indices[:axis] + (index,) + indices[axis:]

    # result has the same shape as `index`
    return array[fancy_index]


def argpercentile(a, q, axis=0):
    """
    Compute the index of qth percentile of the data along the specified axis.
    Returns the index of qth percentile of the array elements.
    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    q : float in range of [0,100] (or sequence of floats)
        Percentile to compute which must be between 0 and 100 inclusive.
    axis : int or sequence of int, optional
        Axis along which the percentiles are computed. The default is 0.
    """
    q = np.array(q, dtype=np.float64, copy=True) / 100.0
    nans = np.isnan(a).sum(axis=axis)
    q = q.reshape(q.shape + (1,) * nans.ndim)
    index = np.round(q * (a.shape[axis] - 1 - nans)).astype(np.int32)
    # NOTE: assuming nans are gonna sort larger than everything else
    return axisindex(np.argsort(a, axis=axis), index, axis=axis)


def nan_percentile(arr, q, axis=0):
    """
    Return requested percentile(s) of a 3D array, ignoring NaNs

    For the case of 3D->2D reductions, this function is ~200x faster than np.nanpercentile()

    See http://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/ for further explanation

    :param np.ndarray arr:
    :param q: number between 0-100, or list of numbers between 0-100
    :param int axis: must be zero, for compatibility with :meth:`xarray.Dataset.reduce`
    """
    if axis != 0:
        raise ValueError('This function only works with axis=0')

    # valid (non NaN) observations along the first axis
    valid_obs = np.sum(np.isfinite(arr), axis=0)
    # replace NaN with maximum
    max_val = np.nanmax(arr)
    arr[np.isnan(arr)] = max_val
    # sort - former NaNs will move to the end
    arr = np.sort(arr, axis=0)

    # loop over requested quantiles
    if isinstance(q, collections.Sequence):
        qs = []
        qs.extend(q)
    else:
        qs = [q]
    if len(qs) < 2:
        quant_arr = np.zeros(shape=(arr.shape[1], arr.shape[2]))
    else:
        quant_arr = np.zeros(shape=(len(qs), arr.shape[1], arr.shape[2]))

    result = []
    for quant in qs:
        # desired position as well as floor and ceiling of it
        k_arr = (valid_obs - 1) * (quant / 100.0)
        f_arr = np.floor(k_arr).astype(np.int32)
        c_arr = np.ceil(k_arr).astype(np.int32)
        fc_equal_k_mask = f_arr == c_arr

        # linear interpolation (like np percentile) takes the fractional part of desired position
        floor_val = axisindex(a=arr, index=f_arr) * (c_arr - k_arr)
        ceil_val = axisindex(a=arr, index=c_arr) * (k_arr - f_arr)

        quant_arr = floor_val + ceil_val
        # if floor == ceiling take floor value
        quant_arr[fc_equal_k_mask] = axisindex(a=arr, index=k_arr.astype(np.int32))[fc_equal_k_mask]

        result.append(quant_arr)

    if len(result) == 1:
        return result[0]
    else:
        return result


def _compute_medoid(data, index_dtype='int16'):
    flattened = data.to_array(dim='variable')
    variable, time, y, x = flattened.shape
    index = np.empty((y, x), dtype=index_dtype)
    for iy in range(y):
        for ix in range(x):
            index[iy, ix] = argnanmedoid(flattened.values[:, :, iy, ix])
    return index
