"""
Useful utilities used in Stats
"""
import itertools

import numpy as np
import xarray

from datacube.storage.masking import mask_invalid_data


def tile_iter(tile, chunk_size):
    """
    Return the sequence of chunks to split a tile into computable regions.

    :param Tile tile: a tile of `.shape` size containing `.dim` dimensions
    :param chunk_size: dict of dimension sizes
    :return: Sequence of chunks to iterate across the entire tile
    """
    defaults = tuple(size if dim in chunk_size else None
                     for dim, size in zip(tile.dims, tile.shape))
    steps = _tuplify(tile.dims, chunk_size, defaults)
    return _block_iter(steps, tile.shape)


def _tuplify(keys, values, defaults):
    assert not set(values.keys()) - set(keys), 'bad keys'
    return tuple(values.get(key, default) for key, default in zip(keys, defaults))


def _block_iter(steps, shape):
    return itertools.product(*(_slicify(step, size) for step, size in zip(steps, shape)))


def _slicify(step, size):
    if step is None:
        return [slice(None)]
    else:
        return (slice(i, min(i + step, size)) for i in range(0, size, step))


def sensible_mask_invalid_data(data):
    # TODO This should be pushed up to datacube-core
    # xarray.DataArray.where() converts ints to floats, since NaNs are used to represent nodata
    # by default, this uses float64, which is way over the top for an int16 value, so
    # lets convert to float32 first, to save a bunch of memory.
    data = _convert_to_floats(data)  # This is stripping out variable attributes
    return mask_invalid_data(data)


def sensible_where(data, mask):
    data = _convert_to_floats(data)  # This is stripping out variable attributes
    return data.where(mask)


def _convert_to_floats(data):
    # Use float32 instead of float64 if input dtype is int16
    assert isinstance(data, xarray.Dataset)
    for name, dataarray in data.data_vars.items():
        if dataarray.dtype != np.int16:
            return data
    return data.apply(lambda d: d.astype(np.float32), keep_attrs=True)


def is_wet(data):
    """
    :rtype: np.ndarray
    """
    d = data & ~4  # unset land/sea flag
    return d == 128


def wofs_fuser(dest, src):
    mismatched = (is_wet(dest) & ~is_wet(src)) | (is_wet(src) & ~is_wet(dest))

    np.copyto(dest, dest | src)

    np.copyto(dest, 2, where=mismatched)  # Set to non-contiguous
    return dest
