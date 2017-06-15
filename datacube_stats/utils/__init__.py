"""
Useful utilities used in Stats
"""
import itertools

import numpy as np
import xarray

from datacube.storage.masking import mask_invalid_data, create_mask_value


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


wofs_flag_defs = {'cloud': {'bits': 6, 'description': 'Cloudy', 'values': {0: False, 1: True}},
                  'cloud_shadow': {'bits': 5,
                                   'description': 'Cloud shadow',
                                   'values': {0: False, 1: True}},
                  'dry': {'bits': [7, 6, 5, 4, 3, 1, 0],
                          'description': 'No water detected',
                          'values': {0: True}},
                  'high_slope': {'bits': 4,
                                 'description': 'High slope',
                                 'values': {0: False, 1: True}},
                  'nodata': {'bits': 0, 'description': 'No data', 'values': {1: True}},
                  'noncontiguous': {'bits': 1,
                                    'description': 'At least one EO band is missing over over/undersaturated',
                                    'values': {0: False, 1: True}},
                  'sea': {'bits': 2, 'description': 'Sea', 'values': {0: False, 1: True}},
                  'terrain_or_low_angle': {'bits': 3,
                                           'description': 'terrain shadow or low solar angle',
                                           'values': {0: False, 1: True}},
                  'wet': {'bits': [7, 6, 5, 4, 3, 1, 0],
                          'description': 'Clear and Wet',
                          'values': {128: True}}}


def make_numpy_mask(defn):
    def numpy_mask(variable, **flags):
        """
        :rtype: ndarray
        """
        mask, mask_value = create_mask_value(defn, **flags)

        return variable & mask == mask_value

    return numpy_mask


wofs_mask = make_numpy_mask(wofs_flag_defs)


def wofs_fuser(dest, src):
    valid = wofs_mask(src, noncontiguous=False)

    np.copyto(dest, src, where=valid)

    invalid = (wofs_mask(dest, wet=True) & wofs_mask(src, dry=True)) | (
        wofs_mask(src, wet=True) & wofs_mask(dest, dry=True))
    np.copyto(dest, 2, where=invalid)
    return dest
