"""
Useful utilities used in Stats
"""
import itertools
import pickle
import functools
from typing import Dict, Iterator, Tuple, Iterable, Any

import cloudpickle
import numpy as np
import xarray
import click
from datetime import timezone
from datetime import timedelta

from datacube.api.query import Query

from datacube.storage.masking import mask_invalid_data, create_mask_value
from datacube.api import Tile
from datacube.model import Range
from datacube.ui.task_app import pickle_stream, unpickle_stream


def tile_iter(tile: Tile, chunk_size: Dict[str, int]) -> Iterator[Tuple[None, slice, slice]]:
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


def first(xs):
    """ Get first element from a sequence
    """
    return list(itertools.islice(xs, 1))[0]


def first_var(ds):
    """ Get first data variable from dataset
    """
    return first(ds.data_vars.values())


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


def da_is_float(da):
    """
    Check if DataArray is of floating point type
    """
    assert hasattr(da, 'dtype')

    return da.dtype.kind == 'f'


def ds_all_float(ds: xarray.Dataset):
    """
    Check if dataset contains only floating point arrays
    """
    assert isinstance(ds, xarray.Dataset)

    for da in ds.data_vars.values():
        if not da_is_float(da):
            return False
    return True


def da_nodata(da, default=None):
    """
    Lookup `nodata` property of DataArray

    Returns:
      nodata if set
      default if supplied, otherwise

      NaN for floating point arrays
      0   for everything else
    """
    nodata = getattr(da, 'nodata', None)
    if nodata is not None:
        return nodata

    if default is not None:
        return default

    if da_is_float(da):
        return np.nan

    # integer like but has no 'nodata' attribute and default wasn't specified
    return 0


def nodata_like(ds):
    """Similar to xarray.full_like but filled with nodata value or with NaN for
    floating point variables.

    """
    assert isinstance(ds, (xarray.Dataset, xarray.DataArray))

    def _nodata_like_da(da):
        return xarray.full_like(da, da_nodata(da))

    if isinstance(ds, xarray.DataArray):
        return _nodata_like_da(ds)

    return ds.apply(_nodata_like_da, keep_attrs=True)


def sensible_where_inplace(data, mask):
    """
    Apply mask in-place without creating new storage or converting to float

    data -- Dataset or DataArray, if dataset applies mask to all data variables
    mask -- DataArray or ndarray of the same x,y shape as data

    If mask has no time dimension and data does, it will be broadcast along time dimension

    Does equivalent of:
       `data[mask == False] = nodata` or `data[:, mask == False] = nodata`

    """
    mask = xarray.ufuncs.logical_not(mask)
    try:
        mask = mask.values
    except AttributeError:
        # it is already an ndarray
        pass

    def proc_var(a):
        if a.shape == mask.shape:
            a.values[mask] = da_nodata(a)
        elif mask.shape == a.shape[1:]:
            # note this assumes time dimension goes first
            a.values[:, mask] = da_nodata(a)
        else:
            assert "Incompatible mask shape"

        return a

    if isinstance(data, xarray.DataArray):
        return proc_var(data)

    assert isinstance(data, xarray.Dataset)

    for a in data.data_vars.values():
        proc_var(a)

    return data


def _convert_to_floats(data):
    assert isinstance(data, xarray.Dataset)

    if ds_all_float(data):
        return data

    def to_float(da):
        if da_is_float(da):
            return da

        out = da.astype(np.float32)

        nodata = getattr(da, 'nodata', None)
        if nodata is None:
            return out

        return out.where(da != nodata)

    return data.apply(to_float, keep_attrs=True)


def cast_back(data: xarray.Dataset, measurements: Iterable[Dict[str, Any]]) -> xarray.Dataset:
    """
    Cast calculated statistic `Dataset` into intended data types.
    When going through intermediate representation as floats,
    restore `nodata` values in place of `NaN`s.
    """
    assert isinstance(data, xarray.Dataset)
    measurements = {measurement['name']: measurement
                    for measurement in measurements}

    data_vars = [name for name in data.data_vars]
    assert set(data_vars) == set(measurements.keys())

    def cast(da):
        """ Cast `DataArray` into intended type. """
        output_measurement = measurements[da.name]
        expected_dtype = np.dtype(output_measurement['dtype'])
        actual_dtype = da.dtype

        if actual_dtype.kind != 'f' or 'nodata' not in output_measurement:
            # did not go through intermediate representation
            # or nodata is unspecified
            if expected_dtype == actual_dtype:
                return da
            else:
                return da.astype(expected_dtype)

        # replace NaNs with nodata
        nans = np.isnan(da.values)
        clone = da.astype(expected_dtype)
        clone.values[nans] = output_measurement['nodata']
        return clone

    return data.apply(cast, keep_attrs=True)


# pylint: disable=invalid-name
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


def _mk_masker_to_bool(m, v, da):
    return (da & m) == v


def _mk_masker_to_bool_inverted(m, v, da):
    return (da & m) != v


def mk_masker(m, v, invert=False):
    """

    Construct function that converts bit array to boolean given mask and
    expected value after masking.

    x => (x & m) == v
    x => (x & m) != v , when invert == True

    """
    to_bool = functools.partial(_mk_masker_to_bool, m, v)
    to_bool_inverted = functools.partial(_mk_masker_to_bool_inverted, m, v)

    return to_bool_inverted if invert else to_bool


def make_numpy_mask(defn):
    def numpy_mask(variable, **flags):
        """
        :rtype: ndarray
        """
        mask, mask_value = create_mask_value(defn, **flags)

        return variable & mask == mask_value

    return numpy_mask


# pylint: disable=invalid-name
wofs_mask = make_numpy_mask(wofs_flag_defs)


def wofs_fuser(dest, src):
    valid = wofs_mask(src, noncontiguous=False)

    np.copyto(dest, src, where=valid)

    invalid = (wofs_mask(dest, wet=True) & wofs_mask(src, dry=True)) | (
        wofs_mask(src, wet=True) & wofs_mask(dest, dry=True))
    np.copyto(dest, 2, where=invalid)
    return dest


def tile_flatten_sources(tile):
    """
    Extract sources from tile as a flat list of Dataset objects,
    this removes any grouping that might have been applied to tile sources
    """
    return functools.reduce(list.__add__, [list(a.item()) for a in tile.sources])


def report_unmatched_datasets(co_unmatched, logger=None):
    """ Printout in "human" format unmatched datasets

    co_unmatched -- dict (int,int) => Tile
    logger -- function that logs string, by default will print to stdout

    returns number of datasets that were skipped
    """
    def default_logger(s):
        print(s)

    logger = default_logger if logger is None else logger
    n = 0

    for cell_idx, tile in co_unmatched.items():
        dss = tile_flatten_sources(tile)

        if len(dss) == 0:
            continue

        n += len(dss)

        logger('Skipping files in tile {},{}'.format(*cell_idx))

        for ds in dss:
            logger(' {} {}'.format(ds.id, ds.local_path))

    return n


def bunch(**kw):
    """
    Create object with given attributes
    """
    x = type('bunch', (object, ), {})()
    for k, v in kw.items():
        setattr(x, k, v)
    return x


def sorted_interleave(*iterators, key=lambda x: x, reverse=False):
    """
    Given a number of sorted sequences return a single sorted sequence avoiding
    looking ahead as much as possible. Supports infinite sequences, loads one
    item at a time from each sequence at the most.
    """
    def advance(it):
        try:
            return (next(it), it)
        except StopIteration:
            return None

    vv = map(advance, iterators)
    vv = list(filter(lambda x: x is not None, vv))

    while len(vv) > 0:
        (val, it), *vv = sorted(vv, key=lambda a: key(a[0]), reverse=reverse)

        yield val
        del val

        x = advance(it)
        if x is not None:
            vv.append(x)


def _find_periods_with_data(index, product_names, period_duration='1 day',
                            start_date='1985-01-01', end_date='2000-01-01'):
    """
    Search the datacube and find which periods contain data

    This is very useful when running stats in the `daily` mode (which outputs a file for each day). It is
    very slow to create an output for every day regardless of data availability, so it is better to only find
    the useful days at the beginning.

    :return: sequence of (start_date, end_date) tuples
    """
    # TODO: Read 'simple' job configuration from file
    # TODO: need get rid of the hard-coded query
    query = dict(y=(-41*(40000-1600), -41*40000), x=(15*40000, 15*(40000+1600)),
                 crs='EPSG:3577', time=(start_date, end_date))

    valid_dates = set()
    for product in product_names:
        counts = index.datasets.count_product_through_time(period_duration, product=product,
                                                           **Query(**query).search_terms)
        for time_range, count in counts:
            if count > 0:
                time_range = Range(time_range.begin.astimezone(timezone.utc),
                                   time_range.end.astimezone(timezone.utc))
                valid_dates.add(time_range)
    for time_range in sorted(valid_dates):
        yield time_range.begin, time_range.end


class Slice(click.ParamType):
    name = 'slice'

    def convert(self, value, param, ctx):
        if value is None:
            return None

        try:
            words = [None if word == '' else int(word)
                     for word in value.split(':')]

            if len(words) > 3:
                raise ValueError

            return slice(*words)

        except ValueError:
            self.fail('Invalid Python slice')


def prettier_slice(sl):
    def to_str(x):
        if x is None:
            return ''
        return str(x)

    return "[{}:{}:{}]".format(to_str(sl.start), to_str(sl.stop), to_str(sl.step))
