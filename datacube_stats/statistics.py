"""
Functions for performing statistical data analysis.

Functions:
- argnanmedoid
- nanmedoid
- argpercentile
- nan_percentile

- percentile_stat

Classes:
- ValueStat
- WofsStats
- IndexStat
- StreamedStat
- OneToManyStat
- PerBandIndexStat
- PerStatIndexStat
"""
from __future__ import absolute_import

import abc
import collections
from collections import OrderedDict
from datetime import datetime
from functools import reduce as reduce_, partial
from operator import mul as mul_op

import numpy as np
import xarray
from pkg_resources import iter_entry_points

try:
    from bottleneck import anynan, nansum
except ImportError:
    nansum = np.nansum


    def anynan(x, axis=None):
        return np.isnan(x).any(axis=axis)


class StatsConfigurationError(RuntimeError):
    pass


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


def nanmedoid(x, axis=1):
    i = argnanmedoid(x, axis)

    return x[:, i]


def combined_var_reduction(dataset, method, dim='time', keep_attrs=True):
    """
    Apply a reduction to a dataset by combining data variables into a single ndarray, running `method`, then
    un-combining to separate data variables.

    eg::

        med = combined_var_reduction(data, nanmedoid)

    :param dataset: Input `xarray.Dataset`
    :param method: function to apply to DataArray
    :param bool keep_attrs: Should dataset attributes be retained, defaults to True.
    :param dim: Dimension to apply reduction along
    :return: xarray.Dataset with same data_variables but one less dimension
    """
    flattened = dataset.to_array(dim='variable')

    hdmedian_out = flattened.reduce(_reduce_across_variables, dim=dim, keep_attrs=keep_attrs, method=method)

    hdmedian_out = hdmedian_out.to_dataset(dim='variable')

    if keep_attrs:
        for k, v in dataset.attrs.items():
            hdmedian_out.attrs[k] = v

    return hdmedian_out


def _reduce_across_variables(inarray, method, axis=1, out_dtype='float32', **kwargs):
    """
    Apply cross variable reduction across time for each x/y coordinate in a 4-D nd-array

    Helper function used when computing medoids of datasets.

    :param np.ndarray inarray: is expected to have dimensions of (bands, time, y, x)
    """
    if len(inarray.shape) != 4:
        raise ValueError("Can only operate on 4-D arrays")
    if axis != 1:
        raise ValueError("Reduction axis must be 1. Expected axes are (bands, time, y, x)")

    variable, time, y, x = inarray.shape
    output = np.empty((variable, y, x), dtype=out_dtype)
    for iy in range(y):
        for ix in range(x):
            try:
                output[:, iy, ix] = method(inarray[:, :, iy, ix])
            except ValueError:
                output[:, iy, ix] = np.nan
    return output


def prod(a):
    """Product of a sequence"""
    return reduce_(mul_op, a, 1)


def _blah(shape, step=1, dtype=None):
    return np.arange(0, prod(shape) * step, step, dtype=dtype).reshape(shape)


def axisindex(a, index, axis=0):
    """
    Index array 'a' using 'index' as depth along 'axis'
    """
    shape = index.shape
    lshape = shape[:axis] + (1,) * (index.ndim - axis)
    rshape = (1,) * axis + shape[axis:]
    step = prod(shape[axis:])
    idx = _blah(lshape, step * a.shape[axis]) + _blah(rshape) + index * step
    return a.take(idx)


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


class Statistic(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute(self, data):
        """
        Compute a statistic on the given Dataset.

        :param xarray.Dataset data:
        :return: xarray.Dataset
        """
        return

    def measurements(self, input_measurements):
        """
        Turn a list of input measurements into a list of output measurements.

        Base implementation simply copies input measurements to output_measurements.

        :rtype: list(dict)
        """
        return [
            {attr: measurement[attr] for attr in ['name', 'dtype', 'nodata', 'units']}
            for measurement in input_measurements]


class ClearCount(Statistic):
    """Count the number of clear data points through time"""

    def compute(self, data):
        # TODO Fix Hardcoded 'time' and pulling out first data var
        _, sample_data_var = next(data.data_vars.items())
        count_values = sample_data_var.count(dim='time').rename('clear_observations')
        return count_values

    def measurements(self, input_measurements):
        return [
            {
                'name': 'count_observations',
                'dtype': 'int16',
                'nodata': -1,
                'units': '1'
            }
        ]


class NoneStat(Statistic):
    def compute(self, data):
        class Empty:
            data_vars = {}

        return Empty()

    def measurements(self, input_measurements):
        return input_measurements


class SimpleStatistic(Statistic):
    """
    Describes the outputs of a statistic and how to calculate it

    :param stat_func:
        callable to compute statistics. Should both accept and return a :class:`xarray.Dataset`.
    """

    def __init__(self, stat_func):
        self.stat_func = stat_func

    def compute(self, data):
        return self.stat_func(data)


class SimpleXarrayReduction(Statistic):
    """
    Compute statistics using a reduction function defined on :class:`xarray.Dataset`.
    """

    def __init__(self, xarray_function_name):
        """
        :param str xarray_function_name: name of an :class:`xarray.Dataset` reduction function
        """
        self._stat_func_name = xarray_function_name

    def compute(self, data):
        func = getattr(xarray.Dataset, self._stat_func_name)
        return func(data, dim='time')


class WofsStats(Statistic):
    """
    Example stats calculator for Wofs

    It's very hard coded, but maybe that's a good thing.
    """
    def compute(self, data):
        wet = (data.water == 128).sum(dim='time')
        dry = (data.water == 0).sum(dim='time')
        clear = wet + dry
        frequency = wet / clear
        return xarray.Dataset({'count_wet': wet,
                               'count_clear': clear,
                               'frequency': frequency}, attrs=dict(crs=data.crs))

    def measurements(self, input_measurements):
        measurement_names = set(m['name'] for m in input_measurements)
        assert 'water' in measurement_names
        return [
            {
                'name': 'count_wet',
                'dtype': 'int16',
                'nodata': -1,
                'units': '1'
            },
            {
                'name': 'count_clear',
                'dtype': 'int16',
                'nodata': -1,
                'units': '1'
            },
            {
                'name': 'frequency',
                'dtype': 'float32',
                'nodata': -1,
                'units': '1'
            },

        ]


class NormalisedDifferenceStats(Statistic):
    """
    Simple NDVI/NDWI and other Normalised Difference stats

    Computes (band1 - band2)/(band1 + band2), and then summarises using the list of `stats` into
    separate output variables.
    """

    def __init__(self, band1, band2, name, stats=None):
        self.stats = stats if stats else ['min', 'max', 'mean']
        self.band1 = band1
        self.band2 = band2
        self.name = name

    def compute(self, data):
        nd = (data[self.band1] - data[self.band2]) / (data[self.band1] + data[self.band2])
        outputs = {}
        for stat in self.stats:
            name = '_'.join([self.name, stat])
            outputs[name] = getattr(nd, stat)(dim='time')
        return xarray.Dataset(outputs,
                              attrs=dict(crs=data.crs))

    def measurements(self, input_measurements):
        measurement_names = [m['name'] for m in input_measurements]
        if self.band1 not in measurement_names or self.band2 not in measurement_names:
            raise StatsConfigurationError('Input measurements for %s must include "%s" and "%s"',
                                          self.name, self.band1, self.band2)

        return [dict(name='_'.join([self.name, stat]), dtype='float32', nodata=-1, units='1')
                for stat in self.stats]


class IndexStat(SimpleStatistic):
    def __init__(self, stat_func):
        super(IndexStat, self).__init__(stat_func)

    def compute(self, data):
        index = super(IndexStat, self).compute(data)

        def index_dataset(var):
            return axisindex(data.data_vars[var.name].values, var.values)

        data_values = index.apply(index_dataset)
        return data_values


class PerBandIndexStat(SimpleStatistic):
    """
    Each output variable contains values that actually exist in the input data.

    It uses a function that returns the indexes of these values, then pulls them out of the source data,
    along with provenance information.

    :param stat_func: A function which takes an xarray.Dataset and returns an xarray.Dataset of indexes
    """

    def __init__(self, stat_func):
        super(PerBandIndexStat, self).__init__(stat_func)

    def compute(self, data):
        index = super(PerBandIndexStat, self).compute(data)

        def index_dataset(var):
            return axisindex(data.data_vars[var.name].values, var.values)

        data_values = index.apply(index_dataset)

        def index_time(var):
            return data.time.values[var.values]

        time_values = index.apply(
            index_time).rename(
            OrderedDict((name, name + '_observed')
                        for name in index.data_vars))

        text_values = time_values.apply(_datetime64_to_inttime).rename(
            OrderedDict((name, name + '_date')
                        for name in time_values.data_vars))

        def index_source(var):
            return data.source.values[var.values]

        time_values = index.apply(index_source).rename(OrderedDict((name, name + '_source')
                                                                   for name in index.data_vars))

        return xarray.merge([data_values, time_values, text_values, count_values])

    def measurements(self, input_measurements):
        index_measurements = [
            {
                'name': measurement['name'] + '_source',
                'dtype': 'int8',
                'nodata': -1,
                'units': '1'
            }
            for measurement in input_measurements
            ]
        date_measurements = [
            {
                'name': measurement['name'] + '_observed',
                'dtype': 'float64',
                'nodata': 0,
                'units': 'seconds since 1970-01-01 00:00:00'
            }
            for measurement in input_measurements
            ]
        text_measurements = [
            {
                'name': measurement['name'] + '_observed_date',
                'dtype': 'int32',
                'nodata': 0,
                'units': 'Date as YYYYMMDD'
            }
            for measurement in input_measurements
            ]

        return (super(PerBandIndexStat, self).measurements(input_measurements) + date_measurements +
                index_measurements + text_measurements)


class PerPixelMetadata(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, var_name='observed'):
        self._var_name = var_name

    @abc.abstractmethod
    def compute(self, data, selected_indexes):
        """Return a variable name and :class:`xarray.Variable` to add in to the """
        return

    @abc.abstractmethod
    def measurement(self):
        return


class ObservedDaysSince(PerPixelMetadata):
    def __init__(self, since=datetime(1970, 1, 1), var_name='observed'):
        super(ObservedDaysSince, self).__init__(var_name)
        self._since = since

    def compute(self, data, selected_indexes):
        observed = data.time.values[selected_indexes] - np.datetime64(self._since)
        days_since = observed.astype('timedelta64[D]').astype('int16')

        return self._var_name, xarray.Variable(('y', 'x'), days_since)

    def measurement(self):
        return {
            'name': self._var_name,
            'dtype': 'int16',
            'nodata': 0,
            'units': 'days since {:%Y-%m-%d %H:%M:%S}'.format(self._since)
        }


class ObservedDateInt(PerPixelMetadata):
    def compute(self, data, selected_indexes):
        observed = data.time.values[selected_indexes]
        observed_date = xarray.Variable(('y', 'x'), _datetime64_to_inttime(observed))
        return self._var_name, observed_date

    def measurement(self):
        return {
            'name': self._var_name,
            'dtype': 'int32',
            'nodata': 0,
            'units': 'Date as YYYYMMDD'
        }


class SourceIndex(PerPixelMetadata):
    def compute(self, data, selected_indexes):
        return self._var_name, xarray.Variable(('y', 'x'), data.source.values[selected_indexes])

    def measurement(self):
        return {
            'name': self._var_name,
            'dtype': 'int8',
            'nodata': -1,
            'units': '1'
        }


class PerStatIndexStat(SimpleStatistic):
    """
    :param stat_func: A function which takes an xarray.Dataset and returns an xarray.Dataset of indexes
    :param list[PerPixelMetadata] extra_metadata_producers: collection of metadata generators
    """

    def __init__(self, stat_func, extra_metadata_producers=None):
        super(PerStatIndexStat, self).__init__(stat_func)
        self._metadata_producers = extra_metadata_producers or []

    def compute(self, data):
        index = super(PerStatIndexStat, self).compute(data)

        def index_dataset(var, axis):
            return axisindex(var, index, axis=axis)

        data_values = data.reduce(index_dataset, dim='time')

        for metadata_producer in self._metadata_producers:
            var_name, var_data = metadata_producer.compute(data, index)
            data_values[var_data] = var_data

        return data_values

    def measurements(self, input_measurements):
        metadata_variables = [metadata_producer.measurement() for metadata_producer in self._metadata_producers]
        return super(PerStatIndexStat, self).measurements(input_measurements) + metadata_variables


def _compute_medoid(data, index_dtype='int16'):
    flattened = data.to_array(dim='variable')
    variable, time, y, x = flattened.shape
    index = np.empty((y, x), dtype=index_dtype)
    for iy in range(y):
        for ix in range(x):
            index[iy, ix] = argnanmedoid(flattened.values[:, :, iy, ix])
    return index


def percentile_stat(q):
    return PerBandIndexStat(  # pylint: disable=redundant-keyword-arg
        stat_func=partial(getattr(xarray.Dataset, 'reduce'),
                          dim='time',
                          func=argpercentile,
                          q=q))


def percentile_stat_no_prov(q):
    return IndexStat(  # pylint: disable=redundant-keyword-arg
        stat_func=partial(getattr(xarray.Dataset, 'reduce'),
                          dim='time',
                          func=argpercentile,
                          q=q))


def _datetime64_to_inttime(var):
    """
    Return an "inttime" representing a datetime64.

    For example, 2016-09-29 as an "inttime" would be 20160929

    :param var: ndarray of datetime64
    :return: ndarray of ints, representing the given time to the nearest day
    """
    values = getattr(var, 'values', var)
    years = values.astype('datetime64[Y]').astype('int32') + 1970
    months = values.astype('datetime64[M]').astype('int32') % 12 + 1
    days = (values.astype('datetime64[D]') - values.astype('datetime64[M]') + 1).astype('int32')
    return years * 10000 + months * 100 + days


STATS = {
    'min': SimpleXarrayReduction('min'),
    'max': SimpleXarrayReduction('max'),
    'mean': SimpleXarrayReduction('mean'),
    'percentile_10': percentile_stat(10),
    'percentile_25': percentile_stat(25),
    'percentile_50': percentile_stat(50),
    'percentile_75': percentile_stat(75),
    'percentile_90': percentile_stat(90),
    'percentile_10_no_prov': percentile_stat_no_prov(10),
    'percentile_25_no_prov': percentile_stat_no_prov(25),
    'percentile_50_no_prov': percentile_stat_no_prov(50),
    'percentile_75_no_prov': percentile_stat_no_prov(75),
    'percentile_90_no_prov': percentile_stat_no_prov(90),
    'medoid': PerStatIndexStat(stat_func=_compute_medoid, extra_metadata_producers=[ObservedDaysSince()]),
    'medoid_no_prov': PerStatIndexStat(stat_func=_compute_medoid),
    'ndvi_stats': NormalisedDifferenceStats(name='ndvi', band1='nir', band2='red',
                                            stats=['min', 'mean', 'max']),
    'ndwi_stats': NormalisedDifferenceStats(name='ndwi', band1='green', band2='swir1',
                                            stats=['min', 'mean', 'max']),
    'ndvi_daily': NormalisedDifferenceStats(name='ndvi', band1='nir', band2='red', stats=['squeeze']),
    'ndwi_daily': NormalisedDifferenceStats(name='ndvi', band1='nir', band2='red', stats=['squeeze']),
    'none': NoneStat(),
    'wofs_summary': WofsStats(),
    'clear_count': ClearCount()
}

# Dynamically look for and load statistics from other packages

for entry_point in iter_entry_points(group='datacube.stats', name=None):
    STATS[entry_point.name] = entry_point.load()

try:
    from hdmedians import nangeomedian

    def apply_geomedian(inarray, f, axis=3, eps=1e-3, **kwargs):
        assert len(inarray.shape) == 4
        assert axis == 3

        xs, ys, bands, times = inarray.shape
        output = np.ndarray((xs, ys, bands), dtype=inarray.dtype)
        for ix in range(xs):
            for iy in range(ys):
                try:
                    output[ix, iy, :] = f(inarray[ix, iy, :, :], eps=eps, axis=1)
                except ValueError:
                    output[ix, iy, :] = np.nan
        return output

    class GeoMedian(Statistic):
        def __init__(self, eps=1e-3):
            super(GeoMedian, self).__init__()
            self.eps = eps

        def compute(self, data):
            """
            :param xarray.Dataset data:
            :return: xarray.Dataset
            """
            # Assert data shape/dims
            inarray = data.to_array(dim='variable').transpose('x', 'y', 'variable', 'time').copy()

            output = inarray.reduce(apply_geomedian, dim='time', keep_attrs=True, f=nangeomedian, eps=self.eps)

            return output.transpose('variable', 'y', 'x').to_dataset(dim='variable')


    STATS['geomedian'] = GeoMedian()
except ImportError:
    pass
