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

from datacube.storage.masking import make_mask, create_mask_value

try:
    from bottleneck import anynan, nansum
except ImportError:
    nansum = np.nansum

    def anynan(x, axis=None):
        return np.isnan(x).any(axis=axis)


class StatsConfigurationError(RuntimeError):
    pass


class StatsProcessingError(RuntimeError):
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


def section_by_index(array, axis, index):
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


class Statistic(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute(self, data):
        """
        Compute a statistic on the given Dataset.

        :param xarray.Dataset data:
        :return: xarray.Dataset
        """
        return data

    def measurements(self, input_measurements):
        """
        Turn a list of input measurements into a list of output measurements.

        Base implementation simply copies input measurements to output_measurements.

        :rtype: list(dict)
        """
        output_measurements = [
            {attr: measurement[attr] for attr in ['name', 'dtype', 'nodata', 'units']}
            for measurement in input_measurements]
        return output_measurements


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


class ReducingXarrayStatistic(Statistic):
    """
    Compute statistics using a reduction function defined on :class:`xarray.Dataset`.
    """

    def __init__(self, reduction_function):
        """
        :param str reduction_function: name of an :class:`xarray.Dataset` reduction function
        """
        # TODO: Validate that reduction function exists
        self._stat_func_name = reduction_function

    def compute(self, data):
        func = getattr(xarray.Dataset, self._stat_func_name)
        return func(data, dim='time')


class WofsStats(Statistic):
    """
    Example stats calculator for Wofs

    It's very hard coded, but maybe that's a good thing.
    """

    def __init__(self, freq_only=False):
        self.freq_only = freq_only

    def compute(self, data):
        is_integer_type = np.issubdtype(data.water.dtype, np.integer)

        if not is_integer_type:
            raise StatsProcessingError("Attempting to count bit flags on non-integer data. Provided data is: {}"
                                       .format(data.water))

        # 128 == clear and wet, 132 == clear and wet and masked for sea
        # The PQ sea mask that we use is dodgy and should be ignored. It excludes lots of useful data
        wet = ((data.water == 128) | (data.water == 132)).sum(dim='time')
        dry = ((data.water == 0) | (data.water == 4)).sum(dim='time')
        clear = wet + dry
        frequency = wet / clear
        if self.freq_only:
            return xarray.Dataset({'frequency': frequency}, attrs=dict(crs=data.crs))
        else:
            return xarray.Dataset({'count_wet': wet,
                                   'count_clear': clear,
                                   'frequency': frequency}, attrs=dict(crs=data.crs))

    def measurements(self, input_measurements):
        measurement_names = set(m['name'] for m in input_measurements)
        assert 'water' in measurement_names

        wet = {'name': 'count_wet',
               'dtype': 'int16',
               'nodata': -1,
               'units': '1'}
        dry = {'name': 'count_clear',
               'dtype': 'int16',
               'nodata': -1,
               'units': '1'}
        frequency = {'name': 'frequency',
                     'dtype': 'float32',
                     'nodata': -1,
                     'units': '1'}
        if self.freq_only:
            return [frequency]
        else:
            return [wet, dry, frequency]


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
            data_values[var_name] = var_data

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


class PercentileNoProv(Statistic):
    def __init__(self, q):
        self.q = q

    def compute(self, data):
        index = data.reduce(dim='time', func=argpercentile, q=self.q)

        def index_dataset(var):
            return axisindex(data.data_vars[var.name].values, var.values)

        data_values = index.apply(index_dataset)
        return data_values


class MedoidSimple(PerStatIndexStat):
    def __init__(self):
        super(MedoidSimple, self).__init__(stat_func=_compute_medoid,
                                           extra_metadata_producers=[ObservedDaysSince()])


class MedoidNoProv(PerStatIndexStat):
    def __init__(self):
        super(MedoidNoProv, self).__init__(stat_func=_compute_medoid)


def select_names(wanted_names, all_names):
    """ Only select the measurements names in the wanted list. """
    if wanted_names is None:
        # default: include everything
        return all_names

    invalid = [name
               for name in wanted_names
               if name not in all_names]

    if invalid:
        msg = 'Specified measurements not found: {}'
        raise StatsConfigurationError(msg.format(invalid))

    return wanted_names


class Medoid(Statistic):
    """
    Medoid (a multi-dimensional generalization of median) of a set of
    observations through time.

    :arg minimum_valid_observations: if not enough observations are available,
                                     medoid will return `nodata` (default 0)
    :arg input_measurements: list of measurements that contribute to medoid
                             calculation
    :arg output_measurements: list of reported measurements
    :arg metadata_producers: list of additional metadata producers
    """
    def __init__(self,
                 minimum_valid_observations=0,
                 input_measurements=None,
                 output_measurements=None,
                 metadata_producers=None):

        self.minimum_valid_observations = minimum_valid_observations
        self.input_measurements = input_measurements
        self.output_measurements = output_measurements

        # attach observation time (in days) if no other metadata requested
        if metadata_producers is None:
            self._metadata_producers = [ObservedDaysSince()]
        else:
            self._metadata_producers = metadata_producers

    def measurements(self, input_measurements):
        base = super(Medoid, self).measurements(input_measurements)

        selected_names = select_names(self.output_measurements,
                                      [m['name'] for m in base])

        selected = [m for m in base if m['name'] in selected_names]

        extra = [producer.measurement()
                 for producer in self._metadata_producers]

        return selected + extra

    def compute(self, data):
        # calculate medoid using only the fields in `input_measurements`
        input_data = data[select_names(self.input_measurements,
                                       list(data.data_vars))]

        # calculate medoid indices
        arr = input_data.to_array().values
        invalid = anynan(arr, axis=0)
        index = medoid_indices(arr, invalid)

        # pixels for which there is not enough data
        count_valid = np.count_nonzero(~invalid, axis=0)
        not_enough = count_valid < self.minimum_valid_observations

        # only report the measurements requested
        output_data = data[select_names(self.output_measurements,
                                        list(data.data_vars))]

        def reduction(var):
            """ Extracts data at `index` for a `var` of type `DataArray`. """

            def worker(var_array, axis, nodata):
                # operates on the underlying `ndarray`
                result = section_by_index(var_array, axis, index)
                result[not_enough] = nodata
                return result

            return var.reduce(worker, dim='time', nodata=var.nodata)

        def attach_metadata(result):
            """ Attach additional metadata to the `result`. """
            # used to attach time stamp on the medoid observations
            for metadata_producer in self._metadata_producers:
                var_name, var_data = metadata_producer.compute(data, index)
                nodata = metadata_producer.measurement()['nodata']
                var_data.data[not_enough] = nodata
                result[var_name] = var_data

            return result

        return attach_metadata(output_data.apply(reduction,
                                                 keep_attrs=True))

    def __repr__(self):
        if self.minimum_valid_observations == 0:
            msg = 'Medoid'
        else:
            msg = 'Medoid<minimum_valid_observations={}>'
        return msg.format(self.minimum_valid_observations)


class FlagCounter(Statistic):
    """
    Count number of flagged pixels

    Requires:
    - The name of a `measurement` to base the count upon
    - A list of `flags` that must be set in the measurement
    """

    def __init__(self, measurement, flags):
        self.measurement = measurement
        self.flags = flags

    def compute(self, data):
        datavar = data[self.measurement]

        is_integer_type = np.issubdtype(datavar.dtype, np.integer)

        if not is_integer_type:
            raise StatsProcessingError("Attempting to count bit flags on non-integer data. Provided data is: {}"
                                       .format(datavar))

        mask = make_mask(datavar, **self.flags)
        count = mask.sum(dim='time')
        return count.to_dataset().rename({self.measurement: 'count'})

    def measurements(self, input_measurements):
        measurement_names = set(m['name'] for m in input_measurements)
        assert self.measurement in measurement_names

        return [{'name': 'count',
                 'dtype': 'int16',
                 'nodata': -1,
                 'units': '1'}]


class MaskedCount(Statistic):
    """
    Use the provided flags to count the number of True values through time.

    """

    def __init__(self, flags):
        self.flags = flags

    def compute(self, data):
        count = make_mask(data, **self.flags).sum(dim='time')
        return xarray.Dataset({'count': count},
                              attrs=dict(crs=data.crs))

    def measurements(self, input_measurements):
        return [{'name': 'count',
                 'dtype': 'int16',
                 'units': '1',
                 'nodata': 65536}]  # No Data is required somewhere, but doesn't really make sense


class ExternalPlugin(Statistic):
    """
    Run externally defined plugin.

    """
    def __init__(self, impl, *args, **kwargs):
        from pydoc import locate  # TODO: probably should use importlib, but this works so easily

        impl_class = locate(impl)

        if impl_class is None:
            raise StatsProcessingError("Failed to load external plugin: '{}'".format(impl))

        self._impl = impl_class(*args, **kwargs)

    def compute(self, data):
        return self._impl.compute(data)

    def measurements(self, input_measurements):
        return self._impl.measurements(input_measurements)


class MaskMultiCounter(Statistic):
    def __init__(self, vars, nodata_flags=None):
        """

        vars:
           - name: <output_variable_name: String>
             simple: <optional Bool, default: False>
             flags:
               field_name1: expected_value1
               field_name2: expected_value2

        # optional, define input nodata as a mask
        # when all inputs match this, then output will be set to nodata
        # this allows to distinguish 0 from nodata

        nodata_flags:
           contiguous: False

        If variable is marked simple, then there is no distinction between 0 and nodata.
        """
        self._vars = vars
        self._nodata_flags = nodata_flags
        self._nodata_mask = None

    def measurements(self, input_measurements):
        nodata = 0xFFFF
        bit_defs = input_measurements[0]['flags_definition']

        if self._nodata_flags is not None:
            self._nodata_mask = create_mask_value(bit_defs, **self._nodata_flags)

        for v in self._vars:
            flags = v['flags']
            v['mask'] = create_mask_value(bit_defs, **flags)

        return [dict(name=v['name'],
                     dtype='uint16',
                     units='1',
                     nodata=nodata) for v in self._vars]

    def compute(self, ds):
        # print('::compute t:{} {}x{}'.format(ds.dims['time'], ds.dims['x'], ds.dims['y']))

        def build_invalid_mask(pq):
            if self._nodata_mask is None:
                return None

            m, v = self._nodata_mask
            invalid_data_mask = ((pq & m) != v).sum(dim='time') == 0

            if invalid_data_mask.values.any():  # some missing data
                return invalid_data_mask
            return None

        nodata = 0xFFFF
        pq = list(ds.data_vars.values())[0]

        invalid_data_mask = build_invalid_mask(pq)

        def process_var(var):
            name = var['name']
            m, v = var['mask']
            simple = var.get('simple', False)

            cc = ((pq & m) == v).sum(dim='time').astype('uint16')

            if not simple and invalid_data_mask is not None:
                cc.values[invalid_data_mask] = nodata

            return (name, cc)

        vars = dict(process_var(v) for v in self._vars)

        return xarray.Dataset(vars, attrs=dict(crs=ds.crs))

    def __repr__(self):
        return 'MaskMultiCounter<{}>'.format(','.join([v['name'] for v in self._vars]))


STATS = {
    'simple': ReducingXarrayStatistic,
    # 'min': SimpleXarrayReduction('min'),
    # 'max': SimpleXarrayReduction('max'),
    # 'mean': SimpleXarrayReduction('mean'),
    'percentile': percentile_stat,
    'percentile_no_prov': PercentileNoProv,
    'medoid': Medoid,
    'medoid_no_prov': MedoidNoProv,
    'medoid_simple': MedoidSimple,
    'simple_normalised_difference': NormalisedDifferenceStats,
    # 'ndvi_stats': NormalisedDifferenceStats(name='ndvi', band1='nir', band2='red',
    #                                         stats=['min', 'mean', 'max']),
    # 'ndwi_stats': NormalisedDifferenceStats(name='ndwi', band1='green', band2='swir1',
    #                                         stats=['min', 'mean', 'max']),
    # 'ndvi_daily': NormalisedDifferenceStats(name='ndvi', band1='nir', band2='red', stats=['squeeze']),
    # 'ndwi_daily': NormalisedDifferenceStats(name='ndvi', band1='nir', band2='red', stats=['squeeze']),
    'none': NoneStat,
    'wofs_summary': WofsStats,
    'clear_count': ClearCount,
    'masked_count': MaskedCount,
    'masked_multi_count': MaskMultiCounter,
    'flag_counter': FlagCounter,
    'external': ExternalPlugin,
}


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


# Dynamically look for and load statistics from other packages

for entry_point in iter_entry_points(group='datacube.stats', name=None):
    STATS[entry_point.name] = entry_point.load()

try:
    from hdmedians import nangeomedian
    import warnings

    def apply_geomedian(inarray, f, axis=3, eps=1e-3, **kwargs):
        assert len(inarray.shape) == 4
        assert axis == 3

        xs, ys, bands, times = inarray.shape
        output = np.ndarray((xs, ys, bands), dtype=inarray.dtype)
        with warnings.catch_warnings():  # Don't print error about computing mean of empty slice
            warnings.simplefilter("ignore", category=RuntimeWarning)
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
            from_, to = self._vars_to_transpose(data)
            # Assert data shape/dims
            data = data.to_array(dim='variable').transpose(*from_).copy()

            data = data.reduce(apply_geomedian, dim='time', keep_attrs=True, f=nangeomedian, eps=self.eps)

            return data.transpose(*to).to_dataset(dim='variable')

        @staticmethod
        def _vars_to_transpose(data):
            """
            We need to be able to handle data given to use in either Geographic or Projected form.

            The Data Cube provided xarrays will contain different dimensions, latitude/longitude or x/y, which means
            the array reshaping takes different arguments.
            """
            is_proj = 'x' in data and 'y' in data
            is_geo = 'longitude' in data and 'latitude' in data
            if is_proj and is_geo:
                raise StatsProcessingError('Data to process contains both geographic and projected dimensions, unable to proceed')
            elif not is_proj and not is_geo:
                raise StatsProcessingError('Data to process contains neither geographic nor projected dimensions, unable to proceed')
            elif is_proj:
                return ('x', 'y', 'variable', 'time'), ('variable', 'y', 'x')
            else:
                return ('longitude', 'latitude', 'variable', 'time'), ('variable', 'latitude', 'longitude')

    STATS['geomedian'] = GeoMedian
except ImportError:
    pass
