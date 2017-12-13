"""
Classes for performing statistical data analysis.
"""
from __future__ import absolute_import

import abc
from collections import OrderedDict
from copy import copy
from datetime import datetime
from functools import partial

import numpy as np
import xarray
from pkg_resources import iter_entry_points

from datacube.storage.masking import create_mask_value
from datacube_stats.utils.dates import datetime64_to_inttime
from .incremental_stats import (mk_incremental_sum, mk_incremental_or,
                                compose_proc, broadcast_proc)
from .utils import da_nodata, mk_masker, first_var
from .stat_funcs import axisindex, argpercentile, _compute_medoid
from .stat_funcs import anynan, section_by_index, medoid_indices


class StatsConfigurationError(RuntimeError):
    pass


class StatsProcessingError(RuntimeError):
    pass


class Statistic(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute(self, data):
        """
        Compute a statistic on the given Dataset.

        # FIXME: Explain a little bit better, Dataset in, Dataset out, measurements match measurements()

        :param xarray.Dataset data:
        :return: xarray.Dataset
        """

    def measurements(self, input_measurements):
        """
        Turn a list of input measurements into a list of output measurements.

        Base implementation simply copies input measurements to output_measurements.

        # FIXME: Explain the purpose of this

        :rtype: list(dict)
        """
        output_measurements = [
            {attr: measurement[attr] for attr in ['name', 'dtype', 'nodata', 'units']}
            for measurement in input_measurements]
        return output_measurements

    def is_iterative(self):
        """
        Should return True if class supports iterative computation one time slice at a time.

        :rtype: Bool
        """
        return False

    def make_iterative_proc(self):
        """
        Should return `None` if `is_iterative()` returns `False`.

        Should return processing function `proc` that closes over internal
        state that get updated one time slice at time, if `is_iterative()`
        returns `True`.

        proc(dataset_slice)  # Update internal state, called many times
        result = proc()  # Extract final result, called once


        See `incremental_stats.assemble_updater`

        """
        return None


class NoneStat(Statistic):
    def compute(self, data):
        return data

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
        with np.errstate(divide='ignore', invalid='ignore'):
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
    Simple NDVI/NDWI and other Normalised Difference statistics.

    Computes `(band1 - band2)/(band1 + band2)`, and then summarises using the list of `stats` into
    separate output variables.

    Output variables are named {name}_{stat_name}. Eg: `ndwi_median`

    By default will clamp output values in the range [-1, 1] by setting values outside
    this range to NaN.

    :param name: The common name of a normalised difference.
                 eg. `ndvi` for `(nir-red)/(nir+red)`
                     `ndwi` for `(green-nir)/(green+nir)`
    :param List[str] stats: list of common statistic names. Defaults to ['min', 'max', 'mean']
                            Choose from the common xarray/numpy reduction operations
                            which include `std` and `median`.
    """

    def __init__(self, band1, band2, name, stats=None, clamp_outputs=True):
        self.stats = stats if stats else ['min', 'max', 'mean']
        self.band1 = band1
        self.band2 = band2
        self.name = name
        self.clamp_outputs = clamp_outputs

    def compute(self, data):
        nd = (data[self.band1] - data[self.band2]) / (data[self.band1] + data[self.band2])
        outputs = {}
        for stat in self.stats:
            name = '_'.join([self.name, stat])
            outputs[name] = getattr(nd, stat)(dim='time', keep_attrs=True)
            if self.clamp_outputs:
                self._clamp_outputs(outputs[name])
        return xarray.Dataset(outputs, attrs=dict(crs=data.crs))

    @staticmethod
    def _clamp_outputs(dataarray):
        with warnings.catch_warnings():  # Don't print error while comparing nan
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dataarray.values[dataarray.values < -1] = np.nan
            dataarray.values[dataarray.values > 1] = np.nan

    def measurements(self, input_measurements):
        measurement_names = [m['name'] for m in input_measurements]
        if self.band1 not in measurement_names or self.band2 not in measurement_names:
            raise StatsConfigurationError('Input measurements for %s must include "%s" and "%s"',
                                          self.name, self.band1, self.band2)

        return [dict(name='_'.join([self.name, stat]), dtype='float32', nodata=-1, units='1')
                for stat in self.stats]


class IndexStat(SimpleStatistic):
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

        text_values = time_values.apply(datetime64_to_inttime).rename(
            OrderedDict((name, name + '_date')
                        for name in time_values.data_vars))

        def index_source(var):
            return data.source.values[var.values]

        source_values = index.apply(index_source).rename(OrderedDict((name, name + '_source')
                                                                     for name in index.data_vars))

        return xarray.merge([data_values, time_values, text_values, source_values])

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


class Percentile(PerBandIndexStat):
    def __init__(self, q):
        if isinstance(q, list):
            self.qs = q
        else:
            self.qs = [q]

        super(Percentile, self).__init__(stat_func=None)

    def compute(self, data):
        def single(q):
            stat_func = partial(xarray.Dataset.reduce, dim='time',
                                func=argpercentile, q=q)

            renamed = data.rename({var: var + '_PC_' + str(q)
                                   for var in data.data_vars})

            return PerBandIndexStat(stat_func=stat_func).compute(renamed)

        return xarray.merge(single(q) for q in self.qs)

    def measurements(self, input_measurements):
        inputs = Statistic.measurements(self, input_measurements)

        renamed = [{**m, 'name': m['name'] + '_PC_' + str(q)}
                   for q in self.qs
                   for m in inputs]

        return PerBandIndexStat.measurements(self, renamed)


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
        observed_date = xarray.Variable(('y', 'x'), datetime64_to_inttime(observed))
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
                result = section_by_index(var_array, index, axis)
                result[not_enough] = nodata
                return result

            return var.reduce(worker, dim='time', nodata=da_nodata(var))

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

    def is_iterative(self):
        return self._impl.is_iterative()

    def make_iterative_proc(self):
        return self._impl.make_iterative_proc()


class MaskMultiCounter(Statistic):
    # pylint: disable=redefined-builtin
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
        self._vars = [v.copy() for v in vars]
        self._nodata_flags = nodata_flags
        self._valid_pq_mask = None

    def measurements(self, input_measurements):
        nodata = -1
        bit_defs = input_measurements[0]['flags_definition']

        if self._nodata_flags is not None:
            self._valid_pq_mask = mk_masker(*create_mask_value(bit_defs, **self._nodata_flags), invert=True)

        for v in self._vars:
            flags = v['flags']
            v['_mask'] = create_mask_value(bit_defs, **flags)
            v['mask'] = mk_masker(*v['_mask'])

        return [dict(name=v['name'],
                     dtype='int16',
                     units='1',
                     nodata=nodata) for v in self._vars]

    def is_iterative(self):
        return True

    def make_iterative_proc(self):
        def _to_mask(ds):
            da = first_var(ds)
            return xarray.Dataset({v['name']: v['mask'](da) for v in self._vars},
                                  attrs=ds.attrs)

        # PQ -> BoolMasks: DataSet<Bool> -> Sum: DataSet<int16>
        proc = compose_proc(_to_mask,
                            proc=mk_incremental_sum(dtype='int16'))

        if self._valid_pq_mask is None:
            return proc

        def invalid_data_mask(da):
            mm = da.values
            if mm.all():  # All pixels had at least one valid observation
                return None
            return np.logical_not(mm, out=mm)

        # PQ -> ValidMask:DataArray<Bool> -> OR:DataArray<Bool> |> InvertMask:ndarray<Bool>|None
        valid_proc = compose_proc(lambda ds: self._valid_pq_mask(first_var(ds)),
                                  proc=mk_incremental_or(),
                                  output_transform=invalid_data_mask)

        _vars = {v['name']: v for v in self._vars}

        def apply_mask(ds, mask, nodata=-1):
            if mask is None:
                return ds

            for name, da in ds.data_vars.items():
                simple = _vars[name].get('simple', False)
                if not simple:
                    da.values[mask] = nodata

            return ds

        # Counts      ----\
        #                  +----------- Counts[ InvalidMask ] = nodata
        # InvalidMask ----/

        return broadcast_proc(proc, valid_proc, combine=apply_mask)

    def compute(self, data):
        proc = self.make_iterative_proc()

        for i in range(data.time.shape[0]):
            proc(data.isel(time=slice(i, i + 1)))

        return proc()

    def __repr__(self):
        return 'MaskMultiCounter<{}>'.format(','.join([v['name'] for v in self._vars]))


STATS = {
    'simple': ReducingXarrayStatistic,
    'percentile': Percentile,
    'percentile_no_prov': PercentileNoProv,
    'medoid': Medoid,
    'medoid_no_prov': MedoidNoProv,
    'medoid_simple': MedoidSimple,
    'simple_normalised_difference': NormalisedDifferenceStats,
    # 'ndvi_stats': NormalisedDifferenceStats(name='ndvi', band1='nir', band2='red', stats=['min', 'mean', 'max']),
    # 'ndwi_stats': NormalisedDifferenceStats(name='ndwi', band1='green', band2='swir1', stats=['min', 'mean', 'max']),
    # 'ndvi_daily': NormalisedDifferenceStats(name='ndvi', band1='nir', band2='red', stats=['squeeze']),
    'none': NoneStat,
    'wofs_summary': WofsStats,
    'masked_multi_count': MaskMultiCounter,
    'external': ExternalPlugin,
}


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
        def __init__(self, eps=1e-3, maxiters=None):
            super(GeoMedian, self).__init__()
            self.eps = eps
            self.maxiters = maxiters

        def compute(self, data):
            from_, to = self._vars_to_transpose(data)
            # Assert data shape/dims
            data = data.to_array(dim='variable').transpose(*from_).copy()

            data = data.reduce(apply_geomedian, dim='time', keep_attrs=True, f=nangeomedian,
                               eps=self.eps, maxiters=self.maxiters)

            return data.transpose(*to).to_dataset(dim='variable')

        def measurements(self, input_measurements):
            """
            Outputs will have the same name as inputs, but dtype will always be float32.
            """
            output_measurements = [
                {attr: measurement[attr] for attr in ['name', 'dtype', 'nodata', 'units']}
                for measurement in input_measurements]
            for measurement in output_measurements:
                measurement['dtype'] = 'float32'
                measurement['nodata'] = np.nan

            return output_measurements

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
                raise StatsProcessingError(
                    'Data to process contains both geographic and projected dimensions, unable to proceed')
            elif not is_proj and not is_geo:
                raise StatsProcessingError(
                    'Data to process contains neither geographic nor projected dimensions, unable to proceed')
            elif is_proj:
                return ('x', 'y', 'variable', 'time'), ('variable', 'y', 'x')
            else:
                return ('longitude', 'latitude', 'variable', 'time'), ('variable', 'latitude', 'longitude')

    STATS['geomedian'] = GeoMedian

except ImportError:
    pass

try:
    from pcm import gmpcm

    class NewGeomedianStatistic(Statistic):
        def __init__(self, eps=1e-3):
            super(NewGeomedianStatistic, self).__init__()
            self.eps = eps

        def compute(self, data):
            """
            :param xarray.Dataset data:
            :return: xarray.Dataset
            """
            # We need to reshape our data into Y, X, Band, Time

            squashed_together_dimensions, normal_datacube_dimensions = self._vars_to_transpose(data)

            squashed = data.to_array(dim='variable').transpose(*squashed_together_dimensions)
            assert squashed.dims == squashed_together_dimensions

            # Grab a copy of the coordinates we need for creating the output DataArray
            output_coords = copy(squashed.coords)
            if 'time' in output_coords:
                del output_coords['time']
            if 'source' in output_coords:
                del output_coords['source']

            # Call Dale's function here
            squashed = gmpcm(squashed.data)

            # Jam the raw numpy array back into a pleasantly labelled DataArray
            output_dims = squashed_together_dimensions[:-1]
            as_datarray = xarray.DataArray(squashed, dims=output_dims, coords=output_coords)

            return as_datarray.transpose(*normal_datacube_dimensions).to_dataset(dim='variable')

        @staticmethod
        def _vars_to_transpose(data):
            """
            We need to be able to handle data given to use in either Geographic or Projected form.

            The Data Cube provided xarrays will contain different dimensions, latitude/longitude or x/y, which means
            the array reshaping takes different arguments.

            The dimension ordering returned by this function is specific to the Geometric Median PCM functions
            included from the `pcm` module.

            :return: pcm input array dimension order, datacube dimension ordering
            """
            is_projected = 'x' in data.dims and 'y' in data.dims
            is_geographic = 'longitude' in data.dims and 'latitude' in data.dims

            if is_projected and is_geographic:
                raise StatsProcessingError('Data to process contains BOTH geographic and projected dimensions, '
                                           'unable to proceed')
            elif not is_projected and not is_geographic:
                raise StatsProcessingError('Data to process contains NEITHER geographic nor projected dimensions, '
                                           'unable to proceed')
            elif is_projected:
                return ('y', 'x', 'variable', 'time'), ('variable', 'y', 'x')
            else:
                return ('latitude', 'longitude', 'variable', 'time'), ('variable', 'latitude', 'longitude')

    STATS['new_geomedian'] = NewGeomedianStatistic

except ImportError:
    pass


try:
    import pcm

    class SpectralMAD(Statistic):
        def __init__(self, eps=1e-3):
            super(SpectralMAD, self).__init__()
            self.eps = eps

        def compute(self, data):
            """
            :param xarray.Dataset data:
            :return: xarray.Dataset
            """

            # We need to reshape our data into Y, X, Band, Time
            squashed_together_dimensions, output_dimensions = self._vars_to_transpose(data)

            squashed = data.to_array(dim='variable').transpose(*squashed_together_dimensions)
            assert squashed.dims == squashed_together_dimensions

            # Grab a copy of the coordinates we need for creating the output DataArray
            output_coords = copy(squashed.coords)
            if 'variable' in output_coords:
                del output_coords['variable']
            if 'time' in output_coords:
                del output_coords['time']
            if 'source' in output_coords:
                del output_coords['source']

            # Call Dale's geometric median & spectral mad functions here
            gm = pcm.gmpcm(squashed.data)
            squashed = pcm.smad(squashed.data, gm)

            # Jam the raw numpy array back into a pleasantly labelled DataArray
            as_datarray = xarray.DataArray(squashed, dims=output_dimensions, coords=output_coords)

            return as_datarray.transpose(*output_dimensions).to_dataset(name='smad')

        def measurements(self, input_measurements):
            return [dict(name='smad', dtype='float32', nodata=np.nan, units='1')]

        @staticmethod
        def _vars_to_transpose(data):
            """
            We need to be able to handle data given to use in either Geographic or Projected form.

            The Data Cube provided xarrays will contain different dimensions, latitude/longitude or x/y, which means
            the array reshaping takes different arguments.

            The dimension ordering returned by this function is specific to the spectral median absolute deviation
            function included from the `pcm` module.

            :return: pcm input array dimension order, datacube dimension ordering
            """
            is_projected = 'x' in data.dims and 'y' in data.dims
            is_geographic = 'longitude' in data.dims and 'latitude' in data.dims

            if is_projected and is_geographic:
                raise StatsProcessingError('Data to process contains BOTH geographic and projected dimensions, '
                                           'unable to proceed')
            elif not is_projected and not is_geographic:
                raise StatsProcessingError('Data to process contains NEITHER geographic nor projected dimensions, '
                                           'unable to proceed')
            elif is_projected:
                return ('y', 'x', 'variable', 'time'), ('y', 'x')
            else:
                return ('latitude', 'longitude', 'variable', 'time'), ('variable', 'latitude', 'longitude')

    STATS['spectral_mad'] = SpectralMAD
except ImportError:
    pass
