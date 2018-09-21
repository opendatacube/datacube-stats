import warnings

from collections import OrderedDict, Sequence
from functools import partial
from datetime import datetime

import numpy as np
import xarray

from datacube.model import Measurement
from datacube_stats.utils.dates import datetime64_to_inttime
from datacube_stats.utils import da_nodata
from datacube_stats.stat_funcs import axisindex, argpercentile, _compute_medoid
from datacube_stats.stat_funcs import anynan, section_by_index, medoid_indices

from .core import Statistic, PerPixelMetadata, SimpleStatistic
from .core import StatsProcessingError, StatsConfigurationError


class NoneStat(Statistic):
    def compute(self, data):
        return data


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
        measurement_names = set(m.name for m in input_measurements)
        assert 'water' in measurement_names

        wet = Measurement(name='count_wet', dtype='int16', nodata=-1, units='1')
        dry = Measurement(name='count_clear', dtype='int16', nodata=-1, units='1')
        frequency = Measurement(name='frequency', dtype='float32', nodata=-1, units='1')

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
        measurement_names = set(m.name for m in input_measurements)
        if self.band1 not in measurement_names or self.band2 not in measurement_names:
            raise StatsConfigurationError('Input measurements for %s must include "%s" and "%s"' %
                                          (self.name, self.band1, self.band2))

        return [Measurement(name='_'.join([self.name, stat]), dtype='float32', nodata=-1, units='1')
                for stat in self.stats]


class TCWStats(Statistic):
    """
    Simple Tasseled Cap Wetness, Brightness and Greeness summary statistics.

    Based on the Crist 1985 RF coefficients
    You can provide your own coefficients, however at this stage it will run the same coefficients for all sensors.

    Default Tasseled Cap coefficient Values:
    brightness_coeff = {'blue':0.2043, 'green':0.4158, 'red':0.5524, 'nir':0.5741, 'swir1':0.3124, 'swir2':0.2303}
    greenness_coeff = {'blue':-0.1603, 'green':-0.2819, 'red':-0.4934, 'nir':0.7940, 'swir1':-0.0002, 'swir2':-0.1446}
    wetness_coeff = {'blue':0.0315, 'green':0.2021, 'red':0.3102, 'nir':0.1594, 'swir1':-0.6806, 'swir2':-0.6109}

    Default Thresholds used for calculating the percentage exceedance statistics for Brightness, Greenness and Wetness:
        brightness': 4000
        greenness': 600
        wetness': -600

    Outputs
    If you output as geotiff these will be your bands:
    Band1: pct_exceedance_brightness
    Band2: pct_exceedance_greenness
    Band3: pct_exceedance_wetness
    Band4: mean_brightness
    Band5: mean_greenness
    Band6: mean_wetness
    Band7: std_brightness
    Band8: std_greenness
    Band9: std_wetness

    """

    def __init__(self, thresholds=None, coeffs=None):
        if thresholds is None:
            self.thresholds = {
                'brightness': 4000,
                'greenness': 600,
                'wetness': -600
            }
        else:
            self.thresholds = thresholds

        if coeffs is None:
            self.coeffs = {
                'brightness': {'blue': 0.2043, 'green': 0.4158, 'red': 0.5524, 'nir': 0.5741,
                               'swir1': 0.3124, 'swir2': 0.2303},
                'greenness': {'blue': -0.1603, 'green': -0.2819, 'red': -0.4934, 'nir': 0.7940,
                              'swir1': -0.0002, 'swir2': -0.1446},
                'wetness': {'blue': 0.0315, 'green': 0.2021, 'red': 0.3102, 'nir': 0.1594,
                            'swir1': -0.6806, 'swir2': -0.6109}
            }
        else:
            self.coeffs = coeffs

    def compute(self, data):
        coeffs = self.coeffs
        thresholds = self.thresholds
        bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
        categories = ['brightness', 'greenness', 'wetness']

        results = {}
        for cat in categories:
            data[cat] = sum([data[band] * coeffs[cat][band] for band in bands])
            results['pct_exceedance_' + cat] = \
                data[cat].where(data[cat] > thresholds[cat]).count(dim='time')/data[cat].count(dim='time')

            results['mean_' + cat] = data[cat].mean(dim='time')
            results['std_' + cat] = data[cat].std(dim='time', keep_attrs=True, skipna=True)
            data = data.drop(cat)

        data = data.drop(bands)

        return xarray.Dataset(results, attrs=dict(crs=data.crs))

    def measurements(self, input_measurements):
        measurement_names = [
            'pct_exceedance_brightness',
            'pct_exceedance_greenness',
            'pct_exceedance_wetness',
            'mean_brightness',
            'mean_greenness',
            'mean_wetness',
            'std_brightness',
            'std_greenness',
            'std_wetness']
        return [Measurement(name=m_name, dtype='float32', nodata=-1, units='1')
                for m_name in measurement_names]


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
    :param per_pixel_metadata: list of metadata (source, observed, or observed_date) to attach
    """
    def __init__(self, stat_func=None, per_pixel_metadata=None):
        super(PerBandIndexStat, self).__init__(stat_func=stat_func)
        if per_pixel_metadata is None:
            self.per_pixel_metadata = []
        else:
            assert isinstance(per_pixel_metadata, Sequence)
            self.per_pixel_metadata = per_pixel_metadata

    def compute(self, data):
        index = super(PerBandIndexStat, self).compute(data)

        def index_dataset(var):
            return axisindex(data.data_vars[var.name].values, var.values)

        data_values = index.apply(index_dataset)

        all_values = [data_values]
        metadata = self.per_pixel_metadata

        if 'observed' in metadata or 'observed_date' in metadata:
            def index_time(var):
                return data.time.values[var.values]

            time_values = index.apply(
                index_time).rename(
                    OrderedDict((name, name + '_observed')
                                for name in index.data_vars))

        if 'observed' in metadata:
            all_values += [time_values]

        if 'observed_date' in metadata:
            text_values = time_values.apply(datetime64_to_inttime).rename(
                OrderedDict((name, name + '_date')
                            for name in time_values.data_vars))

            all_values += [text_values]

        if 'source' in metadata:
            def index_source(var):
                return data.source.values[var.values]

            source_values = index.apply(index_source).rename(OrderedDict((name, name + '_source')
                                                                         for name in index.data_vars))

            all_values += [source_values]

        return xarray.merge(all_values)

    def measurements(self, input_measurements):
        index_measurements = [Measurement(name=measurement.name + '_source', dtype='int8', nodata=-1, units='1')
                              for measurement in input_measurements]

        date_measurements = [Measurement(name=measurement.name + '_observed', dtype='float64', nodata=0,
                                         units='seconds since 1970-01-01 00:00:00')
                             for measurement in input_measurements]

        text_measurements = [Measurement(name=measurement.name + '_observed_date', dtype='int32', nodata=0,
                                         units='Date as YYYYMMDD')
                             for measurement in input_measurements]

        all_measurements = super(PerBandIndexStat, self).measurements(input_measurements)

        metadata = self.per_pixel_metadata

        if 'source' in metadata:
            all_measurements += index_measurements

        if 'observed' in metadata:
            all_measurements += date_measurements

        if 'observed_date' in metadata:
            all_measurements += text_measurements

        return all_measurements


class Percentile(PerBandIndexStat):
    """
    Per-band percentiles of observations through time.
    The different percentiles are stored in the output as separate bands.
    The q-th percentile of a band is named `{band}_PC_{q}`.

    :param q: list of percentiles to compute
    :param per_pixel_metadata: provenance metadata to attach to each pixel
    :arg minimum_valid_observations: if not enough observations are available,
                                     percentile will return `nodata`
    """
    def __init__(self, q,
                 minimum_valid_observations=0,
                 per_pixel_metadata=None):

        if isinstance(q, Sequence):
            self.qs = q
        else:
            self.qs = [q]

        self.minimum_valid_observations = minimum_valid_observations
        super(Percentile, self).__init__(per_pixel_metadata=per_pixel_metadata)

    def compute(self, data):
        # calculate masks for pixel without enough data
        arr = data.to_array().values
        invalid = anynan(arr, axis=0)
        count_valid = np.count_nonzero(~invalid, axis=0)
        not_enough = count_valid < self.minimum_valid_observations

        def single(q):
            stat_func = partial(xarray.Dataset.reduce, dim='time',
                                func=argpercentile, q=q)

            per_pixel_metadata = self.per_pixel_metadata

            renamed = data.rename({var: var + '_PC_' + str(q)
                                   for var in data.data_vars})

            result = PerBandIndexStat(stat_func=stat_func,
                                      per_pixel_metadata=per_pixel_metadata).compute(renamed)

            def mask_not_enough(var):
                nodata = da_nodata(var)
                var.values[not_enough] = nodata
                return var

            return result.apply(mask_not_enough, keep_attrs=True)

        return xarray.merge(single(q) for q in self.qs)

    def measurements(self, input_measurements):
        renamed = [Measurement(**{**m, 'name': m.name + '_PC_' + str(q)})
                   for q in self.qs
                   for m in input_measurements]

        return PerBandIndexStat(per_pixel_metadata=self.per_pixel_metadata).measurements(renamed)


class ObservedDaysSince(PerPixelMetadata):
    def __init__(self, since=datetime(1970, 1, 1), var_name='observed'):
        super(ObservedDaysSince, self).__init__(var_name)
        self._since = since

    def compute(self, data, selected_indexes):
        observed = data.time.values[selected_indexes] - np.datetime64(self._since)
        days_since = observed.astype('timedelta64[D]').astype('int16')

        return self._var_name, xarray.Variable(('y', 'x'), days_since)

    def measurement(self):
        return Measurement(name=self._var_name, dtype='int16', nodata=0,
                           units='days since {:%Y-%m-%d %H:%M:%S}'.format(self._since))


class ObservedDateInt(PerPixelMetadata):
    def compute(self, data, selected_indexes):
        observed = data.time.values[selected_indexes]
        observed_date = xarray.Variable(('y', 'x'), datetime64_to_inttime(observed))
        return self._var_name, observed_date

    def measurement(self):
        return Measurement(name=self._var_name, dtype='int32', nodata=0,
                           units='Date as YYYYMMDD')


class SourceIndex(PerPixelMetadata):
    def compute(self, data, selected_indexes):
        return self._var_name, xarray.Variable(('y', 'x'), data.source.values[selected_indexes])

    def measurement(self):
        return Measurement(name=self._var_name, dtype='int8', nodata=-1, units='1')


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
                                      [m.name for m in base])

        selected = [m for m in base if m.name in selected_names]

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

        def reduction(var: xarray.DataArray) -> xarray.DataArray:
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
                nodata = metadata_producer.measurement().nodata
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
