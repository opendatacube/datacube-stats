import xarray as xr
import numpy as np
import dask
import dask.array as da

from collections import Sequence
from datacube.virtual.impl import Transformation, Measurement
from .stat_funcs import argpercentile,  axisindex
from pcm import gmpcm


class Percentile(Transformation):
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
                 not_sure_mark=None,
                 quality_band=None):

        if isinstance(q, Sequence):
            self.qs = q
        else:
            self.qs = [q]

        self.minimum_valid_observations = minimum_valid_observations
        self.not_sure_mark = not_sure_mark
        self.quality_band = quality_band

    def compute(self, data):

        def single(var_data, q, nodata):
            indices = argpercentile(var_data, q=q, axis=-1)
            result = axisindex(var_data, index=indices, axis=-1)
            result[np.isnan(result)] = nodata
            return result

        percentile = []
        quality_count = None

        data = data.chunk({'time': -1})
        if self.quality_band is not None:
            quality_count = (data[self.quality_band].data).sum(axis=0)
            data = data.drop(self.quality_band)

        for var in data.data_vars:
            if data[var].dtype == 'int8':
                data_type = 'int16'
            else:
                data_type = data[var].dtype

            nodata = getattr(data[var], 'nodata', float('nan'))
            if not np.isnan(nodata):
                data[var] = data[var].where(da.fabs(data[var] - nodata) > 1e-8)
            else:
                data[var].data = data[var].data.astype(np.float)

            valid_count = data[var].data.shape[0] - da.isnan(data[var].data).sum(axis=0)
            # differentiate "sure not" and "not sure"
            if quality_count is not None:
                not_sure = da.logical_and((quality_count == valid_count),
                                          (valid_count < self.minimum_valid_observations))
                sure_not = da.logical_and((quality_count != valid_count),
                                          (valid_count < self.minimum_valid_observations))
            else:
                not_sure = None
                sure_not = valid_count < self.minimum_valid_observations

            if nodata is None:
                nodata = -1

            for q in self.qs:
                result = xr.apply_ufunc(single, data[var], kwargs={'q': q, 'nodata': nodata},
                                        input_core_dims=[['time']], output_dtypes=[np.float],
                                        dask='parallelized', keep_attrs=True)

                if not_sure is not None:
                    if self.not_sure_mark is not None:
                        result.data[not_sure] = self.not_sure_mark
                    else:
                        result.data[not_sure] = nodata
                result.data[sure_not] = nodata

                result.data = result.data.astype(data_type)
                result.name = var + '_PC_' + str(q)
                percentile.append(result)

        result = xr.merge(percentile)
        result.attrs['crs'] = data.attrs['crs']
        return result

    def measurements(self, input_measurements):
        renamed = dict()
        for key, m in input_measurements.items():
            if self.quality_band is not None:
                if m.name == self.quality_band:
                    continue

            if m.dtype == 'int8':
                data_type = 'int16'
            else:
                data_type = m.dtype

            for q in self.qs:
                renamed[key + '_PC_' + str(q)] = Measurement(**{**m, 'name': key + '_PC_' + str(q), 'dtype': data_type})
        return renamed


class NewGeomedianStatistic(Transformation):
    """
        geomedian
    """
    def __init__(self, eps=1e-3, num_threads=None):
        self.eps = eps
        self.num_threads = num_threads

    def compute(self, data):
        """
        param xarray.Dataset data:
        :return: xarray.Dataset
        """
        dtypes = {}
        for var in data.data_vars:
            dtypes[var] = data[var].dtype
            nodata = getattr(data[var], 'nodata', np.float('nan'))
            if not np.isnan(nodata):
                data[var] = data[var].where(da.fabs(data[var] - nodata) > 1e-8)
            else:
                data[var].data = data[var].data.astype(np.float)
        # We need to reshape our data into Time, Y, X, Band
        squashed_together_dimensions = self._vars_to_transpose(data)

        squashed = data.to_array(dim='variable').transpose(*squashed_together_dimensions)
        assert squashed.dims == squashed_together_dimensions

        # Call Dale's function here
        squashed = squashed.chunk({'time': -1, 'variable': -1})
        result = xr.apply_ufunc(lambda x, num_threads: gmpcm(x, num_threads=num_threads), squashed,
                                kwargs={'num_threads': self.num_threads},
                                input_core_dims=[['time']], output_dtypes=[np.float],
                                dask='parallelized', keep_attrs=True)
        squashed = result.to_dataset(dim='variable')
        for var in squashed.data_vars:
            squashed[var].data = squashed[var].data.astype(dtypes[var])
            squashed[var].attrs = data[var].attrs.copy()

        return squashed

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
            return ('time', 'y', 'x', 'variable')
        else:
            return ('time', 'latitude', 'longitude', 'variable')

    def measurements(self, input_measurements):
        return input_measurements


class WofsStats(Transformation):
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
        wet.attrs = dict(nodata=-1, units=1, crs=data.crs)
        dry = ((data.water == 0) | (data.water == 4)).sum(dim='time')
        dry.attrs = dict(nodata=-1, units=1, crs=data.crs)
        clear = wet + dry
        with np.errstate(divide='ignore', invalid='ignore'):
            frequency = wet / clear
            frequecy.attrs = dict(nodata=-1, units=1, crs=data.crs)
        if self.freq_only:
            return xr.Dataset({'frequency': frequency}, attrs=dict(crs=data.crs))
        else:
            return xr.Dataset({'count_wet': wet,
                               'count_clear': clear,
                               'frequency': frequency}, attrs=dict(crs=data.crs))

    def measurements(self, input_measurements):
        measurement_names = set(m.name for m in input_measurements)
        assert 'water' in measurement_names

        wet = Measurement(name='count_wet', dtype='int16', nodata=-1, units='1')
        dry = Measurement(name='count_clear', dtype='int16', nodata=-1, units='1')
        frequency = Measurement(name='frequency', dtype='float32', nodata=-1, units='1')

        if self.freq_only:
            return {'frequency': frequency}
        else:
            return {'count_wet': wet, 'count_clear': dry, 'frequency': frequency}
