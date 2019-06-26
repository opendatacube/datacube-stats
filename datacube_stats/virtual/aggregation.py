import xarray as xr
import numpy as np
import dask
import dask.array as da

from collections import Sequence
from datacube.virtual.impl import Transformation, Measurement
from datacube_stats.stat_funcs import argpercentile,  axisindex
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

            nodata = getattr(data[var], 'nodata', None)
            if nodata is not None:
                data[var] = data[var].where(da.fabs(data[var] - nodata) > 1e-8)
            else:
                data[var] = data[var].astype(np.float)

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

                result.name = var + '_PC_' + str(q)
                result.attrs['nodata'] = nodata
                result = result.astype(data_type)
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
        :param xarray.Dataset data:
        :return: xarray.Dataset
        """
        dtypes = {}
        for var in data.data_vars:
            dtypes[var] = data[var].dtype
            nodata = getattr(data[var], 'nodata', None)
            if nodata is not None:
                data[var] = data[var].where(data[var] > nodata)
            else:
                data[var] = data[var].astype(np.float)
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
            squashed[var] = squashed[var].astype(dtypes[var])
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
