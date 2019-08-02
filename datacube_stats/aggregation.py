import xarray as xr
import numpy as np
import dask
import dask.array as da

from collections import Sequence
from datacube.virtual.impl import Transformation, Measurement
from .stat_funcs import argpercentile,  axisindex
from hdstats import pcm


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
    def __init__(self, eps=1e-3, num_threads=None, split=4):
        self.eps = eps
        self.num_threads = num_threads
        self.split = split

    def compute(self, data):
        """
        param xarray.Dataset data:
        :return: xarray.Dataset
        """
        dtypes = {}
        squashed = []
        for var in data.data_vars:
            dtypes[var] = data[var].dtype
            nodata = getattr(data[var], 'nodata', np.float32('nan'))
            if not np.isnan(nodata):
                data[var] = data[var].where(da.fabs(data[var] - nodata) > 1e-8)
            else:
                data[var].data = data[var].data.astype(np.float32)
            squashed.append(data[var].data.astype(np.float32))

        squashed = da.stack(squashed, axis=-1)
        chunk_size = np.ceil(np.array(squashed.chunksize[1:3])/self.split).astype(int)
        squashed = squashed.rechunk((-1,)+tuple(chunk_size)+(-1,))

        # Call Dale's function here
        squashed = squashed.map_blocks(lambda x, num_threads: pcm.gm(x.transpose([1, 2, 3, 0]),
                                                                     num_threads=num_threads, nocheck=True),
                                       num_threads=self.num_threads, name='hdstats_gm',
                                       drop_axis=0, dtype=np.float32)
        i = 0
        result = []
        for var in data.data_vars:
            variable = xr.DataArray(squashed[:, :, i].rechunk(data[var].data.chunksize[1:]),
                                    dims=data[var].dims[1:], name=var,
                                    attrs=data[var].attrs.copy())
            variable.data[da.isnan(variable.data)] = getattr(data[var], 'nodata', np.float32('nan'))
            variable.data = variable.data.astype(dtypes[var])

            for dim in variable.dims:
                variable.coords[dim] = data[var].coords[dim]

            result.append(variable)
            i += 1

        result = xr.merge(result)
        result.attrs = data.attrs.copy()

        return result

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
        data = data.chunk({'time': -1})
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


class Count(Transformation):
    """
    Count the number of True=1 along time axis

    Use `expressions: formula` to convert values as needed
    """
    def __init__(self):
        return

    def compute(self, data):
        data = data.chunk({'time': -1})
        results = []
        for var in data.data_vars:
            count = da.count_nonzero(data[var].data, axis=0)
            re = xr.DataArray(count, dims=data[var].dims[1:], name='count_'+var)
            for dim in re.dims:
                re.coords[dim] = data[var].coords[dim]
            re.attrs = dict(nodata=0, units=1, crs=data[var].crs)
            results.append(re)
        results = xr.merge(results)
        results.attrs['crs'] = data.crs
        return results

    def measurements(self, input_measurements):
        output_measurements = {}
        for m in input_measurements.keys():
            count = Measurement(name='count_'+m, dtype='int16', nodata=0, units='1')
            output_measurements.update({'count_'+m: count})

        return output_measurements
