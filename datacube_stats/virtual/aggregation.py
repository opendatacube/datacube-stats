import xarray as xr
import numpy as np
import dask
import dask.array as da

from collections import Sequence
from functools import partial
from datacube.virtual.impl import Transformation, Measurement
from datacube_stats.stat_funcs import argpercentile, anynan, axisindex


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

        def single(var_data, q, nodata,):
            indices = argpercentile(var_data, q=q, axis=-1)
            result = axisindex(var_data, index=indices, axis=-1)
            result[np.isnan(result)] = nodata 
            return result

        percentile = []
        quality_count = None

        if self.quality_band is not None:
            quality_count =  (data[self.quality_band].data == True).sum(axis=0)
        for var in data.data_vars:
            if self.quality_band is not None:
                if var == self.quality_band:
                    continue

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
                not_sure = da.logical_and((quality_count == valid_count), (valid_count < self.minimum_valid_observations))
                sure_not = da.logical_and((quality_count != valid_count), (valid_count < self.minimum_valid_observations))
            else:
                not_sure = None
                sure_not = valid_count < self.minimum_valid_observations

            if nodata is None:
                nodata = -1

            for q in self.qs:
                result = xr.apply_ufunc(single, data[var], kwargs={'q':q, 'nodata': nodata}, input_core_dims=[['time']],
                                        output_dtypes=[np.float], dask='parallelized', keep_attrs=True)

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
