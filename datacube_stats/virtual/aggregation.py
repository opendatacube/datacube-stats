import xarray
import numpy as np
import pandas as pd

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
        # calculate masks for pixel without enough data
        for var in data.data_vars:
            if self.quality_band is not None:
                if var == self.quality_band:
                    continue
            nodata = getattr(data[var], 'nodata', None)
            if nodata is not None:
                data[var].attrs['dtype'] = data[var].dtype
                data[var] = data[var].where(data[var] > nodata)

        def single(q):
            stat_func = partial(xarray.Dataset.reduce, dim='time', keep_attrs=True,
                                func=argpercentile, q=q)
            if self.quality_band is not None:
                result = stat_func(data.drop(self.quality_band))
            else:
                result = stat_func(data)

            def index_dataset(var):
                return axisindex(data.data_vars[var.name].values, var.values)

            result = result.apply(index_dataset)

            def mask_not_enough(var):
                nodata = getattr(data[var.name], 'nodata', -1)
                valid_count = data[var.name].count(dim='time')

                if self.quality_band is not None:
                    quality_count = data[self.quality_band].where(data[self.quality_band]).count(dim='time')
                    not_sure = (quality_count == valid_count).where(valid_count < self.minimum_valid_observations) == 1
                    sure_not = (quality_count != valid_count).where(valid_count < self.minimum_valid_observations) == 1
                else:
                    not_sure = None
                    sure_not = valid_count < self.minimum_valid_observations

                if not_sure is not None:
                    if self.not_sure_mark is not None:
                        var.values[not_sure] = self.not_sure_mark
                    else:
                        var.values[not_sure] = nodata

                var.values[sure_not] = nodata
                var.values[np.isnan(var.values)] = nodata
                var.attrs['nodata'] = nodata

                if data[var.name].attrs['dtype'] == 'int8':
                    data_type = 'int16'
                else:
                    data_type = data[var.name].attrs['dtype']

                var = var.astype(data_type)
                return var

            return result.apply(mask_not_enough, keep_attrs=True).rename(
                                {var: var + '_PC_' + str(q) for var in result.data_vars})

        result = xarray.merge(single(q) for q in self.qs)
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
