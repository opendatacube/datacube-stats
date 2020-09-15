from copy import copy

import numpy as np
import xarray
import sys

from .core import Statistic, StatsProcessingError, Measurement

GEOMEDIAN_STATS = {}

try:
    from hdstats import pcm

    class GeoMedian(Statistic):
        def __init__(self, eps=1e-3, num_threads=None):
            super(GeoMedian, self).__init__()
            self.eps = eps
            self.num_threads = num_threads

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
            squashed = pcm.gm(squashed.data, num_threads=self.num_threads)
            all_zeros = (squashed == 0.).all(axis=-1)
            squashed[all_zeros] = np.nan

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

    GEOMEDIAN_STATS['geomedian'] = GeoMedian

    class SpectralMAD(Statistic):
        def __init__(self, eps=1e-3, num_threads=None):
            super(SpectralMAD, self).__init__()
            self.eps = eps
            self.num_threads = num_threads

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
            gm = pcm.gm(squashed.data, num_threads=self.num_threads)
            squashed = pcm.smad(squashed.data, gm, num_threads=self.num_threads)

            # Jam the raw numpy array back into a pleasantly labelled DataArray
            as_datarray = xarray.DataArray(squashed, dims=output_dimensions, coords=output_coords)

            return as_datarray.transpose(*output_dimensions).to_dataset(name='smad')

        def measurements(self, input_measurements):
            return [Measurement(name='smad', dtype='float32', nodata=np.nan, units='1')]

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

    GEOMEDIAN_STATS['spectral_mad'] = SpectralMAD
except ImportError:
    import warnings
    warnings.warn("hdstats required", ImportWarning)
