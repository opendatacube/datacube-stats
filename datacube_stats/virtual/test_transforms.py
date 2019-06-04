import numpy as np
import pytest
import xarray as xr
import dask.array as da
from datetime import datetime, timedelta
from datacube_stats.virtual.aggregation import Percentile, NewGeomedianStatistic
from datacube_stats.virtual.external import MangroveCC, TCIndex, MaskByValue

x_range = (-33000, -23000)
y_range = (-1310000, -1300000)

time_range = (datetime(2016, 1, 1), datetime(2016, 12, 31))
time_step = timedelta(days=8)
crs = 'EPSG:3577'
resolution = (25, -25)


def test_percentile():
    x_coords = np.arange(x_range[0], x_range[1], resolution[0]).astype(np.float)
    y_coords = np.arange(y_range[0], y_range[1], resolution[0]).astype(np.float)
    time = np.arange(time_range[0], time_range[1], time_step)
    bands = []
    for i in range(2):
        data = da.random.random(time.shape+x_coords.shape+y_coords.shape, chunks=(1, 200, 200))
        data = (100 * data - 5).astype(np.int16)
        data[data < 0] = -1
        bands.append(xr.DataArray(data, name='band' + str(i+1), dims=('time', 'y', 'x'),
                                  coords={'time': time, 'y': y_coords, 'x': x_coords},
                                  attrs={'crs': crs, 'nodata': -1, 'dtype': np.int16}))

    data = da.random.random(time.shape+x_coords.shape+y_coords.shape, chunks=(1, 200, 200))
    data = (data - .5 > 0).astype(np.bool)
    bands.append(xr.DataArray(data, name='bandq', dims=('time', 'y', 'x'),
                              coords={'time': time, 'y': y_coords, 'x': x_coords},
                              attrs={'crs': crs, 'nodata': -1, 'dtype': np.bool}))

    input_data = xr.merge(bands)
    input_data.attrs['crs'] = crs

    percentile = Percentile(q=[10, 50, 90], minimum_valid_observations=3, not_sure_mark=2, quality_band='bandq')
    output_data = percentile.compute(input_data)
    return output_data


def test_mangroves():
    x_coords = np.arange(x_range[0], x_range[1], resolution[0]).astype(np.float)
    y_coords = np.arange(y_range[0], y_range[1], resolution[0]).astype(np.float)
    time = np.array(time_range[0], dtype=np.datetime64).reshape(1,)
    data = da.random.random(time.shape+x_coords.shape+y_coords.shape, chunks=(1, 200, 200))
    data = (100 * data - 5).astype(np.int16)
    data[data < 0] = -1
    band = xr.DataArray(data, name='band1', dims=('time', 'y', 'x'),
                        coords={'time': time, 'y': y_coords, 'x': x_coords},
                        attrs={'crs': crs, 'nodata': -1, 'dtype': np.int16})
    band = band.to_dataset()
    band.attrs['crs'] = crs
    mangroves = MangroveCC(thresholds=[15, 40, 62], shape_file='maximum_extent_of_mangroves_Apr2019.shp')
    output_data = mangroves.compute(band)
    return output_data


def test_tci():
    x_coords = np.arange(x_range[0], x_range[1], resolution[0]).astype(np.float)
    y_coords = np.arange(y_range[0], y_range[1], resolution[0]).astype(np.float)
    time = np.array(time_range[0], dtype=np.datetime64).reshape(1,)
    band_names = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    bands = []
    for band in band_names:
        data = da.random.random(time.shape+x_coords.shape+y_coords.shape, chunks=(1, 200, 200))
        data = (10000 * data - 5).astype(np.int16)
        data[data <= 0] = -9999
        bands.append(xr.DataArray(data, name=band, dims=('time', 'y', 'x'),
                                  coords={'time': time, 'y': y_coords, 'x': x_coords},
                                  attrs={'crs': crs, 'nodata': -9999, 'dtype': np.int16}))
    input_data = xr.merge(bands)
    input_data.attrs['crs'] = crs
    tci = TCIndex()
    output_data = tci.compute(input_data)
    return output_data


def test_maskbyvalue():
    x_coords = np.arange(x_range[0], x_range[1], resolution[0]).astype(np.float)
    y_coords = np.arange(y_range[0], y_range[1], resolution[0]).astype(np.float)
    time = np.array(time_range[0], dtype=np.datetime64).reshape(1,)
    data = da.random.random(time.shape+x_coords.shape+y_coords.shape, chunks=(1, 200, 200))
    data = (100 * data - 5).astype(np.int16)
    data[data < 0] = -1
    band = xr.DataArray(data, name='band1', dims=('time', 'y', 'x'),
                        coords={'time': time, 'y': y_coords, 'x': x_coords},
                        attrs={'crs': crs, 'nodata': -1, 'dtype': np.int16})
    band = band.to_dataset()
    band.attrs['crs'] = crs
    try:
        mask = MaskByValue(mask_measurement_name='band1', greater_than=50, smaller_than=49)
    except Exception as e:
        print('test exception', e)
    mask = MaskByValue(mask_measurement_name='band1', greater_than=50)
    output_data = mask.compute(band)
    return output_data


def test_geomedian():
    x_coords = np.arange(x_range[0], x_range[1], resolution[0]).astype(np.float)
    y_coords = np.arange(y_range[0], y_range[1], resolution[0]).astype(np.float)
    time = np.arange(time_range[0], time_range[1], time_step)
    band_names = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    bands = []
    for band in band_names:
        data = da.random.random(time.shape+x_coords.shape+y_coords.shape, chunks=(1, 200, 200))
        data = (10000 * data - 5).astype(np.int16)
        data[data <= 0] = -9999
        bands.append(xr.DataArray(data, name=band, dims=('time', 'y', 'x'),
                                  coords={'time': time, 'y': y_coords, 'x': x_coords},
                                  attrs={'crs': crs, 'nodata': -9999, 'dtype': np.int16}))
    input_data = xr.merge(bands)
    input_data.attrs['crs'] = crs
    geomedian = NewGeomedianStatistic(num_threads=2)
    output_data = geomedian.compute(input_data)
    return output_data
