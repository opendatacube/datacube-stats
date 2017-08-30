"""
Tests for the custom statistics functions

"""
from __future__ import absolute_import

from datacube_stats.statistics import nan_percentile, argpercentile, axisindex
import numpy as np
import xarray as xr

import pytest


def test_nan_percentile():
    # create array of shape(5,100,100) - image of size 100x100 with 5 layers
    test_arr = np.random.randint(0, 10000, 50000).reshape(5, 100, 100).astype(np.float32)
    np.random.shuffle(test_arr)
    # place random NaNs
    random_nans = np.random.randint(0, 50000, 500).astype(np.float32)
    for r in random_nans:
        test_arr[test_arr == r] = np.NaN

    # Test with single q
    q = 45
    input_arr = np.array(test_arr, copy=True)
    std_np_func = np.nanpercentile(input_arr, q=q, axis=0)
    new_func = nan_percentile(input_arr, q=q)

    assert np.allclose(std_np_func, new_func)

    # Test with all qs
    qs = range(0, 100)
    input_arr = np.array(test_arr, copy=True)
    std_np_func = np.nanpercentile(input_arr, q=qs, axis=0)
    new_func = nan_percentile(input_arr, q=qs)

    assert np.allclose(std_np_func, new_func)


def test_argpercentile():
    # Create random Data
    test_arr = np.random.randint(0, 10000, 50000).reshape(5, 100, 100).astype(np.float32)
    np.random.shuffle(test_arr)
    # place random NaNs
    rand_nan = np.random.randint(0, 50000, 500).astype(np.float32)
    for r in rand_nan:
        test_arr[test_arr == r] = np.NaN

    np_result = np.nanpercentile(test_arr, q=25, axis=0, interpolation='nearest')
    argpercentile_result = axisindex(test_arr, argpercentile(test_arr, q=25, axis=0), axis=0)
    assert np.isclose(np_result, argpercentile_result).all()


def test_xarray_reduce():
    arr = np.random.random((100, 100, 5))
    dataarray = xr.DataArray(arr, dims=('x', 'y', 'time'))

    def reduction(in_arr, axis):
        assert axis == 2
        output = np.average(in_arr, axis)
        return output

    dataarray = dataarray.reduce(reduction, dim='time')

    assert dataarray.dims == ('x', 'y')


@pytest.mark.xfail
def test_masked_count():

    arr = np.random.random((100, 100, 5))
    dataarray = xr.DataArray(arr, dims=('x', 'y', 'time'))

    # Added flag_values

    from datacube_stats.statistics import MaskedCount, ClearCount

    mc = MaskedCount(flats={'foo_bar': True, 'wop_zoo': False})

    result = mc.compute(dataarray)

    assert result


def test_new_geometry_median():
    from datacube_stats.statistics import NewGeomedianStatistic

    arr = np.random.random((5, 100, 100))
    dataarray = xr.DataArray(arr, dims=('time', 'y', 'x'))
    dataset = xr.Dataset(data_vars={'band1': dataarray, 'band2': dataarray})

    new_geomedian_stat = NewGeomedianStatistic()
    result = new_geomedian_stat.compute(dataset)

    assert result
