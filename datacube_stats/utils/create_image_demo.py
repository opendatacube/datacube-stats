#!/usr/bin/env python

import datacube
#from datacube.api.geo_xarray import _solar_day, _get_mean_longitude, append_solar_day
from datacube.api.grid_workflow import GridWorkflow, Tile
from datetime import datetime
import datetime as DT
from datacube.api.query import query_group_by
from datacube.utils.geometry import CRS, GeoBox
from datacube.utils import geometry
import numpy as np
import fiona
import warnings
import itertools
import rasterio.features
import shapely.ops
from shapely.geometry import shape, mapping
from datacube.api import API, make_mask, list_flag_names
from datacube.utils.geometry import CRS, GeoBox, Geometry
from hdmedians import nangeomedian
import copy

import xarray as xr


dc=datacube.Datacube(app='test')
#product = ['ls5_nbar_albers', 'ls8_nbar_albers']
product = ['ls5_nbar_albers', 'ls7_nbar_albers', 'ls8_nbar_albers']
#filepath = '/g/data/r78/intertidal/GA_tidal_model.shp'
filepath = '/g/data/r78/bxb547/GW_works/burdekin_polygons_albers.shp'
fildir = '/g/data/r78/bxb547/TIDE_MODEL/test_images/'

DEFAULT_PROFILE = {
    'blockxsize': 256,
    'blockysize': 256,
    'compress': 'lzw',
    'driver': 'GTiff',
    'interleave': 'band',
    'nodata': 'nan',
    'tiled': True}

mask_spec=dict()

mask_spec['flags'] = { 'contiguous': True,
          'cloud_acca': 'no_cloud',
          'cloud_fmask': 'no_cloud',
          'cloud_shadow_acca': 'no_cloud_shadow',
          'cloud_shadow_fmask': 'no_cloud_shadow',
          'blue_saturated': False,
          'green_saturated': False,
          'red_saturated': False,
          'nir_saturated': False,
          'swir1_saturated': False,
          'swir2_saturated': False
          }


def pq_fuser(dest, src):
    valid_bit = 8
    valid_val = (1 << valid_bit)
    no_data_dest_mask = ~(dest & valid_val).astype(bool)
    np.copyto(dest, src, where=no_data_dest_mask)
    both_data_mask = (valid_val & dest & src).astype(bool)
    np.copyto(dest, src & dest, where=both_data_mask)

def compute_count(data):
    # TODO Fix Hardcoded 'time' and pulling out first data var
    _, sample_data_var = next(iter(data.data_vars.items())) 
    count_values = sample_data_var.count(dim='time').rename('count_observations')
    return count_values


def compute(data):
            """
            :param xarray.Dataset data:
            :return: xarray.Dataset
            """
            # Assert data shape/dims
            data = data.to_array(dim='variable').transpose('x', 'y', 'variable', 'time').copy()
            data = data.reduce(apply_geomedian, dim='time', keep_attrs=True, f=nangeomedian, eps=1e-6,
                               maxiters=5000)

            return data.transpose('variable', 'y', 'x').to_dataset(dim='variable')


def apply_geomedian(inarray, f, axis=3, eps=1e-6, **kwargs):
        assert len(inarray.shape) == 4
        assert axis == 3

        maxiters = kwargs.get('maxiters', 5000) 
        xs, ys, bands, times = inarray.shape
        output = np.ndarray((xs, ys, bands), dtype=inarray.dtype)
        with warnings.catch_warnings():  # Don't print error about computing mean of empty slice
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for ix in range(xs):
                for iy in range(ys):
                    try:
                        output[ix, iy, :] = f(inarray[ix, iy, :, :], eps=eps, maxiters
=maxiters, axis=1)
                    except ValueError:
                        output[ix, iy, :] = np.nan
        return output


def geometry_mask(geoms, geobox, all_touched=False, invert=False):

    return rasterio.features.geometry_mask([geom.to_crs(geobox.crs) for geom in geoms],
                                           out_shape=geobox.shape,
                                           transform=geobox.affine,
                                           all_touched=all_touched,
                                           invert=invert)


def write_geotiff(filename, dataset, time_index=None, profile_override=None, desc=None):
    """
    Write an xarray dataset to a geotiff

    :attr bands: ordered list of dataset names
    :attr time_index: time index to write to file
    :attr dataset: xarray dataset containing multiple bands to write to file
    :attr profile_override: option dict, overrides rasterio file creation options.
    """
    profile_override = profile_override or {}

    dtypes = {val.dtype for val in dataset.data_vars.values()}
    assert len(dtypes) == 1  # Check for multiple dtypes
    profile = DEFAULT_PROFILE.copy()
    profile.update({
        'width': dataset.dims[dataset.crs.dimensions[1]],
        'height': dataset.dims[dataset.crs.dimensions[0]],
        'affine': dataset.affine,
        'crs': dataset.crs.crs_str,
        'count': len(dataset.data_vars),
        'dtype': str(dtypes.pop())
    })
    profile.update(profile_override)

    with rasterio.open(filename, 'w', **profile) as dest:
        if desc:
            dest.update_tags(TIFFTAG_IMAGEDESCRIPTION = desc['Comment1'] + ' ' + desc['Comment2'])
            dest.update_tags(TIFFTAG_DATETIME = str(datetime.now()))
        for bandnum, data in enumerate(dataset.data_vars.values(), start=1):
            dest.write(data.data, bandnum)

year = 0
with fiona.open(filepath) as input_region:
    crs = CRS(str(input_region.crs_wkt))
    for feature in input_region:
        Id = feature['properties']['Id']
        #if Id == 251:
        if Id == 5:
            year = feature['properties'].get('DN') 
            geom = feature['geometry']
            boundary_polygon = Geometry(geom, crs)
            break
year = year + 1
dt1 = str(year) + "-07-01" 
dt2 = str(year) + "-11-30" 
prod = 'ls5_pq_albers'
indexers = {'time':(dt1, dt2),  'geopolygon': boundary_polygon, 'group_by':'solar_day'}
for st in product:
    if st == 'ls7_nbar_albers':
        prod = 'ls7_pq_albers'
    if st == 'ls8_nbar_albers':
        prod = 'ls8_pq_albers'
    pq = dc.load(product=prod, fuse_func=pq_fuser, **indexers)
    print ("time observed %s" + str(pq.time.data))
    dates = pq.time.data.astype('M8[D]')
    dates = [datetime.strftime(dt, "%Y-%m-%d") for dt in  dates.tolist()]
    print ("observation dates %s" + str(dates))
    if len(pq) > 0:
        stats = "MEDIAN"
        print ("doing for no pq %s for stats %s " , st, stats)
        data = dc.load(product=st, measurements=['red', 'green', 'blue'], **indexers)
        ndata = copy.deepcopy(data)
        red = data.red.median(dim='time')
        green = data.green.median(dim='time')
        blue = data.blue.median(dim='time')
        mask = geometry_mask([boundary_polygon], data.geobox, invert=True)
        red = red.where(mask)
        green = green.where(mask)
        blue = blue.where(mask)
        ndata = ndata.isel(time=0)
        ndata.red.data = red.data
        ndata.green.data = green.data
        ndata.blue.data = blue.data
        filename=fildir + str(Id) + "_NO_PQ_" + stats + "_" + str(year) + "_RGB.tif"
        write_geotiff(filename=filename, dataset=ndata,
                          profile_override={'photometric':'RGB'})
        print ("median finished without pq  on %s", str(datetime.now()))

        mask = make_mask(pq, **mask_spec['flags'])
        ndata = data.where(mask['pixelquality'])
        red = ndata.red.median(dim='time')
        green = ndata.green.median(dim='time')
        blue = ndata.blue.median(dim='time')
        mask = geometry_mask([boundary_polygon], data.geobox, invert=True)
        ndata = ndata.isel(time=0)
        ndata.red.data = red.data
        ndata.green.data = green.data
        ndata.blue.data = blue.data
        ndata = ndata.where(mask)
        filename=fildir + str(Id) + "_" + stats + "_" + str(year) + "_RGB.tif"
        write_geotiff(filename=filename, dataset=ndata,
                          profile_override={'photometric':'RGB'})
        print ("median finished with pq on %s", str(datetime.now()))
        stats = "GEOMED"
        data = dc.load(product=st, measurements=['red', 'green', 'blue', 'nir', 'swir1', 'swir2'], **indexers)
        mask = make_mask(pq, **mask_spec['flags'])
        data = data.where(mask['pixelquality'])
        ndata = compute_count(data)
        mask = geometry_mask([boundary_polygon], data.geobox, invert=True)
        ndata = ndata.where(mask)
        ds = data.coords.to_dataset()
        ds['count_observations'] = ndata
        filename=fildir + str(Id) + "_COUNT_" + str(year) + ".tif"
        write_geotiff(filename=filename, dataset=ds)
        print ('observation count finished and created %s %s', filename, str(datetime.now()))
 
        data = data/10000
        print ('computing statistics %s on %s ', stats, str(datetime.now()))
        data = compute(data)
        data = data.where(mask)
        filename=fildir + str(Id) + "_" + stats + "_" + str(year) + "_RGB.tif"
        write_geotiff(filename=filename, dataset=data[['red', 'green', 'blue']],
                          profile_override={'photometric':'RGB'})
        print ('computing finished and created for geomedian %s %s', filename, str(datetime.now()))
        break

