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
import itertools
import rasterio.features
import shapely.ops
from shapely.geometry import shape, mapping
from datacube.api import API, make_mask, list_flag_names
from datacube.utils.geometry import CRS, GeoBox, Geometry
import datacube.helpers as dh

import xarray as xr


dc=datacube.Datacube(app='test')

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


def sensible_where(data, mask):
    data = _convert_to_floats(data)  # This is stripping out variable attributes
    return data.where(mask)


def _convert_to_floats(data):
    # Use float32 instead of float64 if input dtype is int16
    assert isinstance(data, xr.Dataset)
    for name, dataarray in data.data_vars.items():
        if dataarray.dtype != np.int16:
            return data
    return data.apply(lambda d: d.astype(np.float32), keep_attrs=True)

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

product = ['ls5_nbar_albers', 'ls8_nbar_albers']
#product = ['ls5_nbar_albers']
filepath = '/g/data/u46/users/bxb547/otps/vector_data/GA_tidal_model.shp'
fildir = '/g/data/r78/bxb547/TIDE_MODEL/source_data/'
with fiona.open(filepath) as input_region:
    crs = CRS(str(input_region.crs_wkt))
    for feature in input_region:
        Id = feature['properties']['Id']
        #if Id == 251:
        if Id == 5:
        #if Id == 268:
            geom = feature['geometry']
            boundary_polygon = Geometry(geom, crs)
            break

#dates = ['2010-03-28', '2011-04-16', '2013-12-01', '2014-12-04', '2014-12-20','2015-12-23', '2016-01-08' ]
log_date = [['2010-02-03', 'f'], ['2010-02-05', 'e'], ['2010-02-19', 'pl'], ['2010-03-23', 'e'], ['2011-02-24', 'e'], ['2011-03-10', 'pl'], ['2011-03-12', 'e'], ['2013-04-17', 'pl'], ['2013-06-28', 'f'], ['2013-07-14', 'pl'], ['2014-01-06', 'pl'], ['2014-01-22', 'e'], ['2014-03-04', 'f'], ['2014-03-20', 'f'], ['2014-04-04', 'f'], ['2014-04-05', 'e'], ['2014-04-20', 'f'], ['2014-04-21', 'e'], ['2014-05-06', 'e'], ['2014-05-22', 'e'], ['2014-07-01', 'f'], ['2014-07-17', 'f'], ['2015-01-25', 'pl'], ['2015-02-10', 'e'], ['2015-03-23', 'f'], ['2015-04-08', 'f'], ['2015-04-23', 'f'], ['2015-04-24', 'e'], ['2015-05-09', 'f'], ['2015-05-10', 'e'], ['2015-05-25', 'e'], ['2015-06-10', 'e'], ['2015-07-20', 'f'], ['2015-08-05', 'f'], ['2016-02-13', 'pl'], ['2016-02-29', 'e'], ['2016-04-10', 'f'], ['2016-04-26', 'f'], ['2016-05-11', 'f'], ['2016-05-12', 'e'], ['2016-05-27', 'f'], ['2016-05-28', 'e'], ['2016-06-12', 'e'], ['2016-06-28', 'e'], ['2016-08-07', 'f'], ['2016-08-23', 'f']]

#log_date = [['2010-02-18', 'f'], ['2010-03-22', 'f'], ['2010-06-03', 'f'], ['2010-06-19', 'e'], ['2010-11-26', 'f'], ['2010-12-12', 'f'], ['2010-12-28', 'pl'], ['2011-01-13', 'e'], ['2011-06-22', 'f'], ['2011-07-08', 'e'], ['2011-10-19', 'f'], ['2013-05-01', 'f'], ['2013-07-13', 'f'], ['2013-07-29', 'f'], ['2013-08-14', 'f'], ['2013-10-24', 'f'], ['2013-11-09', 'f'], ['2014-01-05', 'f'], ['2014-01-21', 'f'], ['2014-02-06', 'f'], ['2014-02-22', 'f'], ['2014-05-20', 'f'], ['2014-07-16', 'f'], ['2014-08-01', 'f'], ['2014-08-17', 'f'], ['2014-09-02', 'f'], ['2014-11-12', 'f'], ['2014-11-28', 'f'], ['2014-12-14', 'f'], ['2015-01-24', 'f'], ['2015-02-09', 'f'], ['2015-02-25', 'f'], ['2015-03-13', 'f'], ['2015-05-23', 'f'], ['2015-06-08', 'f'], ['2015-06-24', 'e'], ['2015-08-04', 'f'], ['2015-08-20', 'f'], ['2015-09-05', 'f'], ['2015-09-21', 'f'], ['2015-12-01', 'f'], ['2015-12-17', 'f'], ['2016-01-02', 'pl'], ['2016-01-18', 'e'], ['2016-02-12', 'f'], ['2016-02-28', 'f'], ['2016-03-15', 'f'], ['2016-03-31', 'f'], ['2016-06-10', 'f'], ['2016-06-26', 'f'], ['2016-07-12', 'e'], ['2016-08-22', 'f'], ['2016-09-07', 'f'], ['2016-09-23', 'f'], ['2016-10-09', 'f']]













dates = list()
for dt in log_date:
    dates.append(dt[0])

indexers = {'geopolygon': boundary_polygon,  'group_by':'solar_day'}
print "length of dates " + str(len(dates)) + str(dates)
for dt in dates:
    dt1 = datetime.strptime(dt, "%Y-%m-%d")
    dt2 = dt1 + DT.timedelta(0,86399) 
    prod = 'ls5_pq_albers'
    for st in product:
        if st == 'ls8_nbar_albers':
            prod = 'ls8_pq_albers'
        pq = dc.load(product=prod, time=(dt1, dt2), fuse_func=pq_fuser, **indexers)
        
        if len(pq) > 0:
            data = dc.load(product=st, time=(dt1, dt2), measurements=['red', 'green', 'blue'], **indexers)
            #mask = make_mask(pq, **mask_spec['flags'])
            #import pdb; pdb.set_trace()
            #data = sensible_where(data, mask)
            #data = data.where(mask['pixelquality'])
            mask = geometry_mask([boundary_polygon], data.geobox, invert=True)
            ndata = data.where(mask)
            ndata = ndata.isel(time=0)
            filename=fildir + str(Id) + "_NO_MASK_" + prod.split('_')[0] + "_" + str(dt) + "_RGB.tif"
            print "writing for " + filename
            write_geotiff(filename=filename, dataset=ndata[['red', 'green', 'blue']],
                              profile_override={'photometric':'RGB'})
            mask = make_mask(pq, **mask_spec['flags'])
            data = data.where(mask['pixelquality'])
            mask = geometry_mask([boundary_polygon], data.geobox, invert=True)
            data = data.where(mask)
            data = data.isel(time=0)
            filename=fildir + str(Id) + "_" + prod.split('_')[0] + "_" + str(dt) + "_RGB.tif"
            print "writing for " + filename
            write_geotiff(filename=filename, dataset=data[['red', 'green', 'blue']],
                              profile_override={'photometric':'RGB'})
            break

