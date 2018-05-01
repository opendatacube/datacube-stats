from pathlib import Path

from datacube.utils import geometry
from affine import Affine
from click.testing import CliRunner
from datacube_stats.main import main

import pytest

try:
    import otps
    OTPS_MODULE_EXISTS = True
except ImportError:
    OTPS_MODULE_EXISTS = False

CONFIG_TEMPLATE = """
## Define inputs to perform statistics on
sources:
  - product: ls8_nbar_albers
    measurements: [blue, green]
    group_by: solar_day
    resampling: bilinear
    masks:
      - product: ls8_pq_albers
        measurement: pixelquality
        group_by: solar_day
        fuse_func: datacube.helpers.ga_pq_fuser
        flags:
          contiguous: True
          cloud_acca: no_cloud
          cloud_fmask: no_cloud
          cloud_shadow_acca: no_cloud_shadow
          cloud_shadow_fmask: no_cloud_shadow
          blue_saturated: False
          green_saturated: False
          red_saturated: False
          nir_saturated: False
          swir1_saturated: False
          swir2_saturated: False

## Define whether and how to chunk over time
date_ranges:
  start_date: 2015-01-01
  end_date: 2015-02-01
  stats_duration: 1m
  step_size: 1m

storage:
  driver: NetCDF CF

  crs: EPSG:3577
  tile_size:
          x: 100000.0
          y: 100000.0
  resolution:
          x: 250
          y: -250
  chunking:
      x: 200
      y: 200
      time: 1
  dimension_order: [time, y, x]

input_region:
   tile: [12, -43]

location: /g/data/u46/users/ia1511/Work/data/dummy

## Define statistics to perform and how to store the data
output_products:
 - name: landsat_seasonal_mean
   statistic: simple
   statistic_args:
      reduction_function: mean
   output_params:
     zlib: True
     fletcher32: True
   file_path_template: 'SR_N_MEAN/SR_N_MEAN_3577_{x:02d}_{y:02d}_{epoch_start:%Y%m%d}.nc'

"""

CONFIG_TEMPLATE_ITEM_NDWI = """
## Define inputs to perform statistics on
sources:
  - product: ls8_nbar_albers
    measurements: [blue, green, red, nir, swir1, swir2]
    group_by: solar_day
    resampling: bilinear
    masks:
      - product: ls8_pq_albers
        measurement: pixelquality
        group_by: solar_day
        fuse_func: datacube.helpers.ga_pq_fuser
        flags:
          contiguous: True
          cloud_acca: no_cloud
          cloud_fmask: no_cloud
          cloud_shadow_acca: no_cloud_shadow
          cloud_shadow_fmask: no_cloud_shadow
          blue_saturated: False
          green_saturated: False
          red_saturated: False
          nir_saturated: False
          swir1_saturated: False
          swir2_saturated: False

## Define whether and how to chunk over time
date_ranges:
  start_date: 2015-01-01
  end_date: 2015-04-01
  stats_duration: 3m
  step_size: 3m

storage:
  driver: NetCDF CF

  crs: EPSG:3577
  tile_size:
          x: 100000.0
          y: 100000.0
  resolution:
          x: 25000
          y: -25000
  chunking:
      x: 200
      y: 200
      time: 1
  dimension_order: [time, y, x]

input_region:
   from_file: /g/data/v10/ITEM/ITEMv2_tidalmodel.shp
   feature_id: [280]

location: /g/data/u46/users/ia1511/Work/data/dummy

output_products:
 - name: med_ndwi
   statistic: med_ndwi
   # statistic: std_ndwi
   output_params:
      zlib: True
      fletcher32: True
   file_path_template: 'ITEM_{x}_{y}_{epoch_start:%Y%m%d}_{epoch_end:%Y%m%d}_{stat_name}.nc'
   product_type: ITEM

filter_product:
  method: by_tide_height
  args:
     # tide_range used to differentiate item with low/high composite and for exploring future incremental change
     tide_range: 10
     tide_percent: 10

"""
CONFIG_TEMPLATE_ITEM_STD = """
## Define inputs to perform statistics on
sources:
  - product: ls8_nbar_albers
    measurements: [blue, green, red, nir, swir1, swir2]
    group_by: solar_day
    resampling: bilinear
    masks:
      - product: ls8_pq_albers
        measurement: pixelquality
        group_by: solar_day
        fuse_func: datacube.helpers.ga_pq_fuser
        flags:
          contiguous: True
          cloud_acca: no_cloud
          cloud_fmask: no_cloud
          cloud_shadow_acca: no_cloud_shadow
          cloud_shadow_fmask: no_cloud_shadow
          blue_saturated: False
          green_saturated: False
          red_saturated: False
          nir_saturated: False
          swir1_saturated: False
          swir2_saturated: False

## Define whether and how to chunk over time
date_ranges:
  start_date: 2015-01-01
  end_date: 2015-04-01


storage:
  driver: NetCDF CF

  crs: EPSG:3577
  tile_size:
          x: 100000.0
          y: 100000.0
  resolution:
          x: 25000
          y: -25000
  chunking:
      x: 200
      y: 200
      time: 1
  dimension_order: [time, y, x]

input_region:
   from_file: /g/data/v10/ITEM/ITEMv2_tidalmodel.shp
   feature_id: [280]

location: /g/data/u46/users/ia1511/Work/data/dummy

output_products:
 - name: med_stddev
   statistic: std_ndwi
   output_params:
      zlib: True
      fletcher32: True
   file_path_template: 'ITEM_{x}_{y}_{epoch_start:%Y%m%d}_{epoch_end:%Y%m%d}_{stat_name}.nc'
   product_type: ITEM

filter_product:
  method: by_tide_height
  args:
     # tide_range used to differentiate item with low/high composite and for exploring future incremental change
     tide_range: 10
     tide_percent: 10

"""
CONFIG_TEMPLATE_DRY = """

## Define inputs to perform statistics on
sources:
  - product: ls5_nbar_albers
    name: dry_period
    measurements: [blue, green]
    group_by: solar_day
    masks:
      - product: ls5_pq_albers
        measurement: pixelquality
        group_by: solar_day
        fuse_func: datacube.helpers.ga_pq_fuser
        flags:
          contiguous: True
          cloud_acca: no_cloud
          cloud_fmask: no_cloud
          cloud_shadow_acca: no_cloud_shadow
          cloud_shadow_fmask: no_cloud_shadow
          blue_saturated: False
          green_saturated: False
          red_saturated: False
          nir_saturated: False
          swir1_saturated: False
          swir2_saturated: False


## Define whether and how to chunk over time
date_ranges:
  start_date: 1993-01-01
  end_date: 1994-01-01


storage:
  driver: NetCDF CF

  crs: EPSG:3577
  tile_size:
          x: 100000.0
          y: 100000.0
  resolution:
          x: 250
          y: -250
  chunking:
      x: 200
      y: 200
      time: 1
  dimension_order: [time, y, x]

input_region:
   from_file: /g/data/u46/users/ia1511/Work/data/ITEM-files/bur_dry_albers.shp
   feature_id: [49]

location: /g/data/u46/users/ia1511/Work/data/dummy

output_products:
 - name: dry
   statistic: geomedian
   output_params:
      zlib: True
      fletcher32: True
   file_path_template: 'GW_DRY_{x}_{y}.nc'
   product_type: COMPOSITE DRY

filter_product:
  method: by_hydrological_months
  args:
     type: dry
# Here is to consider these months for the following year from polygon data
     months: ['10', '11']
"""
CONFIG_TEMPLATE_WET = """
## Define inputs to perform statistics on
global_attributes:
  title: WET composite
sources:
  - product: ls5_nbar_albers
    name: wet_period
    measurements: [blue, green]
    group_by: solar_day
    masks:
      - product: ls5_pq_albers
        measurement: pixelquality
        group_by: solar_day
        fuse_func: datacube.helpers.ga_pq_fuser
        flags:
          contiguous: True
          cloud_acca: no_cloud
          cloud_fmask: no_cloud
          cloud_shadow_acca: no_cloud_shadow
          cloud_shadow_fmask: no_cloud_shadow
          blue_saturated: False
          green_saturated: False
          red_saturated: False
          nir_saturated: False
          swir1_saturated: False
          swir2_saturated: False

## Define whether and how to chunk over time
date_ranges:
  start_date: 2008-01-01
  end_date: 2012-01-01

## Define output directory and file structure
location: '/g/data/r78/tmp'

input_region:
  from_file: /g/data/u46/users/ia1511/Work/data/ITEM-files/bur_dry_albers.shp
  feature_id: [49]

storage:
  driver: NetCDF CF

  crs: EPSG:3577
  tile_size:
          x: 100000.0
          y: 100000.0
  resolution:
          x: 25
          y: -25
  chunking:
      x: 200
      y: 200
      time: 1
  dimension_order: [time, y, x]


## Define statistics to perform and how to store the data

output_products:
 - name: wet
   statistic: geomedian
   output_params:
      zlib: True
      fletcher32: True
   file_path_template: 'GW_WET_{x}_{y}.nc'
   product_type: COMPOSITE WET

filter_product:
  method: by_hydrological_months
  args:
     type: wet
# Here is to consider these months for the following year from polygon data
     months: ['10', '11']
"""

CONFIG_FILENAME = 'config.yaml'


def sample_geometry():
    gb = geometry.GeoBox(40, 40, Affine(2500, 0.0, 1200000.0, 0.0, -2500, -4300000.0), geometry.CRS('EPSG:3577'))
    json = gb.extent.json
    return json


RUNNING_ON_NCI_ENV = Path('/g/data/u46').exists()


# takes ~30 seconds to complete
@pytest.mark.xfail(not RUNNING_ON_NCI_ENV,
                   reason="This test currently expects to be run in the DEA environment on NCI.")
def test_input_region_single_tile():
    runner = CliRunner()
    with runner.isolated_filesystem() as tmpdir:
        with open(CONFIG_FILENAME, 'w') as f:
            f.write(CONFIG_TEMPLATE)

        result = runner.invoke(main, ['-v', '-v', '-v', CONFIG_FILENAME,
                                      '--output-location', str(Path(tmpdir).absolute())])

        assert 'error' not in result.output.lower()
        assert 'exception' not in result.output.lower()
        assert result.exit_code == 0

        outputfile = Path(tmpdir) / 'SR_N_MEAN' / 'SR_N_MEAN_3577_12_-43_20150101.nc'
        assert outputfile.exists()


@pytest.mark.skip(reason="currently NDWI stats are unavailable")
@pytest.mark.xfail(not OTPS_MODULE_EXISTS, reason="otps module is not available")
def test_input_region_from_shapefile_item_ndwi():
    runner = CliRunner()
    with runner.isolated_filesystem() as tmpdir:
        with open(CONFIG_FILENAME, 'w') as f:
            f.write(CONFIG_TEMPLATE_ITEM_NDWI)

        outputfile = Path(tmpdir) / 'ITEM_280_142.63_-10.31_PER_10_20150101_20150401_medndwi.nc'

        result = runner.invoke(main, ['-v', '-v', '-v', CONFIG_FILENAME,
                                      '--output-location', str(Path(tmpdir).absolute())])

        assert 'error' not in result.output.lower()
        assert 'exception' not in result.output.lower()
        assert result.exit_code == 0

        assert outputfile.exists()


@pytest.mark.skip(reason="currently NDWI stats are unavailable")
@pytest.mark.xfail(not OTPS_MODULE_EXISTS, reason="otps module is not available")
def test_input_region_from_shapefile_item_std():
    runner = CliRunner()
    with runner.isolated_filesystem() as tmpdir:
        with open(CONFIG_FILENAME, 'w') as f:
            f.write(CONFIG_TEMPLATE_ITEM_STD)

        outputfile = Path(tmpdir) / 'ITEM_280_142.63_-10.31_PER_10_20150101_20150401_STD.nc'

        result = runner.invoke(main, ['-v', '-v', '-v', CONFIG_FILENAME,
                                      '--output-location', str(Path(tmpdir).absolute())])

        assert 'error' not in result.output.lower()
        assert 'exception' not in result.output.lower()
        assert result.exit_code == 0

        assert outputfile.exists()


# takes ~1 mins to complete
@pytest.mark.xfail(not RUNNING_ON_NCI_ENV,
                   reason="This test currently expects to be run in the DEA environment on NCI.")
def test_input_region_from_shapefile_dry():
    runner = CliRunner()
    with runner.isolated_filesystem() as tmpdir:
        with open(CONFIG_FILENAME, 'w') as f:
            f.write(CONFIG_TEMPLATE_DRY)

        outputfile = Path(tmpdir) / 'GW_DRY_49_1991_1992.nc'

        result = runner.invoke(main, ['-v', '-v', '-v', CONFIG_FILENAME,
                                      '--output-location', str(Path(tmpdir).absolute())])

        assert 'error' not in result.output.lower()
        assert 'exception' not in result.output.lower()
        assert result.exit_code == 0

        assert outputfile.exists()


# takes ~2 mins to complete
@pytest.mark.xfail(not RUNNING_ON_NCI_ENV,
                   reason="This test currently expects to be run in the DEA environment on NCI.")
def test_input_region_from_shapefile_wet():
    runner = CliRunner()
    with runner.isolated_filesystem() as tmpdir:
        with open(CONFIG_FILENAME, 'w') as f:
            f.write(CONFIG_TEMPLATE_WET)

        outputfile = Path(tmpdir) / 'GW_WET_49_2007_2010.nc'

        result = runner.invoke(main, ['-v', '-v', '-v', CONFIG_FILENAME,
                                      '--output-location', str(Path(tmpdir).absolute())])

        assert 'error' not in result.output.lower()
        assert 'exception' not in result.output.lower()
        assert result.exit_code == 0

        assert outputfile.exists()


@pytest.mark.xfail
def test_input_region_from_geojson():
    assert False


@pytest.mark.xfail
def test_output_to_netcdf():
    assert False


@pytest.mark.xfail
def test_output_to_geotiff_single_band():
    assert False


@pytest.mark.xfail
def test_output_to_geotiff_multi_band():
    assert False


@pytest.mark.xfail
def test_output_to_geotiff_multi_file():
    assert False


@pytest.mark.xfail
def test_output_to_envibil():
    assert False
