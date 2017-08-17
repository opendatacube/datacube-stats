from pathlib import Path
import pytest

from click.testing import CliRunner
from datacube_stats.main import main


CONFIG_TEMPLATE = """
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
  driver: NetCDFCF

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
   tile: [12, -43]

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

CONFIG_FILENAME = 'config.yaml'


def sample_geometry():
    from datacube.utils import geometry
    from affine import Affine
    gb = geometry.GeoBox(40, 40, Affine(2500, 0.0, 1200000.0, 0.0, -2500, -4300000.0), geometry.CRS('EPSG:3577'))
    json = gb.extent.json


def test_input_region_single_tile():
    runner = CliRunner()
    with runner.isolated_filesystem() as tmpdir:
        with open(CONFIG_FILENAME, 'w') as f:
            f.write(CONFIG_TEMPLATE)

        result = runner.invoke(main, ['-v', '-v', '-v', CONFIG_FILENAME])
        assert 'error' not in result.output.lower()
        assert 'exception' not in result.output.lower()
        assert result.exit_code == 0

        tmpdir = Path(tmpdir)
        outputfile = tmpdir / 'SR_N_MEAN' / 'SR_N_MEAN_3577_12_-43_20150101.nc'

        assert outputfile.exists()


@pytest.mark.xfail
def test_input_region_from_shapefile():
    assert False


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
