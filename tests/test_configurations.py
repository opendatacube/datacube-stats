from pathlib import Path

import pytest
import yaml

from datacube_stats.schema import stats_schema

config = yaml.safe_load("""
    date_ranges:
        start_date: 2015-01-01
        end_date: 2015-04-01

    location:
    sources:
    -   product: fake_source_product
        measurements: [red]
    storage:
        driver: NetCDF CF
        tile_size:
            x: 100
            y: 100
        resolution:
            x: 100
            y: 100
        crs: EPSG:3577
    output_products:
    -   name: fake_output
        product_type: sample_statistics
        statistic: simple
        statistic_args:
            reduction_function: mean
""")


def test_sample_config(sample_stats_config):
    stats_schema(sample_stats_config)


configs_dir = Path(__file__).parent.parent / 'configurations'
stats_configs = configs_dir.rglob('*/*.yaml')


@pytest.mark.parametrize('stats_config', stats_configs, ids=lambda p: p.name)
def test_all_schemas(stats_config):
    with stats_config.open() as src:
        loaded_config = yaml.load(src)
        stats_schema(loaded_config)
