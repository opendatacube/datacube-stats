from pathlib import Path

import pytest
from voluptuous import Schema, Required, All, Length, Date, ALLOW_EXTRA, Optional, Any, In
import yaml
from voluptuous.error import Invalid

from datacube_stats.statistics import STATS
import datetime

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


validate_source = Schema({
    Required('product'): str,
    Required('measurements'): [str],
    'group_by': str,
    'fuse_func': str
}, extra=ALLOW_EXTRA)


output_product_schema = Schema({
    Required('name'): str,
    'product_type': str,
    Required('statistic'): In(STATS.keys(), msg='not a valid statistic name'),
    'statistic_args': dict,
    'output_params': dict,
    'file_path_template': str
})


storage_schema = Schema({
    'driver': str,
    'tile_size': dict,
    'resolution': dict,
    'chunking': dict,
    'crs': str,
    'dimension_order': list
})

single_tile = Schema({'tile': [int, int]})
from_file = Schema({'from_file': str})
geometry = Schema({
    'geometry': {
        'type': 'Polygon',
        'coordinates': list}})
boundary_coords = Schema({
    'crs': str,
    'x': list,
    'y': list
})
date_ranges_schema = Schema({
    Required('start_date'): Any(datetime.date, Date),
    Required('end_date'): Any(datetime.date, Date),
    'stats_duration': str,
    'step_size': str
})

stats_schema = Schema({
    'date_ranges': date_ranges_schema,
    'location': str,
    'sources': All([validate_source], Length(min=1)),
    'storage': storage_schema,
    'output_products': All([output_product_schema], Length(min=1)),
    Optional('computation'): {'chunking': dict},
    Optional('input_region'): Any(single_tile, from_file, geometry, boundary_coords),
    Optional('global_attributes'): dict,
    Optional('var_attributes'): {str: {str: str}}
}, required=True)


def test_sample_config(sample_stats_config):
    stats_schema(sample_stats_config)


configs_dir = Path(__file__).parent.parent / 'configurations'
stats_configs = configs_dir.rglob('*/*.yaml')


@pytest.mark.parametrize('stats_config', stats_configs, ids=lambda p: p.name)
def test_all_schemas(stats_config):
    with stats_config.open() as src:
        loaded_config = yaml.load(src)
        stats_schema(loaded_config)

