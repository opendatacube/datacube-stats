from pathlib import Path
from string import Formatter

import pytest
from datacube_stats.output_drivers import OUTPUT_DRIVERS
from voluptuous import Schema, Required, All, Length, Date, ALLOW_EXTRA, Optional, Any, In, Invalid, Inclusive
import yaml

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

valid_filepath_fields = ['x', 'y', 'epoch_start', 'epoch_end', 'name', 'stat_name']


def valid_format_string(valid_fields):
    """
    Ensure that the provided string can be parsed as a python format string, and contains only `valid_fields`
    :param valid_fields: set or sequence of valid field names
    """
    f = Formatter()
    valid_fields = set(valid_fields)

    def validate_string(format_string):
        fields = set(field_name for _, field_name, _, _ in f.parse(format_string) if field_name)

        if fields < valid_fields:
            return format_string
        else:
            raise Invalid('format string specifies invalid field(s): %s' % (fields - valid_fields))

    return validate_string


source_schema = Schema({
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
    'file_path_template': All(str, valid_format_string(valid_filepath_fields)),
})

spatial_attrs = {Inclusive('x', 'proj'): Any(float, int),
                 Inclusive('y', 'proj'): Any(float, int),
                 Inclusive('latitude', 'geo'): Any(float, int),
                 Inclusive('longitude', 'geo'): Any(float, int)}

storage_schema = Schema({
    'driver': In(OUTPUT_DRIVERS.keys(), msg='not a valid output driver'),
    'tile_size': spatial_attrs,
    'resolution': spatial_attrs,
    'chunking': {**spatial_attrs, 'time': int},
    'crs': str,
    'dimension_order': list,
})

filter_prod_args = Schema({
    Optional('tide_range'): int,
    Optional('tide_percent'): int,
    Optional('type'): str,
    Optional('months'): [str, str],
    Optional('sub_class'): str
})

single_tile = Schema({'tile': [int, int]})
from_file = Schema({'from_file': str, Optional('feature_id'): [int]})
filter_product = Schema({
    Required('method'): str,
    Required('args'): filter_prod_args
})

geometry = Schema({
    'geometry': {
        'type': 'Polygon',
        'coordinates': list}})

boundary_coords = Schema({
    'crs': str,
    Inclusive('x', 'proj'): [Any(float, int), Any(float, int)],
    Inclusive('y', 'proj'): [Any(float, int), Any(float, int)],
    Inclusive('latitude', 'geo'): [Any(float, int), Any(float, int)],
    Inclusive('longitude', 'geo'): [Any(float, int), Any(float, int)],
})

date_ranges_schema = Schema({
    Required('start_date'): Any(datetime.date, Date),
    Required('end_date'): Any(datetime.date, Date),
    'stats_duration': str,
    'step_size': str,
    'type': Any('simple', 'find_daily_data'),
})

stats_schema = Schema({
    'date_ranges': date_ranges_schema,
    'location': str,
    'sources': All([source_schema], Length(min=1)),
    'storage': storage_schema,
    'output_products': All([output_product_schema], Length(min=1)),
    Optional('computation'): {'chunking': dict},
    Optional('input_region'): Any(single_tile, from_file, geometry, boundary_coords),
    Optional('global_attributes'): dict,
    Optional('var_attributes'): {str: {str: str}},
    Optional('filter_product'): filter_product
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
