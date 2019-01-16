from string import Formatter
import datetime

import pandas as pd
from voluptuous import Schema, Required, All, Length, Date, ALLOW_EXTRA, Optional, Any, In, Invalid, Inclusive

from .statistics import STATS
from .output_drivers import OUTPUT_DRIVERS

# pylint: disable=invalid-name

valid_filepath_fields = ['x', 'y', 'feature_id', 'epoch_start', 'epoch_end', 'name', 'stat_name', 'var_name']


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

available_stats_msg = ('not a valid statistic name (available statistics are {})'
                       .format(", ".join(list(STATS.keys()))))

output_product_schema = Schema({
    Required('name'): str,
    'product_type': str,
    Required('statistic'): In(list(STATS.keys()), msg=available_stats_msg),
    'statistic_args': dict,
    'output_params': dict,
    'file_path_template': All(str, valid_format_string(valid_filepath_fields)),
    'metadata': dict
})

spatial_attrs = {Inclusive('x', 'proj'): Any(float, int),
                 Inclusive('y', 'proj'): Any(float, int),
                 Inclusive('latitude', 'geo'): Any(float, int),
                 Inclusive('longitude', 'geo'): Any(float, int)}

available_drivers_msg = ('not a valid output driver (available drivers are {})'
                         .format(", ".join(list(OUTPUT_DRIVERS.keys()))))

storage_schema = Schema({
    'driver': In(list(OUTPUT_DRIVERS.keys()), msg=available_drivers_msg),
    'tile_size': spatial_attrs,
    'resolution': spatial_attrs,
    'chunking': {**spatial_attrs, 'time': int},
    'crs': str,
    'dimension_order': list,
})

filter_prod_args = Schema({
    'tide_range': int,
    'tide_percent': int,
    'type': str,
    'months': [str, str],
    'sub_class': str
})

single_tile = Schema({'tile': [int, int]})
tile_list = Schema({'tiles': list})
from_file = Schema({'from_file': str, Optional('feature_id'): [int], Optional('gridded'): bool})
filter_product = Schema({
    Required('method'): str,
    Required('args'): filter_prod_args
})

geometry = Schema({
    'geometry': {
        'type': 'Polygon',
        'coordinates': list
    }
})

boundary_coords = Schema({
    'crs': str,
    Inclusive('x', 'proj'): [Any(float, int), Any(float, int)],
    Inclusive('y', 'proj'): [Any(float, int), Any(float, int)],
    Inclusive('latitude', 'geo'): [Any(float, int), Any(float, int)],
    Inclusive('longitude', 'geo'): [Any(float, int), Any(float, int)],
})

date_ranges_schema = Schema({
    Required('start_date'): Any(datetime.date, Date, pd.to_datetime),
    Required('end_date'): Any(datetime.date, Date, pd.to_datetime),
    'stats_duration': str,
    'step_size': str,
    'type': Any('simple', 'find_daily_data'),
})

computation_schema = Schema({
    Inclusive('x', 'proj'): Any(float, int),
    Inclusive('y', 'proj'): Any(float, int),
    Inclusive('latitude', 'geo'): Any(float, int),
    Inclusive('longitude', 'geo'): Any(float, int)
})

stats_schema = Schema({
    'date_ranges': date_ranges_schema,
    'location': str,
    'sources': All([source_schema], Length(min=1)),
    'storage': storage_schema,
    'output_products': All([output_product_schema], Length(min=1)),
    Optional('computation'): {'chunking': computation_schema},
    Optional('input_region'): Any(single_tile, tile_list, from_file, geometry, boundary_coords),
    Optional('global_attributes'): dict,
    Optional('var_attributes'): {str: {str: str}},
    Optional('filter_product'): filter_product
}, required=True)
