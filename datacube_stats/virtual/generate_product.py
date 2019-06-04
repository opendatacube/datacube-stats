"""
Create statistical summaries command.

This command is run as ``datacube-stats``, all operation are driven by a configuration file.

"""
import copy
import logging
import sys
import yaml

from functools import partial
from itertools import islice
from textwrap import dedent
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, Iterator, Tuple
from os import path

import click
import numpy as np
import pandas as pd
import pydash
import xarray
from dateutil import tz
from datacube import Datacube
from datacube.api import GridWorkflow
from datacube.storage.masking import make_mask
from datacube.ui import click as ui
from datacube.utils import read_documents, import_function
from datacube_stats.utils.dates import date_sequence
from .models import OutputProduct
from datacube.virtual import construct
from .output_drivers import OUTPUT_DRIVERS, OutputFileAlreadyExists, get_driver_by_name, \
    NoSuchOutputDriver
from .output_drivers import OutputDriver

_LOG = logging.getLogger(__name__)

DEFAULT_TILE_INDEX_FILE = 'landsat_tile_list.txt'
TILE_SIZE = 60000.


def u_dont_need_task_generator(dc, config):
    input_region = config['input_region']
    date_ranges = config['date_ranges']
    product_name = ''
    for product in config['output_products']:
        if product_name == product['name']:
            _LOG.error("More than one product with the same name %s", product_name)
        product_name = product['name']
        recipe = dict(product['recipe'])
        virtual_product = construct(**recipe)
        virtual_datasets_with_config = generate_virtual_datasets(dc, virtual_product, input_region, date_ranges)
        yield ProductWithDef(product, virtual_product, virtual_datasets_with_config)


TASK_GENERATOR_REG = {
        'IDNO': u_dont_need_task_generator
        }


def ls8_on(dataset):
    LS8_START_DATE = datetime(2013, 1, 1)
    return dataset.center_time >= LS8_START_DATE


def ls7_on(dataset):
    LS7_STOP_DATE = datetime(2003, 5, 31)
    LS7_STOP_AGAIN = datetime(2013, 5, 31)
    LS7_START_AGAIN = datetime(2010, 1, 1)
    return dataset.center_time <= LS7_STOP_DATE or (dataset.center_time >= LS7_START_AGAIN
                                                    and dataset.center_time <= LS7_STOP_AGAIN)


def ls5_on(dataset):
    LS5_START_AGAIN = datetime(2003, 1, 1)
    LS5_STOP_DATE = datetime(1998, 12, 31)
    LS5_STOP_AGAIN = datetime(2011, 12, 31)
    return dataset.center_time <= LS5_STOP_DATE or (dataset.center_time >= LS5_START_AGAIN
                                                    and dataset.center_time <= LS5_STOP_AGAIN)


# pylint: disable=broad-except
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
@click.command(name='generate-product')
@click.argument('stats_config_file', type=str, default='config.yaml', metavar='STATS_CONFIG_FILE')
@click.option('--task-generator',  type=str, help='Select a task generator or...not')
@click.option('--tile-index', nargs=2, type=int, help='Override input_region specified in configuration with a '
                                                      'single tile_index specified as [X] [Y]')
@click.option('--tile-index-file',
              type=click.Path(exists=True, readable=True, dir_okay=False),
              help="A file consisting of tile indexes specified as [X] [Y] per line")
@click.option('--output-location', help='Override output location in configuration file')
@click.option('--year', type=int, help='Override time period in configuration file')
@ui.global_cli_options
@ui.pass_datacube(app_name='generate-product')
def main(datacube, stats_config_file, task_generator, tile_index, tile_index_file, output_location, year):

    try:
        config = normalize_config(read_config(stats_config_file), output_location)
        config = normalize_time_range(config, year)
        config = normalize_space_range(config, tile_index, tile_index_file)
        if task_generator is None:
            task_generator = TASK_GENERATOR_REG.get('IDNO')
            from dask.distributed import LocalCluster, Client
            cluster = LocalCluster()
            client = Client(cluster)
            _LOG.warning('Run on your own with local cluster %s', cluster)
            products = task_generator(datacube, config)
            _LOG.debug("generate products")
            output_config = config.copy()
            output_config.pop('output_products', None)
            output_config.pop('input_region', None)
            output_config.pop('date_ranges', None)
            for product in products:
                metadata_type = retrieve_metadata_type(datacube)
                _LOG.debug("metadata type %s", metadata_type)
                output_config['output_product'] = product.product_definition
                for virtual_datasets_with_config in product.datasets:
                    output_config['input_region'] = virtual_datasets_with_config.input_region
                    output_config['date_range'] = virtual_datasets_with_config.date_range
                    app = StatsApp(output_config, product.product, virtual_datasets_with_config.datasets)
                    app.generate_products(metadata_type)

        else:
            task_generator = TASK_GENERATOR_REG.get(task_generator)

    except Exception as e:
        _LOG.error(e)
        sys.exit(1)

    return 0


def read_config(config_file):
    _, config = next(read_documents(config_file))
    return config


def normalize_time_range(config, year=None):

    if 'date_ranges' not in config:
        config['date_ranges'] = {}
        config['date_ranges']['start_date'] = '1987-01-01'
        config['date_ranges']['end_date'] = datetime.now().date()

    # if year is not none
    # then override the date_range
    # deal with the inconsistancy between date_sequence not inluding the end_date
    # and the datacube query including the end_date
    if year is not None:
        start_date = '{}-01-01'.format(year)
        end_date = '{}-01-01'.format(year+1)
        config['date_ranges'] = date_sequence(start=pd.to_datetime(start_date),
                                              end=pd.to_datetime(end_date),
                                              stats_duration='1y',
                                              step_size='1y')
        return config

    if 'stats_duration' in config['date_ranges'] and 'step_size' in config['date_ranges']:
        config['date_ranges'] = date_sequence(start=pd.to_datetime(config['date_ranges']['start_date']),
                                              end=pd.to_datetime(config['date_ranges']['end_date']),
                                              stats_duration=config['date_ranges']['stats_duration'],
                                              step_size=config['date_ranges']['step_size'])
    else:
        config['date_ranges'] = [(pd.to_datetime(config['date_ranges']['start_date']),
                                  pd.to_datetime(config['date_ranges']['end_date']))]
    return config


def normalize_space_range(config, tile_index=None, tile_index_file=None):
    if tile_index is not None and len(tile_index) == 0:
        tile_index = None

    # if tile indices or indices file is not none
    # then override the input_region
    tile_indexes = gather_tile_indexes(tile_index, tile_index_file)
    input_region = config.get('input_region')
    if input_region is None:
        input_region = {}
        if tile_indexes is None:
            tile_index_file = DEFAULT_TILE_INDEX_FILE
            tile_indexes = gather_tile_indexes(tile_index_file=tile_index_file)
        input_region['tiles'] = tile_indexes
    elif tile_indexes is not None:
        input_region['tiles'] = tile_indexes

    if tile_index_file is not None:
        if 'sentinel2' in tile_index_file:
            input_region['sensor_source'] = 'sentinel2'
        elif 'landsat' in tile_index_file:
            input_region['sensor_source'] = 'landsat'
    elif input_region.get('sensor_source') is None:
            _LOG.error('Need sensor source to calculate the coordinates for tile %s:'
                       'sentinel2 or landsat', tile_indexes)

    config['input_region'] = input_region
    return config


def normalize_config(config, output_location):
    # Write files to current directory if not set in config or command line
    config['location'] = output_location or config.get('location', '')
    config['computation'] = config.get('computation', {})
    config['global_attributes'] = config.get('global_attributes', {})
    config['var_attributes'] = config.get('var_attributes', {})
    return config


def gather_tile_indexes(tile_index, tile_index_file):
    if tile_index is None and tile_index_file is None:
        return None

    assert tile_index is None or tile_index_file is None, \
        "must not specify both tile_index and tile_index_file"

    if tile_index is not None:
        return [tile_index]

    with open(tile_index_file) as fl:
        tile_indexes = [tuple(int(x) for x in l.split()) for l in fl]
        if len(tile_indexes) == 0:
            return None
        return tile_indexes


def generate_virtual_datasets(dc, virtual_product, input_region, date_range):
    if 'tiles' not in input_region:
        _LOG.debug('tiles only')
    sensor_source = input_region.get('sensor_source')
    if sensor_source == 'sentile2':
        stride = TILE_SIZE
    elif sensor_source == 'landsat':
        stride = 2 * TILE_SIZE
    else:
        _LOG.error('Dont understand the sensor source %s', sensor_source)

    for tile in input_region['tiles']:
        for date in date_range:
            query_string = {}
            query_string['x'] = (tile[0] * stride, (tile[0] + 1) * stride)
            query_string['y'] = (tile[1] * stride, (tile[1] + 1) * stride)
            query_string['time'] = date
            # query by tile index implies crs
            query_string['crs'] = 'EPSG:3577'
            _LOG.debug("query string %s", query_string)
            datasets = virtual_product.query(dc, **query_string)
            grouped = virtual_product.group(datasets, **query_string)
            yield VirtualDatasetsWithConfig(grouped, tile, date)


class VirtualDatasetsWithConfig():
    def __init__(self, datasets, input_region, date_range):
        self.datasets = datasets
        self.input_region = input_region
        self.date_range = date_range


class ProductWithDef():
    def __init__(self, product_def: dict, virtual_product, virtual_datasets: VirtualDatasetsWithConfig):
        self.product_definition = product_def
        self.product = virtual_product
        self.datasets = virtual_datasets


class StatsApp:  # pylint: disable=too-many-instance-attributes
    """
    A StatsApp can produce a set of time based statistical products.
    """

    def __init__(self, config, virtual_product, virtual_datasets):
        """

        Create a StatsApp to run a processing job, based on a configuration dict.
        """
        #: Dictionary containing the configuration
        self.config = config

        #: Description of output file format
        self.storage = config['storage']

        #: List of filenames and statistical methods used, describing what the outputs of the run will be.
        self.output_product_spec = config['output_product']

        #: Base directory to write output files to.
        #: Files may be created in a sub-directory, depending on the configuration of the
        #: :attr:`output_driver`.
        self.location = config['location']

        #: A class which knows how to create and write out data to a permanent storage format.
        #: Implements :class:`.output_drivers.OutputDriver`.
        self.output_driver = _prepare_output_driver(self.storage)

        self.global_attributes = config['global_attributes']
        self.var_attributes = config['var_attributes']

        self.virtual_product = virtual_product
        self.output_product = virtual_datasets

    def _partially_applied_output_driver(self):
        app_info = _get_app_metadata(self.config)

        return partial(self.output_driver,
                       output_path=self.location,
                       app_info=app_info,
                       storage=self.storage,
                       global_attributes=self.global_attributes,
                       var_attributes=self.var_attributes)

    def generate_products(self, metadata_type):
        definition = self.output_product_spec

        extras = dict({'epoch_start': self.config['date_range'][0],
                       'epoch_end': self.config['date_range'][1],
                       'x': self.config['input_region'][0],
                       'y': self.config['input_region'][1]})

        if 'metadata' not in definition:
            definition['metadata'] = {}
        if 'format' not in definition['metadata']:
            definition['metadata']['format'] = {'name': self.output_driver.format_name()}

        output = OutputProduct.from_json_definition(metadata_type=metadata_type,
                                                    virtual_datasets=self.output_product,
                                                    virtual_product=self.virtual_product,
                                                    storage=self.storage,
                                                    definition=definition,
                                                    extras=extras)
        output_driver = self._partially_applied_output_driver()
        if self.config.get('computation') is not None:
            dask_chunks = self.config['computation'].get('chunking')
            _LOG.debug("dask chunks is %s", dask_chunks)
        with output_driver(output_product=output) as output_file:
            results = output.compute(output.datasets, dask_chunks=dask_chunks)
            _LOG.debug("finish dask graph, now loading...%s", results)
            try:
                results.load()
            except Exception as e:
                _LOG.error("some dask error I dont know %s", e)
            output_file.write_data(results)

    def __str__(self):
        return "StatsApp:  output_driver={}, output_products=({})".format(
            self.output_driver,
            self.output_product_spec
        )

    def __repr__(self):
        return str(self)


def retrieve_metadata_type(dc, metadata_type='eo'):
    """
    return metadata_type with the input string
    """
    return dc.index.metadata_types.get_by_name(metadata_type)


def _get_app_metadata(config_file):
    config = copy.deepcopy(config_file)
    if 'global_attributes' in config:
        del config['global_attributes']
    return {
        'lineage': {
            'algorithm': {
                'name': 'virtual-product',
                'parameters': {'configuration_file': config_file}
            },
        }
    }


def _prepare_output_driver(storage):
    try:
        return get_driver_by_name(storage['driver'])
    except NoSuchOutputDriver:
        if 'driver' in storage:
            msg = 'Invalid output driver "{}" specified.'
        else:
            msg = 'No output driver specified.'
        raise StatsConfigurationError('{} Specify one of {} in storage->driver in the '
                                      'configuration file.'.format(msg, list(OUTPUT_DRIVERS.keys())))


if __name__ == '__main__':
    main()
