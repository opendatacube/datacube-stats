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
from .models import OutputProduct
from datacube.virtual import construct
from .output_drivers import OUTPUT_DRIVERS, OutputFileAlreadyExists, get_driver_by_name, \
    NoSuchOutputDriver
from .output_drivers import OutputDriver, OutputDriverResult

_LOG = logging.getLogger(__name__)


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


# pylint: disable=broad-except
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
@click.command(name='generate-product')
@click.argument('stats_config_file', type=str, default='config.yaml', metavar='STATS_CONFIG_FILE')
@click.option('--tile-index', nargs=2, type=int, help='Override input_region specified in configuration with a '
                                                      'single tile_index specified as [X] [Y]')
@click.option('--tile-index-file',
              type=click.Path(exists=True, readable=True, dir_okay=False),
              help="A file consisting of tile indexes specified as [X] [Y] per line")
@click.option('--output-location', help='Override output location in configuration file')
@click.option('--year', type=int, help='Override time period in configuration file')
@ui.global_cli_options
@ui.pass_datacube(app_name='generate-product')
def main(datacube, stats_config_file, tile_index, tile_index_file, output_location, year):

    try:
        config = normalize_config(read_config(stats_config_file),
                                  tile_index, tile_index_file, year, output_location)

        app = StatsApp(config, datacube)
        app.generate_products()
    except Exception as e:
        _LOG.error(e)
        sys.exit(1)

    return 0


def read_config(config_file):
    _, config = next(read_documents(config_file))
    return config


def normalize_config(config, tile_index=None, tile_index_file=None,
                     year=None, output_location=None):
    if tile_index is not None and len(tile_index) == 0:
        tile_index = None

    tile_indexes = gather_tile_indexes(tile_index, tile_index_file)

    input_region = config.get('input_region')
    if tile_indexes is not None:
        if input_region is None:
            input_region = {'tiles': tile_indexes}
        elif 'geometry' in input_region:
            input_region.update({'tiles': tile_indexes})
        elif 'from_file' not in input_region:
            input_region = {'tiles': tile_indexes}

    config['input_region'] = input_region

    if year is not None:
        if 'date_ranges' not in config:
            config['date_ranges'] = {}

        config['date_ranges']['start_date'] = '{}-01-01'.format(year)
        config['date_ranges']['end_date'] = '{}-12-31'.format(year)

    # Write files to current directory if not set in config or command line
    config['location'] = output_location or config.get('location', '')

    config['computation'] = config.get('computation', {})
    config['global_attributes'] = config.get('global_attributes', {})
    config['var_attributes'] = config.get('var_attributes', {})

    return config


class StatsApp:  # pylint: disable=too-many-instance-attributes
    """
    A StatsApp can produce a set of time based statistical products.
    """

    def __init__(self, config, index=None):
        """
        Create a StatsApp to run a processing job, based on a configuration dict.
        """
        config = normalize_config(config)

        #: Dictionary containing the configuration
        self.config_file = config

        #: Description of output file format
        self.storage = config['storage']

        #: List of filenames and statistical methods used, describing what the outputs of the run will be.
        self.output_product_specs = config['output_products']

        #: Base directory to write output files to.
        #: Files may be created in a sub-directory, depending on the configuration of the
        #: :attr:`output_driver`.
        self.location = config['location']

        #: A class which knows how to create and write out data to a permanent storage format.
        #: Implements :class:`.output_drivers.OutputDriver`.
        self.output_driver = _prepare_output_driver(self.storage)

        self.global_attributes = config['global_attributes']
        self.var_attributes = config['var_attributes']

        self.validate()
        self.output_products = self.configure_outputs(index)

    def validate(self):
        """Check StatsApp is correctly configured and raise an error if errors are found."""
        self._ensure_unique_output_product_names()

        # assert callable(self.output_driver)
        # assert hasattr(self.output_driver, 'open_output_files')
        # assert hasattr(self.output_driver, 'write_data')

    def _ensure_unique_output_product_names(self):
        """Part of configuration validation"""
        output_names = [prod['name'] for prod in self.output_product_specs]
        duplicate_names = [x for x in output_names if output_names.count(x) > 1]
        if duplicate_names:
            raise StatsConfigurationError('Output products must all have different names. '
                                          'Duplicates found: %s' % duplicate_names)

    def log_config(self):
        config = self.config_file
        _LOG.debug('statistic: \'%s\' location: \'%s\'',
                   config['output_products'][0]['statistic'],
                   config['location'])

    def _partially_applied_output_driver(self):
        app_info = _get_app_metadata(self.config_file)

        return partial(self.output_driver,
                       output_path=self.location,
                       app_info=app_info,
                       storage=self.storage,
                       global_attributes=self.global_attributes,
                       var_attributes=self.var_attributes)

    def generate_virtual_datasets(self, dc, output_spec, metadata_type):
        input_region = self.config_file['input_region']
        definition = dict(output_spec)
        virtual_product = construct(**definition['recipe'])
        if 'metadata' not in definition:
            definition['metadata'] = {}
            if 'format' not in definition['metadata']:
                definition['metadata']['format'] = {'name': self.output_driver.format_name()}
        if 'tiles' not in input_region:
            _LOG.debug('tiles only')
        for tile in input_region['tiles']:
            query_string = {}
            query_string['x'] = (tile[0] * 100000, (tile[0] + 1) * 100000)
            query_string['y'] = (tile[1] * 100000, (tile[1] + 1) * 100000)
            query_string['time'] = (self.config_file['date_ranges']['start_date'],
                                    self.config_file['date_ranges']['end_date'])
            query_string['crs'] = definition['crs']
            _LOG.debug("query string %s", query_string)
            datasets = virtual_product.query(dc, **query_string)
            grouped = virtual_product.group(datasets, **query_string)
            start = pd.to_datetime(self.config_file['date_ranges']['start_date'])
            end = pd.to_datetime(self.config_file['date_ranges']['end_date'])

            extras = dict({'epoch_start': start,
                           'epoch_end': end,
                           'x': tile[0],
                           'y': tile[1]})
            yield OutputProduct.from_json_definition(
                                                    metadata_type=metadata_type,
                                                    virtual_datasets=grouped,
                                                    virtual_product=virtual_product,
                                                    storage=self.storage,
                                                    definition=definition,
                                                    extras=extras)

    def generate_products(self):
        output_driver = self._partially_applied_output_driver()
        for product in self.output_products.values():
            for output in product:
                with output_driver(output_product=output) as output_file:
                    results = output.compute(output.datasets)
                    output_file.write_data(results)

    def configure_outputs(self, dc, metadata_type='eo') -> Dict[str, OutputProduct]:
        """
        Return dict mapping Output Product Name<->Output Product

        StatProduct describes the structure and how to compute the output product.
        """

        output_products = {}
        metadata_type = dc.index.metadata_types.get_by_name(metadata_type)

        for output_spec in self.output_product_specs:
            output_products[output_spec['name']] = self.generate_virtual_datasets(
                dc=dc,
                output_spec=output_spec,
                metadata_type=metadata_type)

        # TODO: Write the output product to disk somewhere

        return output_products

    def __str__(self):
        return "StatsApp: sources=({}), output_driver={}, output_products=({})".format(
            ', '.join(source['product'] for source in self.sources),
            self.output_driver,
            ', '.join(out_spec['name'] for out_spec in self.output_product_specs)
        )

    def __repr__(self):
        return str(self)


class StatsProcessingException(Exception):
    pass


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
