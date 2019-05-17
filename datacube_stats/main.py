"""
Create statistical summaries command.

This command is run as ``datacube-stats``, all operation are driven by a configuration file.

"""
import copy
import logging
import sys

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
import rasterio.features
import xarray
from dateutil import tz
import datacube
import datacube_stats
from datacube.api import GridWorkflow
from datacube.storage.masking import make_mask
from datacube.ui import click as ui
from datacube.utils import read_documents, import_function
from datacube.utils.geometry import Geometry, GeoBox
from datacube_stats.models import OutputProduct
from datacube_stats.output_drivers import OUTPUT_DRIVERS, OutputFileAlreadyExists, get_driver_by_name, \
    NoSuchOutputDriver
from datacube_stats.statistics import StatsConfigurationError, STATS
from datacube_stats.utils import cast_back, pickle_stream, unpickle_stream, _find_periods_with_data
from datacube_stats.utils import tile_iter, sensible_mask_invalid_data, sensible_where, sensible_where_inplace
from datacube_stats.utils.dates import date_sequence
from .utils.timer import MultiTimer, wrap_in_timer
from .utils import sorted_interleave, Slice, prettier_slice
from .schema import stats_schema
from .output_drivers import OutputDriver, OutputDriverResult

__all__ = ['StatsApp', 'main']
_LOG = logging.getLogger(__name__)


def _default_config(ctx, param, value):
    if path.exists(value):
        return value

    ctx.fail('STATS_CONFIG_FILE not provided.')


def _print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return

    click.echo(
        '{prog}, version {version}'.format(
            prog='Data Cube',
            version=datacube.__version__
        )
    )

    click.echo(
        '{prog}, version {version}'.format(
            prog='Data Cube Statistics',
            version=datacube_stats.__version__
        )
    )
    ctx.exit()


def list_statistics(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return

    for name, stat in STATS.items():
        click.echo(name)

    ctx.exit()


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
@click.command(name='datacube-stats')
@click.argument('stats_config_file', type=str, callback=_default_config, default='config.yaml',
                metavar='STATS_CONFIG_FILE')
@click.option('--tile-index', nargs=2, type=int, help='Override input_region specified in configuration with a '
                                                      'single tile_index specified as [X] [Y]')
@click.option('--tile-index-file',
              type=click.Path(exists=True, readable=True, dir_okay=False),
              help="A file consisting of tile indexes specified as [X] [Y] per line")
@click.option('--output-location', help='Override output location in configuration file')
@click.option('--year', type=int, help='Override time period in configuration file')
@click.option('--list-statistics', is_flag=True, callback=list_statistics, expose_value=False)
@ui.global_cli_options
@click.option('--version', is_flag=True, callback=_print_version,
              expose_value=False, is_eager=True)
@ui.pass_index(app_name='datacube-stats')
def main(index, stats_config_file, runner, save_tasks, load_tasks,
         tile_index, tile_index_file, output_location, year, task_slice, batch):

    try:
        _log_setup()

        timer = MultiTimer().start('main')

        config = normalize_config(read_config(stats_config_file),
                                  tile_index, tile_index_file, year, output_location)

        app = StatsApp(config, index)
        app.log_config()
        successful, failed = app.run_tasks(tasks, runner, task_slice)

        timer.pause('main')
        _LOG.info('Stats processing completed in %s seconds.', timer.run_times['main'])

        if failed > 0:
            raise click.ClickException('%s of %s tasks not completed successfully.' % (failed, successful + failed))

    except Exception as e:
        _LOG.error(e)
        sys.exit(1)

    return 0


def _log_setup():
    _LOG.debug('Loaded datacube_stats %s from %s.', datacube_stats.__version__, datacube_stats.__path__)
    _LOG.debug('Running against datacube-core %s from %s', datacube.__version__, datacube.__path__)


def read_config(stats_config_file):
    _, config = next(read_documents(stats_config_file))
    stats_schema(config)
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
        config['date_ranges']['end_date'] = '{}-01-01'.format(year + 1)

    # Write files to current directory if not set in config or command line
    config['location'] = output_location or config.get('location', '')

    config['computation'] = config.get('computation', {})
    config['filter_product'] = config.get('filter_product', {})

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

        #: Definition of source products, including their name, which variables to pull from them, and
        #: a specification of any masking that should be applied.
        self.sources = config['sources']

        #: List of filenames and statistical methods used, describing what the outputs of the run will be.
        self.output_product_specs = config['output_products']

        #: Base directory to write output files to.
        #: Files may be created in a sub-directory, depending on the configuration of the
        #: :attr:`output_driver`.
        self.location = config['location']

        #: How to slice a task up spatially to to fit into memory.
        self.computation = config['computation']

        #: Define filter product to accept all derive product attributes
        self.filter_product = config['filter_product']

        #: An iterable of date ranges.
        self.date_ranges = _configure_date_ranges(config, index=index)

        #: A class which knows how to create and write out data to a permanent storage format.
        #: Implements :class:`.output_drivers.OutputDriver`.
        self.output_driver = _prepare_output_driver(self.storage)

        self.global_attributes = config['global_attributes']
        self.var_attributes = config['var_attributes']

        self.validate()

    def validate(self):
        """Check StatsApp is correctly configured and raise an error if errors are found."""
        self._ensure_unique_output_product_names()
        self._check_consistent_measurements()

        assert callable(self.output_driver)
        assert hasattr(self.output_driver, 'open_output_files')
        assert hasattr(self.output_driver, 'write_data')

        _LOG.debug('config file is valid.')

    def _check_consistent_measurements(self):
        """Part of configuration validation"""
        try:
            first_source = self.sources[0]
        except IndexError:
            raise StatsConfigurationError('No data sources specified.')
        if not all(first_source.get('measurements') == source.get('measurements') for source in self.sources):
            raise StatsConfigurationError("Configuration Error: listed measurements of source products "
                                          "are not all the same.")

    def _ensure_unique_output_product_names(self):
        """Part of configuration validation"""
        output_names = [prod['name'] for prod in self.output_product_specs]
        duplicate_names = [x for x in output_names if output_names.count(x) > 1]
        if duplicate_names:
            raise StatsConfigurationError('Output products must all have different names. '
                                          'Duplicates found: %s' % duplicate_names)

    def log_config(self):
        config = self.config_file
        _LOG.info('statistic: \'%s\' location: \'%s\'',
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

    def execute_task(self, task):
        """
        Execute an individual task locally.
        Intended to be used interactively rather than on a cluster.
        """
        try:
            execute_task(task,
                         output_driver=self._partially_applied_output_driver(),
                         chunking=self.computation.get('chunking', {}))

            _LOG.debug('task %s finished', task)
        except OutputDriverResult as e:
            return e

    def run_tasks(self, tasks, runner=None, task_slice=None):
        from digitalearthau.qsub import TaskRunner
        from digitalearthau.runners.model import TaskDescription, DefaultJobParameters

        if task_slice is not None:
            tasks = islice(tasks, task_slice.start, task_slice.stop, task_slice.step)

        output_driver = self._partially_applied_output_driver()
        task_runner = partial(execute_task,
                              output_driver=output_driver,
                              chunking=self.computation.get('chunking', {}))

        # does not need to be thorough for now
        task_desc = TaskDescription(type_='datacube_stats',
                                    task_dt=datetime.utcnow().replace(tzinfo=tz.tzutc()),
                                    events_path=Path(self.location) / 'events',
                                    logs_path=Path(self.location) / 'logs',
                                    jobs_path=Path(self.location) / 'jobs',
                                    parameters=DefaultJobParameters(query={},
                                                                    source_products=[],
                                                                    output_products=[]))

        task_desc.logs_path.mkdir(parents=True, exist_ok=True)
        task_desc.events_path.mkdir(parents=True, exist_ok=True)
        task_desc.jobs_path.mkdir(parents=True, exist_ok=True)

        if runner is None:
            runner = TaskRunner()

        result = runner(task_desc, tasks, task_runner)

        _LOG.debug('Stopping runner.')
        runner.stop()
        _LOG.debug('Runner stopped.')

        return result

    def configure_outputs(self, index, metadata_type='eo') -> Dict[str, OutputProduct]:
        """
        Return dict mapping Output Product Name<->Output Product

        StatProduct describes the structure and how to compute the output product.
        """
        _LOG.debug('Creating output products')

        output_products = {}

        measurements = _source_measurement_defs(index, self.sources)

        metadata_type = index.metadata_types.get_by_name(metadata_type)

        stats_metadata = _get_stats_metadata(self.config_file)

        for output_spec in self.output_product_specs:
            definition = dict(output_spec)
            if 'metadata' not in definition:
                definition['metadata'] = {}
                if 'format' not in definition['metadata']:
                    definition['metadata']['format'] = {'name': self.output_driver.format_name()}

            output_products[output_spec['name']] = OutputProduct.from_json_definition(
                metadata_type=metadata_type,
                input_measurements=measurements,
                storage=self.storage,
                definition=definition,
                stats_metadata=stats_metadata)

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


class EmptyChunkException(Exception):
    pass


def _remove_emptys(datasets):
    return [dataset
            for dataset in datasets
            if dataset is not None]


def _source_measurement_defs(index, sources):
    """

    Look up desired measurements from sources in the database index. Note that
    multiple sources are meant to be of the same shape, we only support
    combining equivalent products from different sensors.

    :return: list of measurement definitions
    """
    # Check all source measurements are equal
    first_source = sources[0]

    # Ensure specified sources match
    for other_source in sources[1:]:
        if other_source.get('measurements') != first_source.get('measurements'):
            raise StatsConfigurationError('Measurements in configured sources do not match. To combine sources'
                                          'they must all be identical. %s measurements are %s while %s measurements '
                                          'are %s' % (first_source['product'], first_source['measurements'],
                                                      other_source['product'], other_source['measurements']))

    # TODO: should probably check that all products exist and are of compatible shape

    available_measurements = index.products.get_by_name(first_source['product']).measurements
    requested_measurements = first_source.get('measurements', available_measurements.keys())

    try:
        return [available_measurements[name] for name in requested_measurements]
    except KeyError:
        raise StatsConfigurationError('Some of the requested measurements were not present in the product definition')


def _get_app_metadata(config_file):
    config = copy.deepcopy(config_file)
    if 'global_attributes' in config:
        del config['global_attributes']
    return {
        'lineage': {
            'algorithm': {
                'name': 'datacube-stats',
                'version': datacube_stats.__version__,
                'repo_url': 'https://github.com/GeoscienceAustralia/datacube-stats.git',
                'parameters': {'configuration_file': config_file}
            },
        }
    }


def _get_stats_metadata(cfg):
    """ Build metadata.stats subtree for the product definition
    """
    period = pydash.get(cfg, 'date_ranges.stats_duration', '*')
    step = pydash.get(cfg, 'date_ranges.step_size', '*')
    return dict(period=period, step=step)


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


def _configure_date_ranges(config, index=None):
    if 'date_ranges' not in config:
        raise StatsConfigurationError(dedent("""\
        No Date Range specification was found in the stats configuration file, please add a section similar to:

        date_ranges:
          start_date: 2010-01-01
          end_date: 2011-01-01
          stats_duration: 3m
          step_size: 3m

        This will produce 4 x quarterly statistics from the year 2010.
        """))
    date_ranges = config['date_ranges']
    if 'start_date' not in date_ranges or 'end_date' not in date_ranges:
        raise StatsConfigurationError("Must specified both `start_date` and `end_date`"
                                      " in `date_ranges:` section of configuration")

    if 'stats_duration' not in date_ranges and 'step_size' not in date_ranges:
        start = pd.to_datetime(date_ranges['start_date'])
        end = pd.to_datetime(date_ranges['end_date'])
        output = [(start, end)]

    elif date_ranges.get('type', 'simple') == 'simple':
        output = list(date_sequence(start=pd.to_datetime(date_ranges['start_date']),
                                    end=pd.to_datetime(date_ranges['end_date']),
                                    stats_duration=date_ranges['stats_duration'],
                                    step_size=date_ranges['step_size']))

    elif date_ranges.get('type') == 'find_daily_data':
        if index is None:
            raise ValueError('find_daily_data needs a datacube index to be passed')

        sources = config['sources']
        product_names = [source['product'] for source in sources]
        output = list(_find_periods_with_data(index, product_names=product_names,
                                              start_date=date_ranges['start_date'],
                                              end_date=date_ranges['end_date']))
    else:
        raise StatsConfigurationError('Unknown date_ranges specification. Should be type=simple or '
                                      'type=find_daily_data')
    _LOG.debug("Selecting data for date ranges: %s", output)

    if not output:
        raise StatsConfigurationError('Time period configuration results in 0 periods of interest.')
    return output


if __name__ == '__main__':
    main()
