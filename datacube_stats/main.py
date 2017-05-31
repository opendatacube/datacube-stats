"""
Create statistical summaries command.

This command is run as ``datacube-stats``, all operation are driven by a configuration file.

"""
from __future__ import absolute_import, print_function

import logging
from functools import partial
from textwrap import dedent

from utils import sensible_mask_invalid_data, sensible_where

try:
    import cPickle as pickle
except ImportError:
    import pickle

import click
import cloudpickle
import numpy as np
import pandas as pd
import xarray
import yaml

import datacube_stats
import datacube
from datacube import Datacube
from datacube.api import make_mask, GridWorkflow, Tile
from datacube.api.query import query_group_by, query_geopolygon, Query
from datacube.model import GridSpec
from datacube.utils.geometry import CRS, GeoBox, Geometry
from datacube.ui import click as ui
from datacube.ui.click import to_pathlib
from datacube.utils import read_documents, import_function
from datacube.utils.dates import date_sequence
from datacube_stats.models import StatsTask, OutputProduct
from datacube_stats.output_drivers import OUTPUT_DRIVERS, OutputFileAlreadyExists
from datacube_stats.runner import run_tasks
from datacube_stats.statistics import StatsConfigurationError, STATS
from datacube_stats.timer import MultiTimer
from datacube_stats.utils import tile_iter

__all__ = ['StatsApp', 'main']
_LOG = logging.getLogger(__name__)
DEFAULT_GROUP_BY = 'time'
DEFAULT_COMPUTATION_OPTIONS = {'chunking': {'x': 1000, 'y': 1000}}


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

#       click.echo("{:>20} | {}".format(name, inspect.getdoc(stat)))
    for name, stat in STATS.items():
        click.echo(name)
        # click.echo(inspect.getdoc(stat))
        # click.echo('\n\n')
#     click.echo(pd.Series({name: inspect.getdoc(stat) for name, stat in STATS.items()}))

    ctx.exit()


@click.command(name='datacube-stats')
@click.argument('stats_config_file',
                type=click.Path(exists=True, readable=True, writable=False, dir_okay=False),
                callback=to_pathlib)
@click.option('--queue-size', type=click.IntRange(1, 100000), default=2000,
              help='Number of tasks to queue at the start')
@click.option('--save-tasks', type=click.Path(exists=False, writable=True, dir_okay=False))
@click.option('--load-tasks', type=click.Path(exists=True, readable=True))
@click.option('--tile-index', nargs=2, type=int, help='Override input_region specified in configuration with a '
                                                      'single tile_index specified as [X] [Y]')
@click.option('--output-location', help='Override output location in configuration file')
@click.option('--list-statistics', is_flag=True, callback=list_statistics, expose_value=False)
@ui.global_cli_options
@ui.executor_cli_options
@click.option('--version', is_flag=True, callback=_print_version,
              expose_value=False, is_eager=True)
@ui.pass_index(app_name='datacube-stats')
def main(index, stats_config_file, executor, queue_size, save_tasks, load_tasks, tile_index, output_location):
    _log_setup()

    timer = MultiTimer().start('main')

    _, config = next(read_documents(stats_config_file))
    app = create_stats_app(config, index, tile_index, output_location)
    app.queue_size = queue_size
    app.validate()

    if save_tasks:
        app.save_tasks_to_file(save_tasks)
        failed = 0
    elif load_tasks:
        successful, failed = app.run(executor, load_tasks)
    else:
        successful, failed = app.run(executor)

    timer.pause('main')
    _LOG.info('Stats processing completed in %s seconds.', timer.run_times['main'])

    if failed > 0:
        raise click.ClickException('%s of %s tasks were not completed successfully.' % (failed, successful))


def _log_setup():
    _LOG.debug('Loaded datacube_stats %s from %s.', datacube_stats.__version__, datacube_stats.__path__)
    _LOG.debug('Running against datacube-core %s from %s', datacube.__version__, datacube.__path__)


class StatsApp(object):
    """
    A StatsApp can produce a set of time based statistical products.
    """

    def __init__(self):
        #: Name of the configuration file used
        self.config_file = None

        #: Description of output file format
        self.storage = None

        #: Definition of source products, including their name, which variables to pull from them, and
        #: a specification of any masking that should be applied.
        self.sources = []

        #: List of filenames and statistical methods used, describing what the outputs of the run will be.
        self.output_product_specs = []

        #: Base directory to write output files to.
        #: Files may be created in a sub-directory, depending on the configuration of the
        #: :attr:`output_driver`.
        self.location = None

        #: How to slice a task up spatially to to fit into memory.
        self.computation = None

        #: An iterable of date ranges.
        self.date_ranges = None

        #: Generates tasks to compute statistics. These tasks should be :class:`StatsTask` objects
        #: and will define spatial and temporal boundaries, as well as statistical operations to be run.
        self.task_generator = None

        #: A class which knows how to create and write out data to a permanent storage format.
        #: Implements :class:`.output_drivers.OutputDriver`.
        self.output_driver = None

        #: An open database connection
        self.index = None

        #: A function to process the result of a complated task
        #: Takes a single argument of the task result
        self.process_completed = None

        self.queue_size = 50

    def validate(self):
        """Check StatsApp is correctly configured and raise an error if errors are found."""
        self._ensure_unique_output_product_names()
        self._ensure_stats_available()
        self._check_consistent_measurements()

        assert callable(self.task_generator)
        assert callable(self.output_driver)
        assert hasattr(self.output_driver, 'open_output_files')
        assert hasattr(self.output_driver, 'write_data')
        assert callable(self.process_completed)

    def _check_consistent_measurements(self):
        """Part of configuration validation"""
        try:
            first_source = self.sources[0]
        except IndexError:
            raise StatsConfigurationError('No data sources specified.')
        if not all(first_source.get('measurements') == source.get('measurements') for source in self.sources):
            raise StatsConfigurationError("Configuration Error: listed measurements of source products "
                                          "are not all the same.")

    def _ensure_stats_available(self):
        """Part of configuration validation"""
        for prod in self.output_product_specs:
            if 'statistic' not in prod:
                raise StatsConfigurationError('Invalid statistic definition %s, must specify which statistic to use. '
                                              'eg. statistic: mean' % yaml.dump(prod, indent=4,
                                                                                default_flow_style=False))
        available_statistics = set(STATS.keys())
        requested_statistics = set(prod['statistic'] for prod in self.output_product_specs)
        if not requested_statistics.issubset(available_statistics):
            raise StatsConfigurationError(
                'Requested Statistic(s) %s are not valid statistics. Available statistics are: %s'
                % (requested_statistics - available_statistics, available_statistics))

    def _ensure_unique_output_product_names(self):
        """Part of configuration validation"""
        output_names = [prod['name'] for prod in self.output_product_specs]
        duplicate_names = [x for x in output_names if output_names.count(x) > 1]
        if duplicate_names:
            raise StatsConfigurationError('Output products must all have different names. '
                                          'Duplicates found: %s' % duplicate_names)

    def run(self, executor, task_file=None):
        if task_file:
            tasks = unpickle_stream(task_file)
        else:
            tasks = self.generate_tasks(self.configure_outputs())

        app_info = _get_app_metadata(self.config_file)
        output_driver = partial(self.output_driver,
                                output_path=self.location,
                                app_info=app_info,
                                storage=self.storage,
                                global_attributes=self.global_attributes)
        task_runner = partial(execute_task,
                              output_driver=output_driver,
                              chunking=self.computation.get('chunking', {}))
        return run_tasks(tasks,
                         executor,
                         task_runner,
                         self.process_completed,
                         queue_size=self.queue_size)

    def save_tasks_to_file(self, filename):
        _LOG.debug('Saving tasks to %s.', filename)
        output_products = self.configure_outputs()

        tasks = self.generate_tasks(output_products)
        num_saved = pickle_stream(tasks, filename)
        _LOG.debug('Successfully saved %s tasks to %s.', num_saved, filename)

    def generate_tasks(self, output_products):
        """
        Generate a sequence of `StatsTask` definitions.

        A Task Definition contains:

          * tile_index
          * time_period
          * sources: (list of)
          * output_products

        Sources is a list of dictionaries containing:

          * data
          * masks (list of)
          * spec - Source specification, containing details about which bands to load and how to apply masks.

        :param output_products: List of output product definitions
        :return:
        """
        for task in self.task_generator(index=self.index, date_ranges=self.date_ranges,
                                        sources_spec=self.sources):
            task.output_products = output_products
            yield task

    def configure_outputs(self, metadata_type='eo'):
        """
        Return dict mapping Output Product Name<->Output Product

        StatProduct describes the structure and how to compute the output product.
        """
        _LOG.info('Creating output products')

        output_products = {}

        measurements = _source_measurement_defs(self.index, self.sources)

        metadata_type = self.index.metadata_types.get_by_name(metadata_type)
        for output_spec in self.output_product_specs:
            output_products[output_spec['name']] = OutputProduct.from_json_definition(
                metadata_type=metadata_type,
                input_measurements=measurements,
                storage=self.storage,
                definition=output_spec)

        # TODO: Create the output product in the database

        return output_products


class StatsProcessingException(Exception):
    pass


def execute_task(task, output_driver, chunking):
    """
    Load data, run the statistical operations and write results out to the filesystem.

    :param datacube_stats.models.StatsTask task:
    :type output_driver: OutputDriver
    :param chunking: dict of dimension sizes to chunk the computation by
    """
    timer = MultiTimer().start('total')
    datacube.set_options(reproject_threads=1)

    try:
        with output_driver(task=task) as output_files:
            for sub_tile_slice in tile_iter(task.sample_tile, chunking):
                load_process_save_chunk(output_files, sub_tile_slice, task, timer)
    except OutputFileAlreadyExists as e:
        _LOG.warning(e)
    except Exception as e:
        _LOG.error("Error processing task: %s", task)
        raise StatsProcessingException("Error processing task: %s" % task)

    timer.pause('total')
    _LOG.info('Completed %s %s task with %s data sources. Processing took: %s', task.tile_index,
              [d.strftime('%Y-%m-%d') for d in task.time_period], task.data_sources_length(), timer)
    return task


def load_process_save_chunk(output_files, chunk, task, timer):
    try:
        timer.start('loading_data')
        data = _load_data(chunk, task.sources)
        timer.pause('loading_data')

        for prod_name, stat in task.output_products.items():
            _LOG.info("Computing %s in tile %s %s. Current timing: %s",
                      prod_name, task.tile_index, chunk, timer)
            timer.start(prod_name)
            data = stat.compute(data)
            timer.pause(prod_name)

            # For each of the data variables, shove this chunk into the output results
            timer.start('writing_data')
            for var_name, var in data.data_vars.items():  # TODO: Move this loop into output_files
                output_files.write_data(prod_name, var_name, chunk, var.values)
            timer.pause('writing_data')
    except EmptyChunkException:
        _LOG.debug('Error: No data returned while loading %s for %s. May have all been masked',
                   chunk, task)


class EmptyChunkException(Exception):
    pass


def _load_data(sub_tile_slice, sources):
    """
    Load a masked chunk of data from the datacube, based on a specification and list of datasets in `sources`.

    :param sub_tile_slice: A portion of a tile, tuple coordinates
    :param sources: a dictionary containing `data`, `spec` and `masks`
    :return: :class:`xarray.Dataset` containing loaded data. Will be indexed and sorted by time.
    """
    datasets = [_load_masked_data(sub_tile_slice, source_prod)
                for source_prod in sources]  # list of datasets
    datasets = _mark_source_idx(datasets)
    datasets = _remove_emptys(datasets)
    if len(datasets) == 0:
        raise EmptyChunkException()

    datasets = xarray.concat(datasets, dim='time')  # Copies all the data
    if len(datasets.time) == 0:
        raise EmptyChunkException()

    # sort along time dim
    return datasets.isel(time=datasets.time.argsort())  # Copies all the data again


def _mark_source_idx(datasets):
    for idx, dataset in enumerate(datasets):
        if dataset is not None:
            dataset.coords['source'] = ('time', np.repeat(idx, dataset.time.size))
    return datasets


def _remove_emptys(datasets):
    return [dataset
            for dataset in datasets
            if dataset is not None]


def _load_masked_data(sub_tile_slice, source_prod):
    data = GridWorkflow.load(source_prod['data'][sub_tile_slice],
                             measurements=source_prod['spec'].get('measurements'),
                             skip_broken_datasets=True)

    mask_nodata = source_prod['spec'].get('mask_nodata', True)
    if mask_nodata:
        data = sensible_mask_invalid_data(data)

    # if all NaN
    completely_empty = all(ds for ds in xarray.ufuncs.isnan(data).all().data_vars.values())
    if completely_empty:
        # Discard empty slice
        return None

    if 'masks' in source_prod and 'masks' in source_prod['spec']:
        for mask_spec, mask_tile in zip(source_prod['spec']['masks'], source_prod['masks']):
            if mask_tile is None:
                # Discard data due to no mask data
                return None
            fuse_func = import_function(mask_spec['fuse_func']) if 'fuse_func' in mask_spec else None
            mask = GridWorkflow.load(mask_tile[sub_tile_slice],
                                     measurements=[mask_spec['measurement']],
                                     fuse_func=fuse_func,
                                     skip_broken_datasets=True)[mask_spec['measurement']]
            mask = make_mask(mask, **mask_spec['flags'])
            data = sensible_where(data, mask)
            del mask
    return data


def _source_measurement_defs(index, sources):
    """
    Look up desired measurements from sources in the database index

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

    source_measurements = index.products.get_by_name(first_source['product']).measurements

    try:
        measurements = [source_measurements[name] for name in first_source['measurements']]
    except KeyError:
        measurements = source_measurements

    return measurements


def _get_app_metadata(config_file):
    return {
        'lineage': {
            'algorithm': {
                'name': 'datacube-stats',
                'version': datacube_stats.__version__,
                'repo_url': 'https://github.com/GeoscienceAustralia/agdc_statistics.git',
                'parameters': {'configuration_file': config_file}
            },
        }
    }


def _find_periods_with_data(index, product_names, period_duration='1 day',
                            start_date='1985-01-01', end_date='2000-01-01'):
    # TODO: Read 'simple' job configuration from file
    query = dict(y=(-3760000, -3820000), x=(1375400.0, 1480600.0), crs='EPSG:3577', time=(start_date, end_date))

    valid_dates = set()
    for product in product_names:
        counts = index.datasets.count_product_through_time(period_duration, product=product,
                                                           **Query(**query).search_terms)
        valid_dates.update(time_range for time_range, count in counts if count > 0)

    for time_range in sorted(valid_dates):
        yield time_range.begin, time_range.end


def create_stats_app(config, index=None, tile_index=None, output_location=None):
    """
    Create a StatsApp to run a processing job, based on a configuration file

    :param config: dictionary based configuration
    :param index: open database connection
    :param tile_index: Only process a single tile of a gridded job. (useful for debugging)
    :return: read to run StatsApp
    """
    input_region = config.get('input_region')
    if tile_index and not input_region:
        input_region = {'tile': tile_index}

    stats_app = StatsApp()
    stats_app.index = index
    stats_app.config_file = config
    stats_app.storage = config['storage']
    stats_app.sources = config['sources']
    stats_app.output_product_specs = config['output_products']
    stats_app.location = config.get('location',
                                    output_location)  # Write files to current directory if not set in config
    stats_app.computation = config.get('computation', {})
    stats_app.date_ranges = _configure_date_ranges(index, config)
    stats_app.task_generator = _create_task_generator(input_region, stats_app.storage)
    stats_app.output_driver = _prepare_output_driver(stats_app.storage)
    stats_app.global_attributes = config.get('global_attributes', {})
    stats_app.process_completed = do_nothing  # TODO: Save dataset to database

    return stats_app


def _prepare_output_driver(storage):
    try:
        return OUTPUT_DRIVERS[storage['driver']]
    except KeyError:
        if 'driver' in storage:
            msg = 'Invalid output driver "{}" specified.'
        else:
            msg = 'No output driver specified.'
        raise StatsConfigurationError('{} Specify one of {} in storage->driver in the '
                                      'configuration file.'.format(msg, OUTPUT_DRIVERS.keys()))


def _configure_date_ranges(index, config):
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
    date_ranges, sources = config['date_ranges'], config['sources']

    if date_ranges.get('type', 'simple') == 'simple':
        return list(date_sequence(start=pd.to_datetime(date_ranges['start_date']),
                                  end=pd.to_datetime(date_ranges['end_date']),
                                  stats_duration=date_ranges['stats_duration'],
                                  step_size=date_ranges['step_size']))
    elif date_ranges['type'] == 'find_daily_data':
        product_names = [source['product'] for source in sources]
        return list(_find_periods_with_data(index, product_names=product_names,
                                            start_date=date_ranges['start_date'],
                                            end_date=date_ranges['end_date']))
    else:
        raise StatsConfigurationError('Unknown date_ranges specification. Should be type=simple or '
                                      'type=find_daily_data')


def _create_task_generator(input_region, storage):
    if input_region is None:
        _LOG.info('No input_region specified. Generating full available spatial region, gridded files.')
        return GriddedTaskGenerator(storage)

    elif 'tile' in input_region:  # Simple, single tile
        return GriddedTaskGenerator(storage, cell_index=input_region['tile'])

    elif 'geometry' in input_region:  # Larger spatial region
        # A large, multi-tile input region, specified as geojson. Output will be individual tiles.
        _LOG.info('Found geojson `input_region`, outputing tiles.')
        geopolygon = Geometry(input_region['geometry'], CRS('EPSG:4326'))  # GeoJSON is always 4326
        return GriddedTaskGenerator(storage, geopolygon=geopolygon)

    elif 'from_file' in input_region:
        _LOG.info('Input spatial region specified by file: %s', input_region['from_file'])
        import fiona
        import shapely.ops
        from shapely.geometry import shape, mapping
        with fiona.open(input_region['from_file']) as input_region:
            joined = shapely.ops.unary_union(list(shape(geom['geometry']) for geom in input_region))
            final = joined.convex_hull
            crs = CRS(input_region.crs_wkt)
            boundary_polygon = Geometry(mapping(final), crs)

        return GriddedTaskGenerator(storage, geopolygon=boundary_polygon)

    else:
        _LOG.info('Generating statistics for an ungridded `input region`. Output as a single file.')
        return NonGriddedTaskGenerator(input_region=input_region, storage=storage)


def do_nothing(*args, **varargs):
    pass


class GriddedTaskGenerator(object):
    def __init__(self, storage, geopolygon=None, cell_index=None):
        self.grid_spec = _make_grid_spec(storage)
        self.geopolygon = geopolygon
        self.cell_index = cell_index

    def __call__(self, index, sources_spec, date_ranges, ):
        """
        Generate the required tasks through time and across a spatial grid.
        
        Input region can be limited by specifying either/or both of `geopolygon` and `cell_index`, which
        will both result in only datasets covering the poly or cell to be included.
    
        :param index: Datacube Index
        :return:
        """
        workflow = GridWorkflow(index, grid_spec=self.grid_spec)

        for time_period in date_ranges:
            _LOG.info('Making output product tasks for time period: %s', time_period)
            timer = MultiTimer().start('creating_tasks')

            # Tasks are grouped by tile_index, and may contain sources from multiple places
            # Each source may be masked by multiple masks
            tasks = {}
            for source_spec in sources_spec:
                group_by_name = source_spec.get('group_by', DEFAULT_GROUP_BY)
                source_filter = source_spec.get('source_filter', None)
                data = workflow.list_cells(product=source_spec['product'], time=time_period,
                                           group_by=group_by_name, geopolygon=self.geopolygon,
                                           cell_index=self.cell_index, source_filter=source_filter)
                masks = [workflow.list_cells(product=mask['product'], time=time_period,
                                             group_by=group_by_name, geopolygon=self.geopolygon,
                                             cell_index=self.cell_index)
                         for mask in source_spec.get('masks', [])]

                for tile_index, sources in data.items():
                    task = tasks.setdefault(tile_index, StatsTask(time_period=time_period, tile_index=tile_index))
                    task.sources.append({
                        'data': sources,
                        'masks': [mask.get(tile_index) for mask in masks],
                        'spec': source_spec,
                    })

            timer.pause('creating_tasks')
            if tasks:
                _LOG.info('Created %s tasks for time period: %s. In: %s', len(tasks), time_period, timer)
            for task in tasks.values():
                yield task


def _make_grid_spec(storage):
    """Make a grid spec based on a storage spec."""
    assert 'tile_size' in storage

    crs = CRS(storage['crs'])
    return GridSpec(crs=crs,
                    tile_size=[storage['tile_size'][dim] for dim in crs.dimensions],
                    resolution=[storage['resolution'][dim] for dim in crs.dimensions])


class NonGriddedTaskGenerator(object):
    def __init__(self, input_region, storage):
        self.input_region = input_region
        self.storage = storage

    def __call__(self, index, sources_spec, date_ranges):
        """
        Make stats tasks for a single defined spatial region, not part of a grid.
    
        :param index: database index
        :param input_region: dictionary of query parameters defining the target input region. Usually
                             x/y spatial boundaries.
        :return:
        """
        make_tile = ArbitraryTileMaker(index, self.input_region, self.storage)

        for time_period in date_ranges:
            task = StatsTask(time_period)

            for source_spec in sources_spec:
                group_by_name = source_spec.get('group_by', DEFAULT_GROUP_BY)

                # Build Tile
                data = make_tile(product=source_spec['product'], time=time_period, group_by=group_by_name)

                masks = [make_tile(product=mask['product'], time=time_period, group_by=group_by_name)
                         for mask in source_spec.get('masks', [])]

                if len(data.sources.time) == 0:
                    continue

                task.sources.append({
                    'data': data,
                    'masks': masks,
                    'spec': source_spec,
                })

            if task.sources:
                _LOG.info('Created task for time period: %s', time_period)
                yield task


class ArbitraryTileMaker(object):
    """
    Create a :class:`Tile` which can be used by :class:`GridWorkflow` to later load the required data.
    
    """
    def __init__(self, index, input_region, storage):
        self.dc = Datacube(index=index)
        self.input_region = input_region
        self.storage = storage

    def __call__(self, product, time, group_by):
        # Find the sources for each layer
        datasets = self.dc.find_datasets(product=product, time=time, **self.input_region)
        group_by = query_group_by(group_by=group_by)
        sources = self.dc.group_datasets(datasets, group_by)

        # Find the geopolygon for the tile of interest
        output_crs = CRS(self.storage['crs'])
        output_resolution = [self.storage['resolution'][dim] for dim in output_crs.dimensions]

        geopoly = query_geopolygon(**self.input_region)
        geopoly = geopoly.to_crs(output_crs)

        geobox = GeoBox.from_geopolygon(geopoly, resolution=output_resolution)

        return Tile(sources, geobox)


def pickle_stream(objs, filename):
    idx = 0
    with open(filename, 'wb') as stream:
        for idx, obj in enumerate(objs, start=1):
            cloudpickle.dump(obj, stream, pickle.HIGHEST_PROTOCOL)
    return idx


def unpickle_stream(filename):
    with open(filename, 'rb') as stream:
        while True:
            try:
                yield pickle.load(stream)
            except EOFError:
                break


if __name__ == '__main__':
    main()
