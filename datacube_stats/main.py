"""
Create statistical summaries command

"""
from __future__ import absolute_import, print_function

import logging
from functools import partial

import click
import numpy as np
import pandas as pd
import xarray

import datacube_stats
from datacube import Datacube
from datacube.api import make_mask
from datacube.api.grid_workflow import GridWorkflow, Tile
from datacube.api.query import query_group_by, query_geopolygon, Query
from datacube.model import GridSpec, CRS, GeoBox, GeoPolygon
from datacube.storage.masking import mask_valid_data as mask_invalid_data
from datacube.ui import click as ui
from datacube.ui.click import to_pathlib
from datacube.utils import read_documents, import_function, tile_iter
from datacube.utils.dates import date_sequence
from datacube_stats.models import StatsTask, OutputProduct
from datacube_stats.output_drivers import OUTPUT_DRIVERS
from datacube_stats.runner import run_tasks
from datacube_stats.statistics import StatsConfigurationError, STATS
from datacube_stats.timer import MultiTimer

__all__ = ['StatsApp', 'main']
_LOG = logging.getLogger(__name__)
DEFAULT_GROUP_BY = 'time'
DEFAULT_COMPUTATION_OPTIONS = {'chunking': {'x': 1000, 'y': 1000}}


@click.command(name='datacube-stats')
@click.argument('stats_config_file',
                type=click.Path(exists=True, readable=True, writable=False, dir_okay=False),
                callback=to_pathlib)
@click.option('--queue-size', type=click.IntRange(1, 100000), default=50,
              help='Number of tasks to queue at the start')
@ui.global_cli_options
@ui.executor_cli_options
@ui.pass_index(app_name='datacube-stats')
def main(index, stats_config_file, executor, queue_size):
    logging.getLogger('datacube.storage.storage').setLevel(logging.INFO)
    timer = MultiTimer()
    timer.start('main')
    _, config = next(read_documents(stats_config_file))
    app = create_stats_app(config, index)
    app.queue_size = queue_size
    app.validate()

    app.run(executor)
    timer.pause('main')
    _LOG.info('Stats processing completed in %s seconds.', timer.run_times['main'])


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
        # Check output product names are unique
        output_names = [prod['name'] for prod in self.output_product_specs]
        duplicate_names = [x for x in output_names if output_names.count(x) > 1]
        if duplicate_names:
            raise StatsConfigurationError('Output products must all have different names. '
                                          'Duplicates found: %s' % duplicate_names)

        # Check statistics are available
        requested_statistics = set(prod['statistic'] for prod in self.output_product_specs)
        available_statistics = set(STATS.keys())
        if not requested_statistics.issubset(available_statistics):
            raise StatsConfigurationError(
                'Requested Statistic(s) %s are not valid statistics. Available statistics are: %s'
                % (requested_statistics - available_statistics, available_statistics))

        # Check consistent measurements
        try:
            first_source = self.sources[0]
        except IndexError:
            raise StatsConfigurationError('No data sources specified.')
        if not all(first_source['measurements'] == source['measurements'] for source in self.sources):
            raise StatsConfigurationError("Configuration Error: listed measurements of source products "
                                          "are not all the same.")

        assert callable(self.task_generator)
        assert callable(self.output_driver)
        assert hasattr(self.output_driver, 'open_output_files')
        assert hasattr(self.output_driver, 'write_data')
        assert callable(self.process_completed)

    def run(self, executor):
        output_products = self.ensure_output_products()

        tasks = self.generate_tasks(output_products)

        app_info = _get_app_metadata(self.config_file)
        output_driver = partial(self.output_driver, output_path=self.location, app_info=app_info,
                                storage=self.storage)
        task_runner = partial(execute_task, output_driver=output_driver, chunking=self.computation['chunking'])
        run_tasks(tasks, executor, task_runner, self.process_completed, queue_size=self.queue_size)

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

    def ensure_output_products(self, metadata_type='eo'):
        """
        Return a dict mapping Output Product Name to OutputProduct

        StatProduct describes the structure and how to compute the output product.
        """
        _LOG.info('Creating output products')

        output_products = {}

        measurements = _source_measurement_defs(self.index, self.sources)

        metadata_type = self.index.metadata_types.get_by_name(metadata_type)
        for product_def in self.output_product_specs:
            output_products[product_def['name']] = OutputProduct.from_json_definition(
                metadata_type=metadata_type,
                input_measurements=measurements,
                storage=self.storage,
                definition=product_def)

        # TODO: Create the output product in the database

        return output_products


def execute_task(task, output_driver, chunking):
    """
    Load data, run the statistical operations and write results out to the filesystem.

    :param datacube_stats.models.StatsTask task:
    :type output_driver: OutputDriver
    :param chunking: dict of dimension sizes to chunk the computation by
    """
    timer = MultiTimer()
    with output_driver(task=task) as output_files:
        for sub_tile_slice in tile_iter(task.sample_tile, chunking):
            try:
                timer.start('loading_data')
                data = _load_data(sub_tile_slice, task.sources)
                timer.pause('loading_data')

                for prod_name, stat in task.output_products.items():
                    _LOG.debug("Computing %s in tile %s", prod_name, sub_tile_slice)
                    assert stat.masked  # TODO: not masked
                    timer.start(prod_name)
                    stats_data = stat.compute(data)
                    timer.pause(prod_name)

                    # For each of the data variables, shove this chunk into the output results
                    timer.start('writing_data')
                    for var_name, var in stats_data.data_vars.items():
                        output_files.write_data(prod_name, var_name, sub_tile_slice, var.values)
                    timer.pause('writing_data')
            except EmptyChunkException:
                _LOG.debug('Error: No data returned while loading %s for %s. May have all been masked',
                           sub_tile_slice, task)

    _LOG.info('Completed %s %s task with %s data sources. Processing took: %s', task.tile_index,
              [d.strftime('%Y-%m-%d') for d in task.time_period], task.data_sources_length(), timer)


class EmptyChunkException(Exception):
    pass


def _load_data(sub_tile_slice, sources):
    """
    Load a masked chunk of data from the datacube, based on a specification and list of datasets in `sources`.

    :param sub_tile_slice: A portion of a tile, tuple coordinates
    :param sources: a dictionary containing `data`, `spec` and `masks`
    :return: :class:`xarray.Dataset` containing loaded data. Will be indexed and sorted by time.
    """
    datasets = [_load_masked_data(sub_tile_slice, source_prod) for source_prod in sources]  # list of datasets
    for idx, dataset in enumerate(datasets):
        dataset.coords['source'] = ('time', np.repeat(idx, dataset.time.size))
    datasets = xarray.concat(datasets, dim='time')  # Copies all the data
    if len(datasets.time) == 0:
        raise EmptyChunkException()
    return datasets.isel(time=datasets.time.argsort())  # sort along time dim  # Copies all the data again
    # return inplace_isel(datasets, time=datasets.time.argsort())


def _load_masked_data(sub_tile_slice, source_prod):
    data = GridWorkflow.load(source_prod['data'][sub_tile_slice],
                             measurements=source_prod['spec']['measurements'])
    crs = data.crs
    data = mask_invalid_data(data)

    if 'masks' in source_prod and 'masks' in source_prod['spec']:
        for mask_spec, mask_tile in zip(source_prod['spec']['masks'], source_prod['masks']):
            fuse_func = import_function(mask_spec['fuse_func']) if 'fuse_func' in mask_spec else None
            mask = GridWorkflow.load(mask_tile[sub_tile_slice],
                                     measurements=[mask_spec['measurement']],
                                     fuse_func=fuse_func)[mask_spec['measurement']]
            mask = make_mask(mask, **mask_spec['flags'])
            data = data.where(mask)
            del mask
    data.attrs['crs'] = crs  # Reattach crs, it gets lost when masking
    return data


def _source_measurement_defs(index, sources):
    """
    Look up desired measurements from sources in the database index

    :return: list of measurement definitions
    """
    source_defn = sources[0]  # TODO: Check sources should have been checked to all have the same measureemnts

    source_measurements = index.products.get_by_name(source_defn['product']).measurements

    measurements = [measurement for name, measurement in source_measurements.items()
                    if name in source_defn['measurements']]

    return measurements


def _get_app_metadata(config_file):
    return {
        'lineage': {
            'algorithm': {
                'name': 'datacube-stats',
                'version': datacube_stats.__version__,
                'repo_url': 'https://github.com/GeoscienceAustralia/agdc_statistics.git',
                'parameters': {'configuration_file': str(config_file)}
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


def create_stats_app(config, index=None):
    """
    Create a StatsApp to run a processing job, based on a configuration file

    :param config: dictionary based configuration
    :param index: open database connection
    :return: read to run StatsApp
    """
    stats_app = StatsApp()
    stats_app.index = index
    stats_app.config_file = config
    stats_app.storage = config['storage']
    stats_app.sources = config['sources']
    stats_app.output_product_specs = config['output_products']
    stats_app.location = config['location']
    stats_app.computation = config.get('computation', DEFAULT_COMPUTATION_OPTIONS)

    date_ranges = config['date_ranges']
    if 'type' not in date_ranges or date_ranges['type'] == 'simple':
        stats_app.date_ranges = list(date_sequence(start=pd.to_datetime(date_ranges['start_date']),
                                                   end=pd.to_datetime(date_ranges['end_date']),
                                                   stats_duration=date_ranges['stats_duration'],
                                                   step_size=date_ranges['step_size']))
    elif date_ranges['type'] == 'find_daily_data':
        product_names = [source['product'] for source in config['sources']]
        stats_app.date_ranges = list(_find_periods_with_data(index, product_names=product_names,
                                                             start_date=date_ranges['start_date'],
                                                             end_date=date_ranges['end_date']))
    else:
        raise StatsConfigurationError('Unknown date_ranges specification. Should be type=simple or '
                                      'type=find_daily_data')

    grid_spec = _make_grid_spec(config['storage'])
    try:
        if 'tile' in config['input_region']:  # Simple, single tile
            stats_app.task_generator = partial(_generate_gridded_tasks, grid_spec=grid_spec,
                                               cell_index=config['input_region']['tile'])
        elif 'geometry' in config['input_region']:  # Larger spatial region
            # A large, multi-tile input region, specified as geojson. Output will be individual tiles.
            _LOG.info('Found geojson `input region`, outputing tiles.')
            geopolygon = GeoPolygon.from_geojson(config['input_region']['geometry'], CRS('EPSG:4326'))
            stats_app.task_generator = partial(_generate_gridded_tasks, grid_spec=grid_spec, geopolygon=geopolygon)
        elif 'from_file' in config['input_region']:
            _LOG.info('Input spatial region specified by file: %s', config['input_region']['from_file'])
            import fiona
            import shapely.ops
            from shapely.geometry import shape, mapping
            with fiona.open(config['input_region']['from_file']) as input_region:
                joined = shapely.ops.unary_union(list(shape(geom['geometry']) for geom in input_region))
                final = joined.convex_hull
                crs = CRS(input_region.crs_wkt)
                boundary_polygon = GeoPolygon.from_geojson(mapping(final), crs)

            stats_app.task_generator = partial(_generate_gridded_tasks, grid_spec=grid_spec,
                                               geopolygon=boundary_polygon)

        else:
            _LOG.info('Generating statistics for an ungridded `input region`. Output as a single file.')
            stats_app.task_generator = partial(_generate_non_gridded_tasks, input_region=config['input_region'],
                                               storage=stats_app.storage)
    except KeyError:
        _LOG.info('Default output, full available spatial region, gridded files.')
        stats_app.task_generator = partial(_generate_gridded_tasks, grid_spec=grid_spec)

    try:
        stats_app.output_driver = OUTPUT_DRIVERS[config['storage']['driver']]
    except KeyError:
        specified_driver = config.get('storage', {}).get('driver')
        if specified_driver is None:
            msg = 'No output driver specified.'
        else:
            msg = 'Invalid output driver "{}" specified.'
        raise StatsConfigurationError('{} Specify one of {} in storage->driver in the '
                                      'configuration file.'.format(msg, OUTPUT_DRIVERS.keys()))
    stats_app.process_completed = do_nothing  # TODO: Save dataset to database

    return stats_app


def do_nothing(task):
    pass


def _make_grid_spec(storage):
    """Make a grid spec based on a storage spec."""
    assert 'tile_size' in storage

    crs = CRS(storage['crs'])
    return GridSpec(crs=crs,
                    tile_size=[storage['tile_size'][dim] for dim in crs.dimensions],
                    resolution=[storage['resolution'][dim] for dim in crs.dimensions])


def _generate_gridded_tasks(index, sources_spec, date_ranges, grid_spec, geopolygon=None, cell_index=None):
    """
    Generate the required tasks through time and across a spatial grid.

    :param index: Datacube Index
    :return:
    """
    workflow = GridWorkflow(index, grid_spec=grid_spec)

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
                                       group_by=group_by_name, geopolygon=geopolygon,
                                       cell_index=cell_index, source_filter=source_filter)
            masks = [workflow.list_cells(product=mask['product'], time=time_period,
                                         group_by=group_by_name, geopolygon=geopolygon,
                                         cell_index=cell_index)
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


def _generate_non_gridded_tasks(index, sources_spec, date_ranges, input_region, storage):
    """
    Make stats tasks for a defined spatial region, that doesn't fit into a standard grid.

    :param index: database index
    :param input_region: dictionary of query parameters defining the target input region. Usually
                         x/y spatial boundaries.
    :return:
    """
    dc = Datacube(index=index)

    def make_tile(product, time, group_by):
        datasets = dc.find_datasets(product=product, time=time, **input_region)
        group_by = query_group_by(group_by=group_by)
        sources = dc.group_datasets(datasets, group_by)

        res = storage['resolution']

        geopoly = query_geopolygon(**input_region)
        geopoly = geopoly.to_crs(CRS(storage['crs']))
        geobox = GeoBox.from_geopolygon(geopoly, (res['y'], res['x']))

        return Tile(sources, geobox)

    for time_period in date_ranges:
        task = StatsTask(time_period)

        for source_spec in sources_spec:
            group_by_name = source_spec.get('group_by', DEFAULT_GROUP_BY)

            # Build Tile
            data = make_tile(product=source_spec['product'], time=time_period,
                             group_by=group_by_name)

            masks = [make_tile(product=mask['product'], time=time_period,
                               group_by=group_by_name)
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


if __name__ == '__main__':
    main()
