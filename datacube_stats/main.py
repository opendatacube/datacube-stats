"""
Create statistical summaries command

"""
from __future__ import absolute_import, print_function

import logging
from functools import partial
from textwrap import dedent

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
import fiona
import itertools
import rasterio.features
import sys

import datacube_stats
import datacube
from datacube import Datacube
from datacube.api import make_mask, GridWorkflow, Tile
from datacube.api.query import query_group_by, query_geopolygon, Query
from datacube.model import GridSpec
from datacube.utils.geometry import CRS, GeoBox, Geometry
from datacube.storage.masking import mask_invalid_data
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
from otps.predict_wrapper import predict_tide
from otps import TimePoint
from datetime import timedelta
from datetime import datetime
from operator import itemgetter

__all__ = ['StatsApp', 'main']
_LOG = logging.getLogger(__name__)
DEFAULT_GROUP_BY = 'time'
DEFAULT_COMPUTATION_OPTIONS = {'chunking': {'x': 400, 'y': 400}}
LS7_SLC_DT = datetime.strptime("2003-05-01", "%Y-%m-%d")


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

        #: Define tide class to accept all attributes
        self.tide_class = []

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
                                storage=self.storage)
        task_runner = partial(execute_task,
                              output_driver=output_driver,
                              chunking=self.computation['chunking'], tide_class=self.tide_class)
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

def geometry_mask(geoms, geobox, all_touched=False, invert=False):

    return rasterio.features.geometry_mask([geom.to_crs(geobox.crs) for geom in geoms],
                                           out_shape=geobox.shape,
                                           transform=geobox.affine,
                                           all_touched=all_touched,
                                           invert=invert)

def execute_task(task, output_driver, chunking, tide_class):
    """
    Load data, run the statistical operations and write results out to the filesystem.

    :param datacube_stats.models.StatsTask task:
    :type output_driver: OutputDriver
    :param chunking: dict of dimension sizes to chunk the computation by
    """
    timer = MultiTimer().start('total')
    datacube.set_options(reproject_threads=1)
    geom = None
    try:
        with output_driver(task=task) as output_files:
            geom =  tide_class.get('geom')
            for sub_tile_slice in tile_iter(task.sample_tile, chunking):
                load_process_save_chunk(output_files, sub_tile_slice, task, timer, geom)
    except OutputFileAlreadyExists as e:
        _LOG.warning(e)
    except Exception as e:
        _LOG.error("Error processing task: %s", task)
        raise StatsProcessingException("Error processing task: %s" % task) from e

    timer.pause('total')
    _LOG.info('Completed %s %s task with %s data sources. Processing took: %s', task.tile_index,
              [d.strftime('%Y-%m-%d') for d in task.time_period], task.data_sources_length(), timer)
    return task


def load_process_save_chunk(output_files, chunk, task, timer, geom):
    try:
        timer.start('loading_data')
        data = _load_data(chunk, task.sources)
        timer.pause('loading_data')

        for prod_name, stat in task.output_products.items():
            _LOG.info("Computing %s in tile %s %s. Current timing: %s",
                      prod_name, task.tile_index, chunk, timer)
            timer.start(prod_name)
            # scale it to represent in float32 for precise geomedian
            data = data/10000 if stat.stat_name == 'precisegeomedian' else data
            geobox = data.geobox
            ndata = stat.compute(data)
            # mask as per geometry now. Ideally Mask should have been done before computation
            # But masking before computation is a problem as it gets 0 value
            mask = geometry_mask([geom], geobox, invert=True)
            ndata = ndata.where(mask)
            timer.pause(prod_name)
            # For each of the data variables, shove this chunk into the output results
            timer.start('writing_data')
            if stat.stat_name == "clearcount":
               output_files.write_data(prod_name, ndata.name, chunk, ndata.values)
            else: 
                for var_name, var in ndata.data_vars.items():  # TODO: Move this loop into output_files
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
    #import pdb; pdb.set_trace() 
    return data


def sensible_mask_invalid_data(data):
    # TODO This should be pushed up to datacube-core
    # xarray.DataArray.where() converts ints to floats, since NaNs are used to represent nodata
    # by default, this uses float64, which is way over the top for an int16 value, so
    # lets convert to float32 first, to save a bunch of memory.
    data = _convert_to_floats(data)  # This is stripping out variable attributes
    return mask_invalid_data(data)


def sensible_where(data, mask):
    data = _convert_to_floats(data)  # This is stripping out variable attributes
    return data.where(mask)


def _convert_to_floats(data):
    # Use float32 instead of float64 if input dtype is int16
    assert isinstance(data, xarray.Dataset)
    for name, dataarray in data.data_vars.items():
        if dataarray.dtype != np.int16:
            return data
    return data.apply(lambda d: d.astype(np.float32), keep_attrs=True)


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
    stats_app.location = config.get('location', output_location)  # Write files to current directory if not set in config
    stats_app.computation = config.get('computation', DEFAULT_COMPUTATION_OPTIONS)
    stats_app.tide_class = config['tide_class']
    stats_app.date_ranges = _configure_date_ranges(index, config)
    stats_app.task_generator = _create_task_generator(config.get('input_region'), stats_app.tide_class, stats_app.storage)
    stats_app.output_driver = _prepare_output_driver(stats_app.storage)
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

def _create_task_generator(input_region, tide_class, storage):
    grid_spec = _make_grid_spec(storage)
   
    if input_region is None:
        _LOG.info('No input_region specified. Generating full available spatial region, gridded files.')
        return partial(_generate_gridded_tasks, grid_spec=grid_spec)

    if tide_class is not None: 
        _LOG.info('Generating polygon images ')
        return partial(_generate_non_gridded_my_tasks, input_region=input_region, tide_class=tide_class, storage=storage)

    if 'tile' in input_region:  # Simple, single tile
        return partial(_generate_gridded_tasks, grid_spec=grid_spec, cell_index=input_region['tile'])

    elif 'geometry' in input_region:  # Larger spatial region
        # A large, multi-tile input region, specified as geojson. Output will be individual tiles.
        _LOG.info('Found geojson `input_region`, outputing tiles.')
        geopolygon = Geometry(input_region['geometry'], CRS('EPSG:4326'))
        return partial(_generate_gridded_tasks, grid_spec=grid_spec, geopolygon=geopolygon)

    elif 'from_file' in input_region:
        _LOG.info('Input spatial region specified by file: %s', input_region['from_file'])
        import fiona
        import shapely.ops
        from shapely.geometry import shape, mapping
        with fiona.open(input_region['from_file']) as input_region:
            joined = shapely.ops.unary_union(list(shape(geom['geometry']) for geom in input_region))
            final = joined.convex_hull
            crs = CRS(str(input_region.crs_wkt))
            boundary_polygon = Geometry(mapping(final), crs)

        return partial(_generate_gridded_tasks, grid_spec=grid_spec, geopolygon=boundary_polygon)

    else:
        _LOG.info('Generating statistics for an ungridded `input region`. Output as a single file.')
        return partial(_generate_non_gridded_tasks, input_region=input_region, storage=storage)


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
    
    Input region can be limited by specifying either/or both of `geopolygon` and `cell_index`, which
    will both result in only datasets covering the poly or cell to be included.

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


def _generate_non_gridded_my_tasks(index, sources_spec, date_ranges, tide_class, input_region, storage):
    """
    Make stats tasks for a defined spatial region, that doesn't fit into a standard grid.

    :param index: database index
    :param input_region: dictionary of query parameters defining the target input region. Usually
                         x/y spatial boundaries.
    :return:
    """
    dc = Datacube(index=index)

    def make_tile(product, time, group_by, otpstime, geopoly):
        datasets = dc.find_datasets(product=product, time=time, geopolygon=geopoly, group_by=group_by)
        fil_datasets = list()
        dt = list()
        if tide_class.get('product') == 'dry' or tide_class.get('product') == 'wet':
            dt = otpstime
        else:
            dt= [dd[0] for dd in otpstime]
        # get the date list out of dt string and compare within the epoch
        for ds in datasets:
            if ds.time.begin.date().strftime("%Y-%m-%d") in dt:
                #_LOG.info("location of filter ds %s", ds.local_path)
                fil_datasets.append(ds)
        if len(fil_datasets) == 0:
            _LOG.info("No matched for product %s on %s", product, str(otpstime))
            return None
        datasets = fil_datasets
        group_by = query_group_by(group_by=group_by)
        sources = dc.group_datasets(datasets, group_by)
        res = storage['resolution']
        geobox = GeoBox.from_geopolygon(geopoly, (res['y'], res['x']))
        return Tile(sources, geobox)

    def list_poly_dates(geom):
        datasets = list()
        all_times = list()
        for source_spec in sources_spec:
            for mask in source_spec.get('masks', []):
                group_by_name = source_spec.get('group_by', 'solar_day')
                gl_range = (tide_class['global_epoch_start'], tide_class['global_epoch_end'])
                if (mask['product'] == 'ls7_pq_albers') and (datetime.strptime(gl_range[0], "%Y-%m-%d") > LS7_SLC_DT) and \
                    (tide_class.get('ls7_slc_off') is None):
                    continue
                if (tide_class.get('ls7_slc_off') is None) and (mask['product'] == 'ls7_pq_albers') and  \
                        (datetime.strptime(gl_range[1], "%Y-%m-%d")  > LS7_SLC_DT):
                    gl_range = (gl_range[0], LS7_SLC_DT)
             

                ds = dc.find_datasets(product=mask['product'], time=gl_range, geopolygon=geom, group_by=group_by_name)
                group_by = query_group_by(group_by=group_by_name)
                sources = dc.group_datasets(ds, group_by)
                if len(ds) > 0 :
                    all_times = all_times + [dd for dd in sources.time.data.astype('M8[s]').astype('O').tolist()]    
        return sorted(all_times)

    def filter_sub_class(list_low, list_high, ebb_flow):
        if tide_class['sub_class'] == 'e':
           if list_low is not None:
               key = set(e[0] for e in eval(ebb_flow) if e[1] == 'e')
               list_low = [ff for ff in list_low if ff[0] in key]
           key = set(e[0] for e in eval(ebb_flow) if e[1] == 'e')
           list_high = [ff for ff in list_high if ff[0] in key]
        if tide_class['sub_class'] == 'f':
           if list_low is not None:
               key = set(e[0] for e in eval(ebb_flow) if e[1] == 'f')
               list_low = [ff for ff in list_low if ff[0] in key]
           key = set(e[0] for e in eval(ebb_flow) if e[1] == 'f')
           list_high = [ff for ff in list_high if ff[0] in key]
        if tide_class['sub_class'] == 'ph':
           if list_low is not None:
               key = set(e[0] for e in eval(ebb_flow) if e[1] == 'ph')
               list_low = [ff for ff in list_low if ff[0] in key]
           key = set(e[0] for e in eval(ebb_flow) if e[1] == 'ph')
           list_high = [ff for ff in list_high if ff[0] in key]
        if tide_class['sub_class'] == 'pl':
           if list_low is not None:
               key = set(e[0] for e in eval(ebb_flow) if e[1] == 'pl')
               list_low = [ff for ff in list_low if ff[0] in key]
           key = set(e[0] for e in eval(ebb_flow) if e[1] == 'pl')
           list_high = [ff for ff in list_high if ff[0] in key]
        return list_low, list_high, ebb_flow


    def extract_otps_computed_data(dates, date_ranges, per, ln, la):
        
        tp = list()
        tide_dict = dict()
        ndate_list=list()
        mnt=timedelta(minutes=15)
        for dt in dates:
            ndate_list.append(dt-mnt)
            ndate_list.append(dt)
            ndate_list.append(dt+mnt)
        for dt in ndate_list:
            tp.append(TimePoint(ln, la, dt))
        tides = predict_tide(tp)
        if len(tides) == 0:
           _LOG.info("No tide height observed from OTPS model within lat/lon range")
           sys.exit()
        _LOG.info("received from predict tides 15 minutes before and after %s", str(datetime.now()))
        # collect in ebb/flow list
        for tt in tides:
            tide_dict[datetime.strptime(tt.timepoint.timestamp.isoformat()[0:19], "%Y-%m-%dT%H:%M:%S")] = tt.tide_m
        tmp_lt = sorted(tide_dict.items(), key=lambda x: x[0])
        dtlist = tmp_lt[1::3]
        my_data = sorted(dtlist, key=itemgetter(1))
        # print ([[dt[0].strftime("%Y-%m-%d %H:%M:%S"), dt[1]] for dt in tmp_lt])
        tmp_lt = [[tmp_lt[i+1][0].strftime("%Y-%m-%d"), 'ph'] \
                 if tmp_lt[i][1] < tmp_lt[i+1][1] and tmp_lt[i+2][1] <  tmp_lt[i+1][1]  else \
                 [tmp_lt[i+1][0].strftime("%Y-%m-%d"), 'pl'] if tmp_lt[i][1] > tmp_lt[i+1][1] and \
                 tmp_lt[i+2][1] >  tmp_lt[i+1][1]  else [tmp_lt[i+1][0].strftime("%Y-%m-%d"),'f'] \
                 if tmp_lt[i][1] < tmp_lt[i+2][1] else [tmp_lt[i+1][0].strftime("%Y-%m-%d"),'e'] \
                 for i in range(0, len(tmp_lt), 3)]
        #_LOG.info('EBB FLOW tide details for entire archive of LS data %s', str(tmp_lt))
        PERC = 25  if per == 50 else per
        max_height=my_data[-1][1]
        min_height=my_data[0][1]
        dr = max_height - min_height
        lmr = min_height + dr*PERC*0.01   # low tide max range
        hlr = max_height - dr*PERC*0.01   # high tide range
        list_low = list()
        list_high = list()
        if PERC == 50:
            list_high = sorted([[x[0].strftime('%Y-%m-%d'), x[1]] for x  in my_data if (x[1] >= lmr) & (x[1] <= hlr) &
                         (x[0] >= date_ranges[0][0]) & (x[0] <= date_ranges[0][1])])
            _LOG.info(" 50 PERCENTAGE sorted date tide list " + str(len(high)))
            #_LOG.info('Created middle dates and tide heights for time period: %s %s', date_ranges, str(list_high))
        else:
            list_low = sorted([[x[0].strftime('%Y-%m-%d'), x[1]] for x in my_data if (x[1] <= lmr) & 
                                (x[0] >= date_ranges[0][0]) & (x[0] <= date_ranges[0][1])])
            list_high = sorted([[x[0].strftime('%Y-%m-%d'), x[1]] for x in my_data if (x[1] >= hlr) &
                                (x[0] >= date_ranges[0][0]) & (x[0] <= date_ranges[0][1])])
            #_LOG.info('Created low percentage dates and tide heights for time period: %s %s', date_ranges, str(list_low))
            #_LOG.info('\nCreated high percentage dates and tide heights for time period: %s %s', date_ranges, str(list_high))

        _LOG.info( "product %s and percentage received %d ", tide_class['product'], per)
        ebb_flow = str([tt for tt in tmp_lt if (datetime.strptime(tt[0], "%Y-%m-%d") >= date_ranges[0][0]) & 
                                               (datetime.strptime(tt[0], "%Y-%m-%d") <= date_ranges[0][1])])
        #_LOG.info('\nCreated EBB FLOW for time period: %s %s', date_ranges, ebb_flow)
        #print (" EBB FLOW for this epoch " + str([tt for tt in tmp_lt if (datetime.strptime(tt[0], "%Y-%m-%d") >= date_ranges[0][0]) & 
        #         (datetime.strptime(tt[0], "%Y-%m-%d") <= date_ranges[0][1])]))
        return dtlist, list_low, list_high, ebb_flow

    def get_hydrologic_months():
        year = tide_class['year']
        months = tide_class.get('months')
        if months is not None:
            st_dt =  str(year+1)+str(months[0])+'01'
            en_dt =  str(year+1)+str(months[1])+'30'
        else:
            st_dt = '01/07/' + str(year+1)
            en_dt = '30/11/' + str(year+1)
        date_list = pd.date_range(st_dt, en_dt)
        date_list = date_list.to_datetime().astype(str).tolist()
        return date_list
        

    with fiona.open(input_region['from_file']) as input_region:
        crs = CRS(str(input_region.crs_wkt))
        if tide_class.get('product') == 'dry' or tide_class.get('product') == 'wet':
            for feature in input_region:
                # get the year id and geometry
                if tide_class.get('product') == 'dry':
                    tide_class['year'] = int(feature['properties']['DN'])
                else:
                    tide_class['year'] = int(feature['properties']['WY1'])
                Id = feature['properties']['ID']
                if Id in tide_class['feature_id']:
                    geom = feature['geometry']
                    boundary_polygon = Geometry(geom, crs)
                    tide_class['geom'] = boundary_polygon
                    tile_index=(str(Id), tide_class['year'])
                    list_time = get_hydrologic_months()  
                    for time_period in date_ranges:
                        task = StatsTask(time_period=time_period, tile_index=tile_index)
                        for source_spec in sources_spec:
                            group_by_name = source_spec.get('group_by', 'solar_day')

                        # Build Tile
                            ep_range = time_period
                            if (source_spec['product'] == 'ls7_nbar_albers') and (ep_range[0] > LS7_SLC_DT)  \
                                    and (tide_class.get('ls7_slc_off') is None):
                                continue
                            if (tide_class.get('ls7_slc_off') is None) and (source_spec['product'] == 'ls7_nbar_albers') \
                                    and (ep_range[1]  > LS7_SLC_DT):
                                ep_range = (ep_range[0], LS7_SLC_DT)
                            data = make_tile(product=source_spec['product'], time=ep_range,
                                             group_by=group_by_name, otpstime=list_time, geopoly=boundary_polygon) 
                            masks = [make_tile(product=mask['product'], time=ep_range,
                                             group_by=group_by_name, otpstime=list_time, geopoly=boundary_polygon)
                                     for mask in source_spec.get('masks', [])]

                            if data is None:
                                continue
                            _LOG.info("source details for %s and values %s", source_spec['product'], 
                                      str(data.sources.time.values))  
                            task.sources.append({
                                'data': data,
                                'masks': masks,
                                'spec': source_spec,
                            })
                        if task.sources:
                            _LOG.info('Created task for time period: %s', time_period)
                            yield task
            return

         
        for feature in input_region:
            lon = feature['properties']['lon']
            lat = feature['properties']['lat']
            Id = feature['properties']['ID']
            #filter out only on selected features
            if Id in tide_class['feature_id']:
            #To run a limited polygons
            #if Id < 10:
                geom = feature['geometry']
                boundary_polygon = Geometry(geom, crs)
                tide_class['geom'] = boundary_polygon
                _LOG.info("feature %d captured for longitude and latitude %.02f  %.02f", Id, lon, lat)
                _LOG.info('Getting all dates corresponding to this polygon for all sensor data')
                all_dates = list_poly_dates(boundary_polygon)
                all_list, list_low, list_high, ebb_flow = extract_otps_computed_data(all_dates, date_ranges, 
                                                                                     tide_class['percent'], lon, lat)
                # filter out dates as per sub classification of ebb flow
                if tide_class.get('sub_class'):
                    list_low, list_high, ebb_flow = filter_sub_class(list_low, list_high, ebb_flow)
                    _LOG.info("SUB class dates extracted %s for list low %s  and for list high %s", tide_class['sub_class'] , list_low, list_high) 
                prod = list()
                if "low_high" in tide_class['product']:
                    prod = ['low', 'high']
                elif "low" in tide_class['product']:
                    prod = ['low']
                elif "high" in tide_class['product']:
                    prod = ['high']
                else:
                    prod = ['middle']
                tile_index=(lon,lat)
                for time_period, pr in itertools.product(date_ranges, prod) :
                    #nlon = tide_class['sub_class'].upper() if None not in tide_class.get('sub_class') else pr.upper()
                    if tide_class.get('sub_class'):
                        nlon = tide_class.get('sub_class').upper() + "_" + pr.upper()
                    else:
                        nlon = pr.upper()
                    nlon = nlon + "_" + str(lon)
                    tile_index=(nlon,lat)
                    list_time = list_low if pr == 'low' else list_high
                    # Needs to update metadata for ebb flow classification
                    key = set(e[0] for e in list_time)
                    ebb_flow_class = [ff for ff in eval(ebb_flow) if ff[0] in key]
                    tide_class['ebb_flow'] = {'ebb_flow':ebb_flow_class}
                    _LOG.info('\nCreated EBB FLOW for feature %d , length %d, time period: %s \t%s', 
                              Id, len(ebb_flow_class), date_ranges, str(ebb_flow_class))
                    _LOG.info('\n %s DATE LIST for feature %d length %d, time period: %s \t%s',
                              pr.upper(), Id, len(list_time), date_ranges, str(list_time))
                    task = StatsTask(time_period=time_period, tile_index=tile_index,
                                     extras={'percent': tide_class['percent']})
                    _LOG.info("doing for product %s for percentage %s ", pr, str(tide_class['percent']))
                    for source_spec in sources_spec:
                        group_by_name = source_spec.get('group_by', 'solar_day')

                        # Build Tile
                        ep_range = time_period
                        if (source_spec['product'] == 'ls7_nbar_albers') and (ep_range[0] > LS7_SLC_DT)  \
                                and (tide_class.get('ls7_slc_off') is None):
                            continue
                        if (tide_class.get('ls7_slc_off') is None) and (source_spec['product'] == 'ls7_nbar_albers') \
                                and (ep_range[1]  > LS7_SLC_DT):
                            ep_range = (ep_range[0], LS7_SLC_DT)

                        data = make_tile(product=source_spec['product'], time=ep_range,
                                         group_by=group_by_name, otpstime=list_time, geopoly=boundary_polygon) 
                        masks = [make_tile(product=mask['product'], time=ep_range,
                                         group_by=group_by_name, otpstime=list_time, geopoly=boundary_polygon)
                                 for mask in source_spec.get('masks', [])]

                        if data is None:
                            continue
                        _LOG.info("source details for %s and values %s", pr, str(data.sources.time.values))  
                        task.sources.append({
                            'data': data,
                            'masks': masks,
                            'spec': source_spec,
                        })
                    if task.sources:
                        _LOG.info('Created task for time period: %s', time_period)
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
