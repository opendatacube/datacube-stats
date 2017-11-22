"""
Create statistical summaries command.

This command is run as ``datacube-stats``, all operation are driven by a configuration file.

"""
from __future__ import absolute_import, print_function

import copy
import logging
from functools import partial
from textwrap import dedent

import click
import numpy as np
import pandas as pd
import pydash
import rasterio.features
import xarray

import datacube
import datacube_stats
from datacube.api import make_mask, GridWorkflow
from datacube.ui import click as ui
from datacube.ui.click import to_pathlib
from datacube.utils import read_documents, import_function
from datacube.utils.geometry import CRS, Geometry
from datacube_stats.models import OutputProduct
from datacube_stats.output_drivers import OUTPUT_DRIVERS, OutputFileAlreadyExists, get_driver_by_name, \
    NoSuchOutputDriver
from datacube_stats.statistics import StatsConfigurationError, STATS
from datacube_stats.utils import cast_back, pickle_stream, unpickle_stream, _find_periods_with_data
from datacube_stats.utils import tile_iter, sensible_mask_invalid_data, sensible_where, sensible_where_inplace
from datacube_stats.utils.dates import date_sequence
from digitalearthau.qsub import with_qsub_runner
from .utils.timer import MultiTimer, wrap_in_timer
from .utils import sorted_interleave
from .tasks import select_task_generator
from .schema import stats_schema

__all__ = ['StatsApp', 'main']
_LOG = logging.getLogger(__name__)


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


def gather_tile_indexes(tile_index, tile_index_file):
    if tile_index is None and tile_index_file is None:
        return None

    assert tile_index is None or tile_index_file is None, \
        "must not specify both tile_index and tile_index_file"

    if tile_index:
        return [tile_index]

    with open(tile_index_file) as fl:
        return [tuple(int(x) for x in l.split())
                for l in fl]


# pylint: disable=too-many-locals
@click.command(name='datacube-stats')
@click.argument('stats_config_file',
                type=click.Path(exists=True, readable=True, writable=False, dir_okay=False),
                callback=to_pathlib)
@click.option('--save-tasks', type=click.Path(exists=False, writable=True, dir_okay=False))
@click.option('--load-tasks', type=click.Path(exists=True, readable=True))
@click.option('--tile-index', nargs=2, type=int, help='Override input_region specified in configuration with a '
                                                      'single tile_index specified as [X] [Y]')
@click.option('--tile-index-file',
              type=click.Path(exists=True, readable=True, dir_okay=False),
              help="A file consisting of tile indexes specified as [X] [Y] per line")
@click.option('--output-location', help='Override output location in configuration file')
@click.option('--year', type=int, help='Override time period in configuration file')
@click.option('--list-statistics', is_flag=True, callback=list_statistics, expose_value=False)
@ui.global_cli_options
@with_qsub_runner()
@click.option('--version', is_flag=True, callback=_print_version,
              expose_value=False, is_eager=True)
@ui.pass_index(app_name='datacube-stats')
def main(index, stats_config_file, qsub, runner, save_tasks, load_tasks,
         tile_index, tile_index_file, output_location, year):
    if qsub is not None:
        # TODO: verify config before calling qsub submit
        click.echo(repr(qsub))
        exit_code, _ = qsub(auto=True)
        return exit_code

    _log_setup()

    timer = MultiTimer().start('main')

    if len(tile_index) == 0:
        tile_index = None

    _, config = next(read_documents(stats_config_file))
    stats_schema(config)

    app = StatsApp.from_configuration_file(config, index,
                                           gather_tile_indexes(tile_index, tile_index_file),
                                           output_location, year)
    app.validate()

    if save_tasks:
        app.save_tasks_to_file(save_tasks)
        failed = 0
    elif load_tasks:
        successful, failed = app.run(runner, load_tasks)
    else:
        successful, failed = app.run(runner)

    timer.pause('main')
    _LOG.info('Stats processing completed in %s seconds.', timer.run_times['main'])

    if failed > 0:
        raise click.ClickException('%s of %s tasks were not completed successfully.' % (failed, successful + failed))

    return 0


def _log_setup():
    _LOG.debug('Loaded datacube_stats %s from %s.', datacube_stats.__version__, datacube_stats.__path__)
    _LOG.debug('Running against datacube-core %s from %s', datacube.__version__, datacube.__path__)


class StatsApp(object):  # pylint: disable=too-many-instance-attributes
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

        #: Define filter product to accept all derive product attributes
        self.filter_product = None

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

        self.global_attributes = None
        self.var_attributes = None

        #: A function to process the result of a complated task
        #: Takes a single argument of the task result
        self.process_completed = None

        self.queue_size = 50

    @classmethod
    def from_configuration_file(cls, config, index=None, tile_indexes=None, output_location=None, year=None):
        """
        Create a StatsApp to run a processing job, based on a configuration file

        :param config: dictionary based configuration
        :param index: open database connection
        :param tile_indexes: list of tiles for a gridded job. (useful for debugging)
        :return: read to run StatsApp
        """
        input_region = config.get('input_region')
        if tile_indexes and not input_region:
            input_region = {'tiles': tile_indexes}

        if year is not None:
            if 'date_ranges' not in config:
                config['date_ranges'] = {}

            config['date_ranges']['start_date'] = '{}-01-01'.format(year)
            config['date_ranges']['end_date'] = '{}-01-01'.format(year + 1)

        stats_app = cls()
        stats_app.index = index
        stats_app.config_file = config
        stats_app.storage = config['storage']
        stats_app.sources = config['sources']
        stats_app.output_product_specs = config['output_products']
        # Write files to current directory if not set in config or command line
        stats_app.location = output_location or config.get('location', '')
        stats_app.computation = config.get('computation', {})
        stats_app.date_ranges = _configure_date_ranges(index, config)
        stats_app.filter_product = config.get('filter_product', {})
        stats_app.task_generator = select_task_generator(input_region, stats_app.storage, stats_app.filter_product)
        stats_app.output_driver = _prepare_output_driver(stats_app.storage)
        stats_app.global_attributes = config.get('global_attributes', {})
        stats_app.var_attributes = config.get('var_attributes', {})
        stats_app.process_completed = None

        return stats_app

    def validate(self):
        """Check StatsApp is correctly configured and raise an error if errors are found."""
        self._ensure_unique_output_product_names()
        self._check_consistent_measurements()

        assert callable(self.task_generator)
        assert callable(self.output_driver)
        assert hasattr(self.output_driver, 'open_output_files')
        assert hasattr(self.output_driver, 'write_data')
        assert self.process_completed is None or callable(self.process_completed)

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

    def run(self, runner, task_file=None):
        if task_file:
            tasks = unpickle_stream(task_file)
        else:
            tasks = self.generate_tasks(self.configure_outputs())

        app_info = _get_app_metadata(self.config_file)

        output_driver = partial(self.output_driver,
                                output_path=self.location,
                                app_info=app_info,
                                storage=self.storage,
                                global_attributes=self.global_attributes,
                                var_attributes=self.var_attributes)
        task_runner = partial(execute_task,
                              output_driver=output_driver,
                              chunking=self.computation.get('chunking', {}))

        return runner(None,  # digitalearthau.runners.model.TaskDescription,
                      tasks,
                      task_runner,
                      self.process_completed)

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
        is_iterative = all(op.is_iterative() for op in output_products.values())

        for task in self.task_generator(index=self.index, date_ranges=self.date_ranges,
                                        sources_spec=self.sources):
            task.output_products = output_products
            task.is_iterative = is_iterative
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

        stats_metadata = _get_stats_metadata(self.config_file)

        for output_spec in self.output_product_specs:
            output_products[output_spec['name']] = OutputProduct.from_json_definition(
                metadata_type=metadata_type,
                input_measurements=measurements,
                storage=self.storage,
                definition=output_spec,
                stats_metadata=stats_metadata)

        # TODO: Write the output product to disk somewhere

        return output_products

    def __str__(self):
        return "StatsApp: sources=({}), output_driver={}, output_products=({})".format(
            ', '.join(source['product'] for source in self.sources),
            self.ouput_driver,
            ', '.join(out_spec['name'] for out_spec in self.output_product_specs)
        )

    def __repr__(self):
        return str(self)


class StatsProcessingException(Exception):
    pass


def geometry_mask(geoms, geobox, all_touched=False, invert=False):
    return rasterio.features.geometry_mask([geom.to_crs(geobox.crs) for geom in geoms],
                                           out_shape=geobox.shape,
                                           transform=geobox.affine,
                                           all_touched=all_touched,
                                           invert=invert)


def execute_task(task, output_driver, chunking):
    """
    Load data, run the statistical operations and write results out to the filesystem.

    :param datacube_stats.models.StatsTask task:
    :type output_driver: OutputDriver
    :param chunking: dict of dimension sizes to chunk the computation by
    """
    timer = MultiTimer().start('total')
    datacube.set_options(reproject_threads=1)

    process_chunk = load_process_save_chunk_iteratively if task.is_iterative else load_process_save_chunk

    try:
        with output_driver(task=task) as output_files:
            # currently for polygons process will load entirely
            if len(chunking) == 0:
                chunking = {'x': task.sample_tile.shape[2], 'y': task.sample_tile.shape[1]}
            for sub_tile_slice in tile_iter(task.sample_tile, chunking):
                process_chunk(output_files, sub_tile_slice, task, timer)
    except OutputFileAlreadyExists as e:
        _LOG.warning(e)
    except Exception as e:
        _LOG.error("Error processing task: %s", task)
        raise StatsProcessingException("Error processing task: %s" % task)

    timer.pause('total')
    _LOG.info('Completed %s %s task with %s data sources. Processing took: %s', task.tile_index,
              [d.strftime('%Y-%m-%d') for d in task.time_period], task.data_sources_length(), timer)
    return task


def load_process_save_chunk_iteratively(output_files, chunk, task, timer):
    procs = [(stat.make_iterative_proc(), name, stat) for name, stat in task.output_products.items()]

    def update(ds):
        for proc, name, _ in procs:
            with timer.time(name):
                proc(ds)

    def save(name, ds):
        for var_name, var in ds.data_vars.items():
            output_files.write_data(name, var_name, chunk, var.values)

    for ds in load_data_lazy(chunk, task.sources, timer=timer):
        update(ds)

    with timer.time('writing_data'):
        for proc, name, stat in procs:
            save(name, cast_back(proc(), stat.data_measurements))


def load_process_save_chunk(output_files, chunk, task, timer):
    try:
        with timer.time('loading_data'):
            data = load_data(chunk, task.sources)

            # mask as per geometry now
            if task.geom_feat:
                geom = Geometry(task.geom_feat, CRS(task.crs_txt))
                data = data.where(geometry_mask([geom], data.geobox, invert=True))

        last_idx = len(task.output_products) - 1
        for idx, (prod_name, stat) in enumerate(task.output_products.items()):
            _LOG.info("Computing %s in tile %s %s. Current timing: %s",
                      prod_name, task.tile_index, chunk, timer)

            measurements = stat.data_measurements
            with timer.time(prod_name):
                result = stat.compute(data)

                if idx == last_idx:  # make sure input data is released early
                    del data

                # restore nodata values back
                result = cast_back(result, measurements)

            # For each of the data variables, shove this chunk into the output results
            with timer.time('writing_data'):
                for var_name, var in result.data_vars.items():  # TODO: Move this loop into output_files
                    output_files.write_data(prod_name, var_name, chunk, var.values)

    except EmptyChunkException:
        _LOG.debug('Error: No data returned while loading %s for %s. May have all been masked',
                   chunk, task)


class EmptyChunkException(Exception):
    pass


def load_data_lazy(sub_tile_slice, sources, reverse=False, timer=None):
    def by_time(ds):
        return ds.time.values[0]

    data = [load_masked_data_lazy(sub_tile_slice, source, reverse=reverse, src_idx=idx, timer=timer)
            for idx, source in enumerate(sources)]

    if len(data) == 1:
        return data[0]

    return sorted_interleave(*data, key=by_time, reverse=reverse)


def load_data(sub_tile_slice, sources):
    """
    Load a masked chunk of data from the datacube, based on a specification and list of datasets in `sources`.

    :param sub_tile_slice: A portion of a tile, tuple coordinates
    :param sources: a dictionary containing `data`, `spec` and `masks`
    :return: :class:`xarray.Dataset` containing loaded data. Will be indexed and sorted by time.
    """
    datasets = [load_masked_data(sub_tile_slice, source_prod)
                for source_prod in sources]  # list of datasets
    datasets = _mark_source_idx(datasets)
    datasets = _remove_emptys(datasets)
    if len(datasets) == 0:
        raise EmptyChunkException()

    # TODO: Add check for compatible data variable attributes
    # flags_definition between pq products is different and is silently dropped
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


def load_masked_tile_lazy(tile, masks,
                          mask_nodata=False,
                          mask_inplace=False,
                          reverse=True,
                          src_idx=None,
                          timer=None,
                          **kwargs):
    """Given data tile and an optional list of masks load data and masks apply
    masks to data and return one time slice at a time.


    tile -- Tile object for main data
    masks -- [(Tile, flags, load_args)] list of triplets describing mask to be applied to data.
             Tile -- tile objects describing where mask data files are
             flags -- dictionary of flags to be checked
             load_args - dictionary of load parameters (e.g. fuse_func, measurements, etc.)

    mask_nodata  -- Convert data to float32 replacing nodata values with nan
    mask_inplace -- Apply mask without conversion to float
    reverse      -- Return data earliest observation first
    src_idx      -- If set adds extra axis called source with supplied value
    timer        -- Optionally track time


    Returns an iterator of DataFrames one time-slice at a time

    """

    ii = range(tile.shape[0])
    if reverse:
        ii = ii[::-1]

    def load_slice(i):
        loc = [slice(i, i + 1), slice(None), slice(None)]
        d = GridWorkflow.load(tile[loc], **kwargs)

        if mask_nodata:
            d = sensible_mask_invalid_data(d)

        # Load all masks and combine them all into one
        mask = None
        for m_tile, flags, load_args in masks:
            m = GridWorkflow.load(m_tile[loc], **load_args)
            m, *other = m.data_vars.values()
            m = make_mask(m, **flags)

            if mask is None:
                mask = m
            else:
                mask &= m

        if mask is not None:
            # Apply mask in place if asked or if we already performed
            # conversion to float32, this avoids reallocation of memory and
            # hence increases the largest data set size one can load without
            # running out of memory
            if mask_inplace or mask_nodata:
                d = sensible_where_inplace(d, mask)
            else:
                d = sensible_where(d, mask)

        if src_idx is not None:
            d.coords['source'] = ('time', np.repeat(src_idx, d.time.size))

        return d

    extract = wrap_in_timer(load_slice, timer, 'loading_data')

    for i in ii:
        yield extract(i)


def load_masked_data_lazy(sub_tile_slice, source_prod, reverse=False, src_idx=None, timer=None):
    data_fuse_func = import_function(source_prod['spec']['fuse_func']) if 'fuse_func' in source_prod['spec'] else None
    data_tile = source_prod['data'][sub_tile_slice]
    data_measurements = source_prod['spec'].get('measurements')

    mask_nodata = source_prod['spec'].get('mask_nodata', True)
    mask_inplace = source_prod['spec'].get('mask_inplace', False)
    masks = []

    if 'masks' in source_prod and 'masks' in source_prod['spec']:
        for mask_spec, mask_tile in zip(source_prod['spec']['masks'], source_prod['masks']):
            flags = mask_spec['flags']
            mask_fuse_func = import_function(mask_spec['fuse_func']) if 'fuse_func' in mask_spec else None
            opts = dict(skip_broken_datasets=True,
                        fuse_func=mask_fuse_func,
                        measurements=[mask_spec['measurement']])

            masks.append((mask_tile[sub_tile_slice], flags, opts))

    return load_masked_tile_lazy(data_tile,
                                 masks,
                                 mask_nodata=mask_nodata,
                                 mask_inplace=mask_inplace,
                                 reverse=reverse,
                                 src_idx=src_idx,
                                 timer=timer,
                                 fuse_func=data_fuse_func,
                                 measurements=data_measurements,
                                 skip_broken_datasets=True)


def load_masked_data(sub_tile_slice, source_prod):
    data_fuse_func = import_function(source_prod['spec']['fuse_func']) if 'fuse_func' in source_prod['spec'] else None
    data = GridWorkflow.load(source_prod['data'][sub_tile_slice],
                             measurements=source_prod['spec'].get('measurements'),
                             fuse_func=data_fuse_func,
                             skip_broken_datasets=True)

    mask_inplace = source_prod['spec'].get('mask_inplace', False)
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
            mask_fuse_func = import_function(mask_spec['fuse_func']) if 'fuse_func' in mask_spec else None
            mask = GridWorkflow.load(mask_tile[sub_tile_slice],
                                     measurements=[mask_spec['measurement']],
                                     fuse_func=mask_fuse_func,
                                     skip_broken_datasets=True)[mask_spec['measurement']]
            mask = make_mask(mask, **mask_spec['flags'])
            if mask_inplace:
                data = sensible_where_inplace(data, mask)
            else:
                data = sensible_where(data, mask)
            del mask
    return data


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
                'repo_url': 'https://github.com/GeoscienceAustralia/agdc_statistics.git',
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
    elif date_ranges['type'] == 'find_daily_data':
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
