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
from datacube.api import make_mask, GridWorkflow
from datacube.ui import click as ui
from datacube.utils import read_documents, import_function
from datacube.utils.geometry import CRS, Geometry, GeoBox
from datacube_stats.models import OutputProduct
from datacube_stats.output_drivers import OUTPUT_DRIVERS, OutputFileAlreadyExists, get_driver_by_name, \
    NoSuchOutputDriver
from datacube_stats.statistics import StatsConfigurationError, STATS
from datacube_stats.utils import cast_back, pickle_stream, unpickle_stream, _find_periods_with_data
from datacube_stats.utils import tile_iter, sensible_mask_invalid_data, sensible_where, sensible_where_inplace
from datacube_stats.utils.dates import date_sequence
from digitalearthau.qsub import with_qsub_runner, TaskRunner
from digitalearthau.runners.model import TaskDescription, DefaultJobParameters
from .utils.timer import MultiTimer, wrap_in_timer
from .utils import sorted_interleave, Slice, prettier_slice
from .tasks import select_task_generator
from .schema import stats_schema
from .models import StatsTask, DataSource
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
@click.option('--save-tasks', type=click.Path(exists=False, writable=True, dir_okay=False))
@click.option('--load-tasks', type=click.Path(exists=True, readable=True))
@click.option('--tile-index', nargs=2, type=int, help='Override input_region specified in configuration with a '
                                                      'single tile_index specified as [X] [Y]')
@click.option('--tile-index-file',
              type=click.Path(exists=True, readable=True, dir_okay=False),
              help="A file consisting of tile indexes specified as [X] [Y] per line")
@click.option('--output-location', help='Override output location in configuration file')
@click.option('--year', type=int, help='Override time period in configuration file')
@click.option('--task-slice', type=Slice(),
              help="The subset of tasks to perform, using Python's slice syntax.")
@click.option('--batch', type=int,
              help="The number of batch jobs to launch using PBS and the serial executor.")
@click.option('--list-statistics', is_flag=True, callback=list_statistics, expose_value=False)
@ui.global_cli_options
@with_qsub_runner()
@click.option('--version', is_flag=True, callback=_print_version,
              expose_value=False, is_eager=True)
@ui.pass_index(app_name='datacube-stats')
def main(index, stats_config_file, qsub, runner, save_tasks, load_tasks,
         tile_index, tile_index_file, output_location, year, task_slice, batch):

    try:
        _log_setup()

        if qsub is not None and batch is not None:
            for i in range(batch):
                child = qsub.clone()
                child.reset_internal_args()
                child.add_internal_args('--task-slice', '{}::{}'.format(i, batch))
                click.echo(repr(child))
                exit_code, _ = child(auto=True, auto_clean=[('--batch', 1)])
                if exit_code != 0:
                    return exit_code
            return 0

        elif qsub is not None:
            # TODO: verify config before calling qsub submit
            click.echo(repr(qsub))
            exit_code, _ = qsub(auto=True)
            return exit_code

        timer = MultiTimer().start('main')

        config = normalize_config(read_config(stats_config_file),
                                  tile_index, tile_index_file, year, output_location)

        app = StatsApp(config, index)

        if save_tasks:
            app.save_tasks_to_file(save_tasks)
            failed = 0
        elif load_tasks:
            successful, failed = app.run(runner, task_file=load_tasks, task_slice=task_slice)
        else:
            successful, failed = app.run(runner, task_slice=task_slice)

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


class StatsApp(object):  # pylint: disable=too-many-instance-attributes
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
        self.date_ranges = _configure_date_ranges(index, config)

        #: Generates tasks to compute statistics. These tasks should be :class:`StatsTask` objects
        #: and will define spatial and temporal boundaries, as well as statistical operations to be run.
        self.task_generator = select_task_generator(config['input_region'],
                                                    self.storage, self.filter_product)

        #: A class which knows how to create and write out data to a permanent storage format.
        #: Implements :class:`.output_drivers.OutputDriver`.
        self.output_driver = _prepare_output_driver(self.storage)

        #: An open database connection
        self.index = index

        self.global_attributes = config['global_attributes']
        self.var_attributes = config['var_attributes']

        self.validate()

    def validate(self):
        """Check StatsApp is correctly configured and raise an error if errors are found."""
        self._ensure_unique_output_product_names()
        self._check_consistent_measurements()

        assert callable(self.task_generator)
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

    def _log_config(self):
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

    def run(self, runner, task_file=None, task_slice=None):
        self._log_config()

        if task_file:
            tasks = unpickle_stream(task_file)
        else:
            tasks = self.generate_tasks(self.configure_outputs())

        return self.run_tasks(tasks, runner=runner, task_slice=task_slice)

    def save_tasks_to_file(self, filename):
        _LOG.debug('Saving tasks to %s.', filename)
        output_products = self.configure_outputs()

        tasks = self.generate_tasks(output_products)
        num_saved = pickle_stream(tasks, filename)
        _LOG.debug('Successfully saved %s tasks to %s.', num_saved, filename)

    def generate_tasks(self,
                       output_products: Dict[str, OutputProduct] = None,
                       metadata_type='eo') -> Iterator[StatsTask]:
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
        if output_products is None:
            output_products = self.configure_outputs(metadata_type)

        is_iterative = all(op.is_iterative() for op in output_products.values())

        for task in self.task_generator(index=self.index, date_ranges=self.date_ranges,
                                        sources_spec=self.sources):
            task.output_products = output_products
            task.is_iterative = is_iterative
            yield task

    def configure_outputs(self, metadata_type='eo') -> Dict[str, OutputProduct]:
        """
        Return dict mapping Output Product Name<->Output Product

        StatProduct describes the structure and how to compute the output product.
        """
        _LOG.debug('Creating output products')

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
            self.output_driver,
            ', '.join(out_spec['name'] for out_spec in self.output_product_specs)
        )

    def __repr__(self):
        return str(self)


class StatsProcessingException(Exception):
    pass


def geometry_mask(geoms: Iterable[Geometry], geobox: GeoBox,
                  all_touched=False, invert=False) -> np.ndarray:
    return rasterio.features.geometry_mask([geom.to_crs(geobox.crs) for geom in geoms],
                                           out_shape=geobox.shape,
                                           transform=geobox.affine,
                                           all_touched=all_touched,
                                           invert=invert)


def execute_task(task: StatsTask, output_driver, chunking) -> StatsTask:
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
        _LOG.warning(str(e))
    except OutputDriverResult as e:
        # was run interactively
        # re-raise result to be caught again by StatsApp.execute_task
        raise e
    except Exception as e:
        _LOG.error("Error processing task: %s", task)
        raise StatsProcessingException("Error processing task: %s" % task)

    timer.pause('total')
    _LOG.debug('Completed %s %s task with %s data sources; %s', task.tile_index,
               [d.strftime('%Y-%m-%d') for d in task.time_period], task.data_sources_length(), timer)
    return task


def load_process_save_chunk_iteratively(output_files: OutputDriver,
                                        chunk: Tuple[slice, slice, slice],
                                        task: StatsTask,
                                        timer: MultiTimer):
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


def load_process_save_chunk(output_files: OutputDriver,
                            chunk: Tuple[slice, slice, slice],
                            task: StatsTask, timer: MultiTimer):
    try:
        with timer.time('loading_data'):
            data = load_data(chunk, task.sources)
            # mask as per geometry now
            if task.geom_feat:
                geom = Geometry(task.geom_feat, CRS(task.crs_txt))
                data = data.where(geometry_mask([geom], data.geobox, invert=True))
            # pylint: disable=protected-access
            if output_files._driver_name == 'None':
                output_files.get_source(chunk, data)

        last_idx = len(task.output_products) - 1
        for idx, (prod_name, stat) in enumerate(task.output_products.items()):
            _LOG.debug("Computing %s in tile %s %s; %s",
                       prod_name, task.tile_index,
                       "({})".format(", ".join(prettier_slice(c) for c in chunk)),
                       timer)

            measurements = stat.data_measurements

            with timer.time(prod_name):
                result = stat.compute(data)

                if idx == last_idx:  # make sure input data is released early
                    del data

                # restore nodata values back
                result = cast_back(result, measurements)

            # For each of the data variables, shove this chunk into the output results
            with timer.time('writing_data'):
                output_files.write_chunk(prod_name, chunk, result)

    except EmptyChunkException:
        _LOG.debug('Error: No data returned while loading %s for %s. May have all been masked',
                   chunk, task)


class EmptyChunkException(Exception):
    pass


def load_data_lazy(sub_tile_slice, sources, reverse=False, timer=None):
    def by_time(ds):
        return ds.time.values[0]

    data = [load_masked_data_lazy(sub_tile_slice, source, reverse=reverse, src_idx=source.source_index, timer=timer)
            for source in sources]

    if len(data) == 1:
        return data[0]

    return sorted_interleave(*data, key=by_time, reverse=reverse)


def load_data(sub_tile_slice: Tuple[slice, slice, slice],
              sources: Iterable[DataSource]) -> xarray.Dataset:
    """
    Load a masked chunk of data from the datacube, based on a specification and list of datasets in `sources`.

    :param sub_tile_slice: A portion of a tile, tuple coordinates
    :param sources: a dictionary containing `data`, `spec` and `masks`
    :return: :class:`xarray.Dataset` containing loaded data. Will be indexed and sorted by time.
    """
    datasets = [load_masked_data(sub_tile_slice, source_prod)
                for source_prod in sources]  # list of datasets

    datasets = _remove_emptys(datasets)
    if len(datasets) == 0:
        raise EmptyChunkException()

    # TODO: Add check for compatible data variable attributes
    # flags_definition between pq products is different and is silently dropped
    ds = xarray.concat(datasets, dim='time')  # Copies all the data
    if len(ds.time) == 0:
        raise EmptyChunkException()

    # sort along time dim
    return ds.sortby('time')  # Copies all the data again


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


def load_masked_data_lazy(sub_tile_slice: Tuple[slice, slice, slice],
                          source_prod: DataSource,
                          reverse=False, src_idx=None, timer=None) -> xarray.Dataset:
    data_fuse_func = import_function(source_prod.spec['fuse_func']) if 'fuse_func' in source_prod.spec else None
    data_tile = source_prod.data[sub_tile_slice]
    data_measurements = source_prod.spec.get('measurements')

    mask_nodata = source_prod.spec.get('mask_nodata', True)
    mask_inplace = source_prod.spec.get('mask_inplace', False)
    masks = []

    if 'masks' in source_prod.spec:
        for mask_spec, mask_tile in zip(source_prod.spec['masks'], source_prod.masks):
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


def load_masked_data(sub_tile_slice: Tuple[slice, slice, slice],
                     source_prod: DataSource) -> xarray.Dataset:
    data_fuse_func = import_function(source_prod.spec['fuse_func']) if 'fuse_func' in source_prod.spec else None
    data = GridWorkflow.load(source_prod.data[sub_tile_slice],
                             measurements=source_prod.spec.get('measurements'),
                             fuse_func=data_fuse_func,
                             skip_broken_datasets=True)

    mask_inplace = source_prod.spec.get('mask_inplace', False)
    mask_nodata = source_prod.spec.get('mask_nodata', True)

    if mask_nodata:
        data = sensible_mask_invalid_data(data)

    # if all NaN
    completely_empty = all(ds for ds in xarray.ufuncs.isnan(data).all().data_vars.values())
    if completely_empty:
        # Discard empty slice
        return None

    if 'masks' in source_prod.spec:
        for mask_spec, mask_tile in zip(source_prod.spec['masks'], source_prod.masks):
            if mask_tile is None:
                # Discard data due to no mask data
                return None
            mask_fuse_func = import_function(mask_spec['fuse_func']) if 'fuse_func' in mask_spec else None
            mask = GridWorkflow.load(mask_tile[sub_tile_slice],
                                     measurements=[mask_spec['measurement']],
                                     fuse_func=mask_fuse_func,
                                     skip_broken_datasets=True)[mask_spec['measurement']]
            if mask_spec.get('flags') is not None:
                mask = make_mask(mask, **mask_spec['flags'])
            elif mask_spec.get('less_than') is not None:
                less_than = float(mask_spec['less_than'])
                mask = mask < less_than
            elif mask_spec.get('greater_than') is not None:
                greater_than = float(mask_spec['greater_than'])
                mask = mask > greater_than

            if mask_inplace:
                data = sensible_where_inplace(data, mask)
            else:
                data = sensible_where(data, mask)
            del mask

    if source_prod.source_index is not None:
        data.coords['source'] = ('time', np.repeat(source_prod.source_index, data.time.size))

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

    elif date_ranges.get('type') == 'find_daily_data':
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
