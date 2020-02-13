"""
Create statistical summaries command.

This command is run as ``datacube-stats``, all operation are driven by a configuration file.

"""
import copy
import logging
import sys
import yaml

from functools import partial
from pathlib import Path
from datetime import datetime
import os
from os import path

import click
import numpy as np
import pandas as pd
from datacube import Datacube
from datacube.ui import click as ui
from datacube.utils import read_documents
from .utils.dates import date_sequence
from datacube.virtual import construct
import json
import hashlib
from shapely.geometry import Polygon, mapping
from datacube.utils.geometry import CRS, Geometry
import pickle
from queue import Queue
from time import sleep
import threading

_LOG = logging.getLogger(__name__)

DEFAULT_TILE_INDEX_FILE = 'landsat_tiles.txt'
TILE_SIZE = 60000.


def generate_task_queue(product_name, input_region, date_range):
    if 'tiles' not in input_region:
        _LOG.debug('tiles only')
    sensor_source = input_region.get('sensor_source')
    if sensor_source == 'sentile2':
        stride = 1
    elif sensor_source == 'landsat':
        stride = 2
    else:
        _LOG.error('Dont understand the sensor source %s', sensor_source)

    task_queue = {}
    for date in date_range:
        for tile in input_region['tiles']:
            _LOG.debug("query tile date %s", (tile, date))
            tile_geojson = convert_tile_to_geojson(tile, stride, TILE_SIZE)
            query_unit = QueryUnit(product_name, date, tile_geojson)
            query_string = {}
            query_string['geopolygon'] = tile_geojson
            query_string['time'] = date
            # query by tile index implies crs
            query_string['crs'] = 'EPSG:3577'
            task_queue[query_unit.hash()] = {'query_string': query_string, 'tile': tile}
    return task_queue


def local_task_generator(client, query_workers, datacube_config, stats_config, tasks):
    for product in stats_config['output_products']:
        product_name = product['name']
        recipe = dict(product['recipe'])
        virtual_product = construct(**recipe)
        virtual_datasets_with_config = generate_virtual_datasets(client, query_workers, datacube_config,
                                                                 virtual_product, tasks[product_name])
        yield ProductWithDef(product, virtual_product, virtual_datasets_with_config)


def generate_virtual_datasets(client, query_workers, datacube_config, virtual_product, tasks):
    def query_group(datacube_config, kwargs):
        dc = Datacube(config=datacube_config)
        kwargs['geopolygon'] = Geometry(kwargs['geopolygon'], CRS(kwargs['crs']))
        kwargs.pop('crs', None)
        try:
            datasets = virtual_product.query(dc, **kwargs)
            grouped = virtual_product.group(datasets, **kwargs)
        except Exception as e:
            _LOG.warning('something weird happend %s', e)
            return None
        else:
            return grouped

    task_priority = 0
    for key, value in tasks.items():
        query_string = value.get('query_string')
        tile = value.get('tile')
        _LOG.debug("query string %s", query_string)
        grouped = client.submit(query_group, datacube_config, query_string,
                                key='query-'+key,
                                priority=task_priority, workers=query_workers)
        task_priority -= 1
        yield VirtualDatasetsWithConfig(task_priority, key, grouped, tile, query_string['time'])


TASK_GENERATOR_REG = {
        'IDNO': local_task_generator
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

def ls5_on_1ym(dataset):
    LS5_START_AGAIN = datetime(2003, 1, 1)
    LS5_STOP_DATE = datetime(1999, 12, 31)
    LS5_STOP_AGAIN = datetime(2011, 12, 31)
    return dataset.center_time <= LS5_STOP_DATE or (dataset.center_time >= LS5_START_AGAIN
                                                    and dataset.center_time <= LS5_STOP_AGAIN)


# pylint: disable=broad-except
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
@click.command(name='stats_product')
@click.argument('stats_config_file', type=str, default='config.yaml', metavar='STATS_CONFIG_FILE')
@click.option('--task-generator',  type=str, help='Select a task generator or...not', default='IDNO')
@click.option('--scheduler-file',  type=str, help='The Json file of the scheduler default=./scheduler.json',
              default='./scheduler.json')
@click.option('--query-workers',  type=int, help='How many workers to perform query default=2', default=2)
@click.option('--queue-size',  type=int, help='The queue size to apply backpressure on finishing tasks '
                                              'defualt=2*query-workers')
@click.option('--tile-index', nargs=2, type=int, help='Override input_region specified in configuration with a '
                                                      'single tile_index specified as [X] [Y]')
@click.option('--tile-index-file',
              type=click.Path(exists=True, readable=True, dir_okay=False),
              help="A file consisting of tile indexes specified as [X] [Y] per line")
@click.option('--output-location', help='Override output location in configuration file')
@click.option('--year', type=int, help='Override time period in configuration file')
@ui.global_cli_options
@ui.pass_config
def main(datacube_config, stats_config_file, task_generator, scheduler_file, query_workers, queue_size,
         tile_index, tile_index_file, output_location, year):

    try:
        stats_config = normalize_config(read_config(stats_config_file), output_location)
        stats_config = normalize_time_range(stats_config, year)
        stats_config = normalize_space_range(stats_config, tile_index, tile_index_file)
        task_generator = TASK_GENERATOR_REG.get(task_generator)

        output_config = stats_config.copy()
        output_config.pop('output_products', None)
        output_config.pop('input_region', None)
        output_config.pop('date_ranges', None)

        # play nice to the database
        query_workers = min(query_workers, 10)
        if queue_size is None:
            queue_size = 2 * query_workers

        from dask.distributed import Client
        client = Client(scheduler_file=scheduler_file)
        num_cores = 0
        num_workers = 1
        while True:
            client.wait_for_workers(num_workers)
            for cores in client.ncores().values():
                _LOG.debug("current cores %s", cores)
                num_cores += cores
            if num_cores >= queue_size:
                break
            else:
                num_workers += 1
                sleep(5)

        _LOG.debug('Run on cluster with workers %s', client.ncores())
        version_info = client.get_versions(check=True)
        _LOG.debug('Cluster version info %s', version_info)

        i = query_workers
        query_worker_list = []
        for worker, detail in client.scheduler_info()['workers'].items():
            if i <= 0:
                break
            if 'query' in detail.get('name', ''):
                query_worker_list.append(worker)
                i -= int(detail.get('ncores', 1))

        _LOG.debug('Workers to perform query %s', query_worker_list)

        checkpoint_path = stats_config.get('location') + '/checkpointing'
        checkpoint_file = checkpoint_path + '/checkpointing.pkl'
        if not path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
            restart = False
        elif not path.exists(checkpoint_file):
            restart = False
        else:
            restart = True
        compute_checking = ComputeCheck(queue_size=queue_size, restart=restart, checkpoint=checkpoint_file)

        if compute_checking.restored != {}:
            tasks = copy.deepcopy(compute_checking.restored)
        else:
            input_region = stats_config['input_region']
            date_range = stats_config['date_ranges']
            tasks = {}
            product_name = ''
            for product in stats_config['output_products']:
                if product_name == product['name']:
                    _LOG.error("More than one product with the same name %s", product_name)
                product_name = product['name']
                tasks[product_name] = generate_task_queue(product_name, input_region, date_range)
            compute_checking.restored = copy.deepcopy(tasks)

        products = task_generator(client, query_worker_list, datacube_config, stats_config, tasks)
        for product in products:
            metadata_type = retrieve_metadata_type(datacube_config)
            _LOG.debug("metadata type %s", metadata_type)
            output_config['output_product'] = product.product_definition
            for virtual_datasets_with_config in product.datasets:
                compute_checking.inc()
                output_config['input_region'] = virtual_datasets_with_config.input_region
                output_config['date_range'] = virtual_datasets_with_config.date_range
                app = StatsApp(output_config, product.product)
                datasets = virtual_datasets_with_config.datasets
                _LOG.debug("output product %s", datasets)
                output = client.submit(app.generate_products, metadata_type, datasets,
                                       priority=virtual_datasets_with_config.task_priority,
                                       key='output-' + virtual_datasets_with_config.key,
                                       workers=query_worker_list)

                future = client.submit(app.load_save, output,
                                       priority=virtual_datasets_with_config.task_priority,
                                       key='save-' + virtual_datasets_with_config.key,
                                       workers=query_worker_list)

                future.add_done_callback(compute_checking.future_done)
                compute_checking.set(virtual_datasets_with_config.key, future)
                _LOG.debug("submit compute future %s", future)
        _LOG.debug("waiting for the result %s", compute_checking.result)
        client.gather(list(compute_checking.result.values()))
        _LOG.debug("computation to be done %s", compute_checking.restored)
        os.remove(compute_checking.checkpoint)

        def shutdown(dask_scheduler=None):
            from dask.distributed import utils
            dask_scheduler.finished()
            with utils.ignoring(RuntimeError):
                dask_scheduler.close(close_workers=True)

        client.run_on_scheduler(shutdown)

    except Exception as e:
        _LOG.error(e)
        sys.exit(1)

    return 0


def read_config(config_file):
    _, stats_config = next(read_documents(config_file))
    return stats_config


def normalize_time_range(stats_config, year=None):

    if 'date_ranges' not in stats_config:
        stats_config['date_ranges'] = {}
        stats_config['date_ranges']['start_date'] = '1987-01-01'
        stats_config['date_ranges']['end_date'] = datetime.now().date()

    # if year is not none
    # then override the date_range
    # deal with the inconsistancy between date_sequence not inluding the end_date
    # and the datacube query including the end_date
    if year is not None:
        start_date = '{}-01-01'.format(year)
        end_date = '{}-01-01'.format(year+1)
        stats_config['date_ranges'] = date_sequence(start=pd.to_datetime(start_date),
                                                    end=pd.to_datetime(end_date),
                                                    stats_duration='1y',
                                                    step_size='1y')
        return stats_config

    if 'stats_duration' in stats_config['date_ranges'] and 'step_size' in stats_config['date_ranges']:
        stats_config['date_ranges'] = date_sequence(start=pd.to_datetime(stats_config['date_ranges']['start_date']),
                                                    end=pd.to_datetime(stats_config['date_ranges']['end_date']),
                                                    stats_duration=stats_config['date_ranges']['stats_duration'],
                                                    step_size=stats_config['date_ranges']['step_size'])
    else:
        stats_config['date_ranges'] = [(pd.to_datetime(stats_config['date_ranges']['start_date']).date(),
                                        pd.to_datetime(stats_config['date_ranges']['end_date']).date())]
    return stats_config


def normalize_space_range(stats_config, tile_index=None, tile_index_file=None):
    if tile_index is not None and len(tile_index) == 0:
        tile_index = None

    # if tile indices or indices file is not none
    # then override the input_region
    tile_indexes = gather_tile_indexes(tile_index, tile_index_file)
    input_region = stats_config.get('input_region')
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

    stats_config['input_region'] = input_region
    return stats_config


def normalize_config(stats_config, output_location):
    # Write files to current directory if not set in stats_config or command line
    stats_config['location'] = output_location or stats_config.get('location', '')
    stats_config['computation'] = stats_config.get('computation', {})
    stats_config['global_attributes'] = stats_config.get('global_attributes', {})
    stats_config['var_attributes'] = stats_config.get('var_attributes', {})
    return stats_config


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


def convert_tile_to_geojson(tile_index, stride=1, tile_size=60000.):
    return mapping(Polygon([(tile_index[0]*tile_size, tile_index[1]*tile_size),
                            ((tile_index[0]+stride)*tile_size, tile_index[1]*tile_size),
                            ((tile_index[0]+stride)*tile_size, (tile_index[1]-stride)*tile_size),
                            (tile_index[0]*tile_size, (tile_index[1]-stride)*tile_size)]))


class QueryUnit():
    def __init__(self, product: str, time: tuple, space: dict):
        if not isinstance(space, dict):
            raise ValueError("space has to be a geojson")
        if not isinstance(time, tuple):
            raise ValueError("time has to be a tuple of (datetime.datetime, datetime.datetime)")
        self.time = time
        self.space = space
        self.product = product

    def hash(self):
        hashable = {'time:': {'start': str(self.time[0]), 'end': str(self.time[1])}, 'product': self.product}
        hashable.update(self.space)
        return hashlib.md5(json.dumps(hashable, sort_keys=True).encode('utf-8')).hexdigest()


class ComputeCheck():
    def __init__(self, queue_size=2, restart=False, checkpoint=None):
        self.checkpoint = checkpoint
        if restart:
            # read in the pickle file of parameters
            with open(checkpoint, 'rb') as f:
                self.restored = pickle.load(f)
        else:
            self.restored = {}
        self.result = {}
        self.output_queue = Queue(maxsize=queue_size)
        self.lock = threading.Lock()

    def inc(self):
        self.output_queue.put(1)

    def set(self, key, result):
        self.result.update({key: result})

    def future_done(self, result):
        if result.done():
            query_hash = result.key.split('-')[1]
            _LOG.debug("future status %s", result.status)
            if result.status == 'finished':
                with self.lock:
                    self.output_queue.get()
                    future = self.result.pop(query_hash, None)
                    for key, value in self.restored.items():
                        query = value.pop(query_hash, None)
                        if query is not None:
                            _LOG.debug("query done %s", query)
                            break
                    with open(self.checkpoint, 'wb') as f:
                        pickle.dump(self.restored, f)
            else:
                _LOG.debug("retry the future %s", result)
                result.retry()
            return 0


class VirtualDatasetsWithConfig():
    def __init__(self, task_priority: int, key, datasets, input_region, date_range):
        self.task_priority = task_priority
        self.key = key
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

    def __init__(self, stats_config, virtual_product):
        """

        Create a StatsApp to run a processing job, based on a configuration dict.
        """
        #: Dictionary containing the configuration
        self.stats_config = stats_config

        #: Description of output file format
        self.storage = stats_config['storage']

        #: List of filenames and statistical methods used, describing what the outputs of the run will be.
        self.output_product_spec = stats_config['output_product']

        #: Base directory to write output files to.
        #: Files may be created in a sub-directory, depending on the configuration of the
        #: :attr:`output_driver`.
        self.location = stats_config['location']

        #: A class which knows how to create and write out data to a permanent storage format.
        #: Implements :class:`.output_drivers.OutputDriver`.
        self.output_driver = _prepare_output_driver(self.storage)

        self.global_attributes = stats_config['global_attributes']
        self.var_attributes = stats_config['var_attributes']

        self.virtual_product = virtual_product

    def _partially_applied_output_driver(self):
        app_info = _get_app_metadata(self.stats_config)

        return partial(self.output_driver,
                       output_path=self.location,
                       app_info=app_info,
                       storage=self.storage,
                       global_attributes=self.global_attributes,
                       var_attributes=self.var_attributes)

    def generate_products(self, metadata_type, virtual_datasets):
        if virtual_datasets is None:
            return None, None
        definition = self.output_product_spec

        extras = dict({'epoch_start': self.stats_config['date_range'][0],
                       'epoch_end': self.stats_config['date_range'][1],
                       'x': self.stats_config['input_region'][0],
                       'y': self.stats_config['input_region'][1]})

        if 'metadata' not in definition:
            definition['metadata'] = {}
        if 'format' not in definition['metadata']:
            definition['metadata']['format'] = {'name': self.output_driver.format_name()}

        from .models import OutputProduct
        output = OutputProduct.from_json_definition(metadata_type=metadata_type,
                                                    virtual_datasets=virtual_datasets,
                                                    virtual_product=self.virtual_product,
                                                    storage=self.storage,
                                                    definition=definition,
                                                    extras=extras)
        if self.stats_config.get('computation') is not None:
            dask_chunks = self.stats_config['computation'].get('chunking')
        results = output.compute(output.datasets, dask_chunks=dask_chunks)
        return output, results

    def load(self, args):
        output, results = args
        from distributed import secede, rejoin
        secede()
        try:
            return output, results.load()
        except Exception as e:
            _LOG.error("some dask error I dont know %s", e)
        return output, None

    def save(self, args):
        output, results = args
        if results is None:
            _LOG.error("fail in loading data")
            return -1
        output_driver = self._partially_applied_output_driver()
        try:
            with output_driver(output_product=output) as output_file:
                    output_file.write_data(results)
        except Exception as e:
            _LOG.warning("Check this %s", e)
            return -1
        return 0

    def load_save(self, args):
        output, results = args
        if results is None:
            _LOG.warning("No data to load")
            return -1
        output_driver = self._partially_applied_output_driver()
        from distributed import secede, rejoin
        secede()
        results.load()
        rejoin()
        try:
            with output_driver(output_product=output) as output_file:
                output_file.write_data(results)
        except Exception as e:
            _LOG.warning("Check this %s", e)
            return -1
        return 0

    def __str__(self):
        return "StatsApp:  output_driver={}, output_products=({})".format(
            self.output_driver,
            self.output_product_spec
        )

    def __repr__(self):
        return str(self)


def retrieve_metadata_type(datacube_config, metadata_type='eo'):
    """
    return metadata_type with the input string
    """
    dc = Datacube(config=datacube_config)
    return dc.index.metadata_types.get_by_name(metadata_type)


def _get_app_metadata(config_file):
    stats_config = copy.deepcopy(config_file)
    if 'global_attributes' in stats_config:
        del stats_config['global_attributes']
    return {
        'lineage': {
            'algorithm': {
                'name': 'virtual-product',
                'parameters': {'configuration_file': config_file}
            },
        }
    }


def _prepare_output_driver(storage):
    from .output_drivers import OUTPUT_DRIVERS, OutputFileAlreadyExists, get_driver_by_name, \
                                NoSuchOutputDriver
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
