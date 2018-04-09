from functools import partial
from datetime import datetime
from dateutil import tz
from pathlib import Path
from itertools import islice

from datacube_stats.main import _get_stats_metadata, _prepare_output_driver
from datacube_stats.main import _get_app_metadata, execute_task
from datacube_stats.models import OutputProduct
from datacube_stats.statistics import StatsConfigurationError

from digitalearthau.qsub import TaskRunner
from digitalearthau.runners.model import TaskDescription, DefaultJobParameters

from datacube_stats.main import gather_tile_indexes
from datacube_stats.main import _configure_date_ranges, _source_measurement_defs

from ..tasks import select_task_generator
import logging

_LOG = logging.getLogger(__name__)


class Config(object):
    def __init__(self, config=None):
        self.config = config
        return

    def normalize_config(self, tile_index=None, tile_index_file=None,
                         year=None, output_location=None):
        if tile_index is not None and len(tile_index) == 0:
            tile_index = None

        tile_indexes = gather_tile_indexes(tile_index, tile_index_file)

        input_region = self.config.get('input_region')
        if tile_indexes and not input_region:
            input_region = {'tiles': tile_indexes}
        self.config['input_region'] = input_region

        if year is not None:
            if 'date_ranges' not in self.config:
                self.config['date_ranges'] = {}

                self.config['date_ranges']['start_date'] = '{}-01-01'.format(year)
                self.config['date_ranges']['end_date'] = '{}-01-01'.format(year + 1)

        self.config['location'] = output_location or self.config.get('location', '')
        self.config['computation'] = self.config.get('computation', {})
        self.config['global_attributes'] = self.config.get('global_attributes', {})
        self.config['var_attributes'] = self.config.get('var_attributes', {})
        self.config['filter_product'] = self.config.get('filter_product', {})


class TaskProducer(object):
    def __init__(self, dc, config):
        self.index = dc.index
        self.config = config.config
        self.date_ranges = _configure_date_ranges(dc.index, config.config)
        self.task_generator = select_task_generator(self.input_region,
                                                    self.storage, self.filter_product)

    def __getattr__(self, key):
        return self.config[key]

    def validate(self):
        """Check TaskProducer is correctly configured and raise an error if errors are found."""
        self._ensure_unique_output_product_names()
        self._check_consistent_measurements()

        assert callable(self.task_generator)

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
        output_names = [prod['name'] for prod in self.output_products]
        duplicate_names = [x for x in output_names if output_names.count(x) > 1]
        if duplicate_names:
            raise StatsConfigurationError('Output products must all have different names. '
                                          'Duplicates found: %s' % duplicate_names)

    def produce_tasks(self, output_products=None, metadata_type='eo'):
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

    def configure_outputs(self, metadata_type='eo'):
        """
        Return dict mapping Output Product Name<->Output Product

        StatProduct describes the structure and how to compute the output product.
        """
        _LOG.info('Creating output products')

        output_products = {}

        measurements = _source_measurement_defs(self.index, self.sources)

        metadata_type = self.index.metadata_types.get_by_name(metadata_type)

        stats_metadata = _get_stats_metadata(self.config)

        for output_spec in self.output_products:
            output_products[output_spec['name']] = OutputProduct.from_json_definition(
                metadata_type=metadata_type,
                input_measurements=measurements,
                storage=self.storage,
                definition=output_spec,
                stats_metadata=stats_metadata)

        # TODO: Write the output product to disk somewhere

        return output_products


class TaskConsumer(object):
    def __init__(self, config, runner=None):
        self.config = config.config
        self.storage = self.config['storage']
        self.output_driver = _prepare_output_driver(self.storage)
        if runner is not None:
            self.runner = runner
        else:
            self.runner = TaskRunner()

    def __getattr__(self, key):
        return self.config[key]

    def validate(self):
        assert callable(self.output_driver)
        assert hasattr(self.output_driver, 'open_output_files')
        assert hasattr(self.output_driver, 'write_data')

    def consume_tasks(self, tasks, task_slice=None):

        if task_slice is not None:
            tasks = islice(tasks, task_slice.start, task_slice.stop, task_slice.step)

        app_info = _get_app_metadata(self.config)

        output_driver = partial(self.output_driver,
                                output_path=self.location,
                                app_info=app_info,
                                storage=self.storage,
                                global_attributes=self.global_attributes,
                                var_attributes=self.var_attributes)
        task_runner = partial(execute_task,
                              output_driver=output_driver,
                              chunking=self.computation.get('chunking', {}))

        # does not need to be thorough for now
        task_desc = TaskDescription(type_='datacube_stats',
                                    task_dt=datetime.utcnow().replace(tzinfo=tz.tzutc()),
                                    events_path=Path(self.location),
                                    logs_path=Path(self.location),
                                    parameters=DefaultJobParameters(query={},
                                                                    source_products=[],
                                                                    output_products=[]))

        self.runner(task_desc, tasks, task_runner)

        _LOG.debug('Stopping runner.')
        self.runner.stop()
        _LOG.debug('Runner stopped.')

        result = self.output_driver.result
        source = self.output_driver.source
        return source, result
