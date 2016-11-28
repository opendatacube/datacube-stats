"""
Test that the stats app can generate tasks and run them.
"""
from collections import defaultdict

import pytest
import mock

from datacube.model import MetadataType
from datacube_stats.main import StatProduct
from datacube_stats.models import StatsTask
from datacube_stats.statistics import StatsConfigurationError, ValueStat

from datacube_stats.main import create_stats_app
from .main import StatsApp


@pytest.fixture
def sample_stats_config():
    config = dict()
    config['date_ranges'] = defaultdict(str)
    config['date_ranges']['step_size'] = '3m'
    config['date_ranges']['stats_duration'] = '3m'
    config['date_ranges']['start_date'] = '2015-01-01'
    config['date_ranges']['end_date'] = '2015-04-01'
    # config['output_products'] = []
    config['location'] = []
    config['sources'] = [{'measurements': '', 'product': 'fake_source_product'}]
    config['storage'] = {
        'driver': 'NetCDF CF',
        'tile_size': {'x': 100, 'y': 100},
        'resolution': {'x': 100, 'y': 100},
        'crs': 'EPSG:3577'
    }
    config['output_products'] = [{'name': 'fake_output', 'statistic': 'mean'}]

    return config


def test_create_and_validate_stats_app(sample_stats_config):
    stats_app = create_stats_app(config=sample_stats_config)
    assert stats_app is not None
    stats_app.validate()


def test_raises_error_on_invalid_driver(sample_stats_config):
    sample_stats_config['storage']['driver'] = 'foo'

    with pytest.raises(StatsConfigurationError):
        create_stats_app(sample_stats_config)


def test_raises_error_on_no_sources(sample_stats_config):
    sample_stats_config['sources'] = []
    stats_app = create_stats_app(sample_stats_config)

    with pytest.raises(StatsConfigurationError):
        stats_app.validate()


def test_create_trivial_stats_app():
    stats_app = StatsApp()
    assert stats_app is not None
    with pytest.raises(Exception):
        stats_app.validate()


def test_can_create_output_products(sample_stats_config, mock_index):
    # GIVEN: A simple stats app
    stats_app = create_stats_app(config=sample_stats_config)
    stats_app.index = mock_index
    stats_app.output_products = _SAMPLE_OUTPUTS_SPEC

    # WHEN: I call ensure_output_products()
    output_prods = stats_app.ensure_output_products()

    # THEN: I should receive an appropriately configured output product
    assert len(output_prods) == 1
    fake_output = output_prods['fake_output']
    assert isinstance(fake_output, StatProduct)
    assert fake_output.name == 'fake_output'
    assert fake_output.stat_name == 'mean'
    assert isinstance(fake_output.statistic, ValueStat)

    # TODO: Check output product is created
    # Based on the source product's measurements
    stats_app.index.products.get_by_name.assert_called_with('fake_source_product')


@pytest.fixture
def mock_grid_workflow():
    with mock.patch('datacube_stats.main.GridWorkflow', spec=True) as mock_gwf_class:
        gwf_instance = mock_gwf_class.return_value
        gwf_instance.list_cells.return_value = {(0, 0): mock.MagicMock()}
        yield gwf_instance


@pytest.fixture
def mock_datacube():
    with mock.patch('datacube_stats.main.Datacube', spec=True) as mock_datacube_class:
        dc_instance = mock_datacube_class.return_value
        # dc_instance.list_cells.return_value = {(0, 0): mock.MagicMock()}
        yield dc_instance


@pytest.fixture
def mock_index():
    fake_index = mock.MagicMock()
    fake_index.metadata_types.get_by_name.return_value = mock.MagicMock(spec=MetadataType)
    fake_index.datasets.get_field_names.return_value = {'time', 'source_filter'}
    return fake_index


_SAMPLE_OUTPUTS_SPEC = [dict(name='landsat_yearly_mean',
                             statistic='mean',
                             file_path_template='SR_N_MEAN/SR_N_MEAN_3577_{tile_index[0]}_'
                                                '{tile_index[1]}_{time_period[0]:%Y%m%d}.nc')]
_EXPECTED_DB_FILTER = {'cell_index': None, 'geopolygon': None, 'group_by': 'time', 'product': 'fake_source_product',
                       'source_filter': None, 'time': mock.ANY}


def create_app_with_products(sample_stats_config, mock_index):
    stats_app = create_stats_app(config=sample_stats_config)
    stats_app.index = mock_index
    stats_app.output_products = _SAMPLE_OUTPUTS_SPEC
    output_prods = stats_app.ensure_output_products()
    return stats_app, output_prods


def test_can_generate_tasks(sample_stats_config, mock_index, mock_grid_workflow):
    # GIVEN: A simple stats app that has created some output products
    stats_app, output_prods = create_app_with_products(sample_stats_config, mock_index)

    # WHEN: I call generate_tasks()
    tasks = stats_app.generate_tasks(output_prods)

    # THEN: I should receive an iterable of StatsTasks

    tasks = list(tasks)
    assert len(tasks) == 1
    assert isinstance(tasks[0], StatsTask)

    # Which were created by doing a grid_workflow search
    mock_grid_workflow.list_cells.assert_called_with(**_EXPECTED_DB_FILTER)


def test_gqa_filtering_passed_in_queries(sample_stats_config, mock_index, mock_grid_workflow):
    # GIVEN: A simple stats app configured to filter on GQA, and that has created some output products,
    gqa_filter = {
            'product': 'ls5_level1_scene',
            'gqa': [-1, 1]}
    sample_stats_config['sources'][0]['source_filter'] = gqa_filter
    stats_app, output_prods = create_app_with_products(sample_stats_config, mock_index)

    # WHEN: I call generate_tasks()
    tasks = stats_app.generate_tasks(output_prods)
    list(tasks)  # Make the generator function run

    # THEN: GridWorkFlow should have been called with the GQA filtering
    _expected_filter_args = _EXPECTED_DB_FILTER.copy()
    _expected_filter_args['source_filter'] = gqa_filter

    mock_grid_workflow.list_cells.assert_called_with(**_expected_filter_args)


@pytest.mark.skip
def test_generate_gridded_tasks(sample_stats_config, mock_index, mock_grid_workflow):
    assert False


@pytest.mark.skip
def test_generate_non_gridded_tasks():
    assert False
