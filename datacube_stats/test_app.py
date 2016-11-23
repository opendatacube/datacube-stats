"""
Test that the stats app can generate tasks and run them.
"""
from collections import defaultdict

import pytest
import mock

from datacube.model import MetadataType
from datacube_stats.main import StatProduct
from datacube_stats.statistics import StatsConfigurationError

from datacube_stats.main import create_stats_app
from .main import StatsApp


@pytest.fixture(name='minimal_config')
def fixture_minimal_config():
    config = dict()
    config['date_ranges'] = defaultdict(str)
    config['date_ranges']['step_size'] = '3m'
    config['date_ranges']['stats_duration'] = '3m'
    config['date_ranges']['start_date'] = '2015-01-01'
    config['date_ranges']['end_date'] = '2015-04-01'
    config['storage'] = {'driver': 'NetCDF CF'}
    config['sources'] = [{'measurements': ''}]
    config['output_products'] = []
    config['location'] = []
    config['input_region'] = []

    return config


def test_create_and_validate_stats_app(minimal_config):
    stats_app = create_stats_app(config=minimal_config)
    assert stats_app is not None
    stats_app.validate()


def test_raises_error_on_invalid_driver(minimal_config):
    minimal_config['storage'] = {'driver': 'foo'}

    with pytest.raises(StatsConfigurationError):
        create_stats_app(minimal_config)


def test_raises_error_on_no_sources(minimal_config):
    minimal_config['sources'] = []
    stats_app = create_stats_app(minimal_config)

    with pytest.raises(StatsConfigurationError):
        stats_app.validate()


def test_create_trivial_stats_app():
    stats_app = StatsApp()
    assert stats_app is not None
    with pytest.raises(Exception):
        stats_app.validate()


def test_can_create_output_products(simple_stats_app):
    output_prods = simple_stats_app.ensure_output_products()
    assert len(output_prods) == 1
    assert isinstance(output_prods['landsat_yearly_mean'], StatProduct)


@pytest.fixture(name='less_minimal_config')
def fixture_less_minimal_config(minimal_config):
    minimal_config['sources'] = [{'measurements': '', 'product': 'fake_product'}]
    minimal_config['storage']['tile_size'] = {'x': 100, 'y': 100}
    minimal_config['storage']['resolution'] = {'x': 100, 'y': 100}
    minimal_config['storage']['crs'] = 'EPSG:3577'
    del minimal_config['input_region']

    return minimal_config


@pytest.fixture
def mock_grid_workflow():
    with mock.patch('datacube_stats.main.GridWorkflow', spec=True) as mock_gwf:
        instance = mock_gwf.return_value
        instance.list_cells.return_value = {(0, 0): mock.MagicMock()}
        yield


@pytest.fixture(name='simple_stats_app')
def fixture_simple_stats_app(less_minimal_config):
    fake_index = mock.MagicMock()
    fake_index.metadata_types.get_by_name.return_value = mock.MagicMock(spec=MetadataType)
    fake_index.datasets.get_field_names.return_value = {'time', 'source_filter'}

    stats_app = create_stats_app(config=less_minimal_config)
    stats_app.index = fake_index
    stats_app.output_products = [dict(name='landsat_yearly_mean',
                                      statistic='mean',
                                      file_path_template='SR_N_MEAN/SR_N_MEAN_3577_{tile_index[0]}_'
                                                         '{tile_index[1]}_{time_period[0]:%Y%m%d}.nc')]
    return stats_app


@pytest.mark.usefixtures('mock_grid_workflow')
def test_can_generate_tasks(simple_stats_app):
    output_prods = simple_stats_app.ensure_output_products()
    tasks = simple_stats_app.generate_tasks(output_prods)

    tasks = list(tasks)
    assert len(tasks) == 1


@pytest.mark.skip
def test_generate_gridded_tasks():
    assert False


@pytest.mark.skip
def test_generate_non_gridded_tasks():
    assert False
