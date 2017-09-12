"""
Test that the stats app can generate tasks and run them.
"""
from collections import defaultdict

import pytest
import mock

from datacube.model import MetadataType
from datacube_stats.main import OutputProduct
from datacube_stats.models import StatsTask
from datacube_stats.statistics import StatsConfigurationError, ReducingXarrayStatistic

from datacube_stats.main import StatsApp
import yaml


@pytest.fixture
def sample_stats_config():
    config = yaml.safe_load("""
        date_ranges:
            start_date: 2015-01-01
            end_date: 2015-04-01
        
        location:
        sources:
        -   product: fake_source_product
            measurements: [red]
        storage:
            driver: NetCDFCF
            tile_size:
                x: 100
                y: 100
            resolution:
                x: 100
                y: 100
            crs: EPSG:3577
        output_products:
        -   name: fake_output
            product_type: sample_statistics
            statistic: simple
            statistic_args:
                reduction_function: mean
    """)

    return config


def test_create_and_validate_stats_app(sample_stats_config):
    stats_app = StatsApp.from_configuration_file(config=sample_stats_config)
    assert stats_app is not None
    stats_app.validate()


def test_raises_error_on_invalid_driver(sample_stats_config):
    sample_stats_config['storage']['driver'] = 'foo'

    with pytest.raises(StatsConfigurationError):
        StatsApp.from_configuration_file(sample_stats_config)


def test_raises_error_on_no_sources(sample_stats_config):
    sample_stats_config['sources'] = []
    stats_app = StatsApp.from_configuration_file(config=sample_stats_config)

    with pytest.raises(StatsConfigurationError):
        stats_app.validate()


def test_create_trivial_stats_app():
    stats_app = StatsApp()
    assert stats_app is not None
    with pytest.raises(Exception):
        stats_app.validate()


def test_can_create_output_products(sample_stats_config, mock_index):
    # GIVEN: A simple stats app
    stats_app = StatsApp.from_configuration_file(config=sample_stats_config)
    stats_app.index = mock_index
    stats_app.output_products = _SAMPLE_OUTPUTS_SPEC

    # WHEN: I call configure_outputs()
    output_prods = stats_app.configure_outputs()

    # THEN: I should receive an appropriately configured output product
    assert len(output_prods) == 1
    fake_output = output_prods['fake_output']
    assert isinstance(fake_output, OutputProduct)
    assert fake_output.name == 'fake_output'
    assert fake_output.stat_name == 'simple'
    assert isinstance(fake_output.statistic, ReducingXarrayStatistic)

    # TODO: Check output product is created
    # Based on the source product's measurements
    stats_app.index.products.get_by_name.assert_called_with('fake_source_product')


@pytest.fixture
def mock_grid_workflow():
    with mock.patch('datacube_stats.utils.query.GridWorkflow', spec=True) as mock_gwf_class:
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
    fake_index.products.get_by_name.return_value.measurements = {'red': {
        'name': 'mock_measurement',
        'dtype': 'int8',
        'nodata': -999,
        'units': '1'}}

    fake_index.metadata_types.get_by_name.return_value = MetadataType(
        {
            'name': 'eo',
            'dataset': dict(
                id=['id'],
                label=['ga_label'],
                creation_time=['creation_dt'],
                measurements=['image', 'bands'],
                sources=['lineage', 'source_datasets']
            )
        },
        dataset_search_fields={}
    )

    return fake_index


_SAMPLE_OUTPUTS_SPEC = [dict(name='landsat_yearly_mean',
                             statistic='mean',
                             file_path_template='SR_N_MEAN/SR_N_MEAN_3577_{tile_index[0]}_'
                                                '{tile_index[1]}_{time_period[0]:%Y%m%d}.nc')]
_EXPECTED_DB_FILTER = {'cell_index': None,
                       'geopolygon': None,
                       'group_by': 'time',
                       'product': 'fake_source_product',
                       'source_filter': None,
                       'time': mock.ANY}


def create_app_with_products(sample_stats_config, mock_index):
    stats_app = StatsApp.from_configuration_file(config=sample_stats_config)
    stats_app.index = mock_index
    stats_app.output_products = _SAMPLE_OUTPUTS_SPEC
    output_prods = stats_app.configure_outputs()
    return stats_app, output_prods


@pytest.mark.skip('Pending resolving issues with DriverManager pickling in ODC')
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


@pytest.mark.skip('Pending resolving issues with DriverManager pickling in ODC')
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


@pytest.mark.xfail
def test_generate_gridded_tasks(sample_stats_config, mock_index, mock_grid_workflow):
    assert False


@pytest.mark.xfail
def test_generate_non_gridded_tasks():
    assert False


@pytest.mark.xfail
def test_generate_single_cell_tasks():
    assert False
