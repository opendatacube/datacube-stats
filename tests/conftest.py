import pytest
import yaml
from mock import mock

from datacube.model import MetadataType


@pytest.fixture
def sample_stats_config():
    config = yaml.safe_load("""
        date_ranges:
            start_date: 2015-01-01
            end_date: 2015-04-01

        location: /tmp/foo
        sources:
        -   product: fake_source_product
            measurements: [red]
        storage:
            driver: NetCDF CF
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


@pytest.fixture
def mock_index():
    fake_index = mock.MagicMock()
    fake_index.metadata_types.get_by_name.return_value = mock.MagicMock(spec=MetadataType)

    # Check is performed validating the name of query fields
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
