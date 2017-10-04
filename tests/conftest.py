import pytest
import yaml


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