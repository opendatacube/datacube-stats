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
    config = defaultdict(list)
    config = exceptdefaultdict(default_factory=list, except_keys=['input_region'])
    config = exceptdefaultdict(list)
    config['date_ranges'] = defaultdict(str)
    config['date_ranges']['step_size'] = '3m'
    config['date_ranges']['stats_duration'] = '3m'
    config['date_ranges']['start_date'] = '2015-01-01'
    config['date_ranges']['end_date'] = '2015-04-01'
    config['storage'] = {'driver': 'NetCDF CF'}
    config['sources'] = [{'measurements': ''}]
    return config


def test_create_empty_stats_app(minimal_config):
    stats_app = create_stats_app(config=minimal_config)
    assert stats_app is not None
    stats_app.validate()


def test_invalid_driver(minimal_config):
    minimal_config['storage'] = {'driver': 'foo'}

    with pytest.raises(StatsConfigurationError):
        create_stats_app(minimal_config)


def test_no_sources(minimal_config):
    minimal_config['sources'] = []
    stats_app = create_stats_app(minimal_config)

    with pytest.raises(StatsConfigurationError):
        stats_app.validate()


def test_stats_app():
    stats_app = StatsApp()
    assert stats_app is not None
    with pytest.raises(Exception):
        stats_app.validate()


def test_ensure_output_products(minimal_config):
    minimal_config['sources'] = [{'measurements': '', 'product': 'fake_product'}]
    fake_index = mock.MagicMock()
    fake_index.metadata_types.get_by_name.return_value = MetadataType({}, {})
    stats_app = create_stats_app(config=minimal_config)
    stats_app.index = fake_index
    stats_app.output_products = [dict(name='landsat_yearly_mean',
                                      statistic='mean',
                                      file_path_template='SR_N_MEAN/SR_N_MEAN_3577_{tile_index[0]}_'
                                                         '{tile_index[1]}_{time_period[0]:%Y%m%d}.nc')]
    output_prods = stats_app.ensure_output_products()
    assert len(output_prods) == 1
    assert isinstance(output_prods['landsat_yearly_mean'], StatProduct)


class exceptdefaultdict(defaultdict):
    # def __init__(self, except_keys=None, *args, **kwargs):
    #     self._except_keys = except_keys if except_keys is not None else []
    #     super(exceptdefaultdict, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        print(key)
        if key in ['input_region']:
        # if key in self._except_keys:
            raise KeyError
        else:
            # return self.default_factory()
            return super(exceptdefaultdict, self).__missing__(key)


@mock.patch('datacube.Datacube')
def test_generate_tasks(mocked_datacube, minimal_config):
    mocked_datacube

    minimal_config['sources'] = [{'measurements': '', 'product': 'fake_product'}]
    fake_index = mock.MagicMock()
    fake_index.metadata_types.get_by_name.return_value = MetadataType({}, {})
    stats_app = create_stats_app(config=minimal_config)
    stats_app.index = fake_index
    stats_app.output_products = [dict(name='landsat_yearly_mean',
                                      statistic='mean',
                                      file_path_template='SR_N_MEAN/SR_N_MEAN_3577_{tile_index[0]}_'
                                                         '{tile_index[1]}_{time_period[0]:%Y%m%d}.nc')]
    output_prods = stats_app.ensure_output_products()
    tasks = stats_app.generate_tasks(output_prods)

    tasks = list(tasks)
    print(tasks)
    assert tasks is None



@pytest.mark.skip
def test_generate_gridded_tasks():
    assert False


@pytest.mark.skip
def test_generate_non_gridded_tasks():
    assert False
