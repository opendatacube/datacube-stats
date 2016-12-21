from mock import MagicMock

import numpy as np
from affine import Affine
from datacube.model import MetadataType, GeoBox, CRS
from datacube_stats.models import OutputProduct
from datacube_stats.output_drivers import RioOutputDriver
from datacube_stats.statistics import NoneStat
from datacube_stats.main import StatsTask
from datetime import datetime


def test_rio(tmpdir):
    m = MetadataType({}, {})
    storage = {'chunking': {'x': 256, 'y': 256}}
    # input_measurement = MagicMock()
    input_measurement = {'name': 'sample_input_measurement',
                         'dtype': 'int16',
                         'nodata': -999,
                         'units': '1'}
    output_product = OutputProduct(m,
                                   input_measurements=[input_measurement],
                                   storage=storage,
                                   name='test_product',
                                   file_path_template='foo_bar.tiff',
                                   stat_name='',
                                   statistic=NoneStat())
    # dtypes = set(m['dtype'] for m in self._output_products[out_prod_name].product.measurements.values())

    # task = StatsTask(time_period=(1, 2), tile_index=None, sources=None, output_products=[output_product])
    task = MagicMock()
    task.geobox = GeoBox(4000, 4000, Affine(0.00025, 0.0, 151.0, 0.0, -0.00025, -29.0), CRS('EPSG:4326'))
    task.output_products = {'sample_prod': output_product}
    task.tile_index = 0, 0
    task.time_period = datetime.now(), datetime.now()
    output_path = str(tmpdir)
    tile_index = (None, slice(0, 2000, None), slice(0, 2000, None))
    values = np.full((2000, 2000), fill_value=7, dtype=np.int16)
    with RioOutputDriver(task=task, storage=storage, output_path=output_path) as output_driver:
        # Why isn't get_dtype being called when opening the file? ... It is
        output_driver.write_data(prod_name='sample_prod', measurement_name='sample_input_measurement',
                                 tile_index=tile_index, values=values)

    print('foo')
