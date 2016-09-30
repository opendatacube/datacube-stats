import rasterio
import xarray as xr
import pandas as pd
import logging
from datacube import Datacube
from datacube.api.query import query_group_by, query_geopolygon
from datacube.model import GeoBox
from datacube.helpers import ga_pq_fuser
from datacube.storage.masking import make_mask
from tqdm import trange
import tqdm

from datacube.ui import click as ui
import click

DEFAULT_PROFILE = {
    'blockxsize': 256,
    'blockysize': 256,
    # 'compress': 'lzw',
    'driver': 'GTiff',
    'interleave': 'band',
    'tiled': True,
    'dtype': 'float32',
    # 'nodata': -999,
    'bigtiff': 'YES',
    # 'num_threads': 8
}

MASK_FLAGS = {
    'contiguous': True,
    'cloud_acca': 'no_cloud',
    'cloud_fmask': 'no_cloud',
    'cloud_shadow_acca': 'no_cloud_shadow',
    'cloud_shadow_fmask': 'no_cloud_shadow',
    'blue_saturated': False,
    'green_saturated': False,
    'red_saturated': False,
    'nir_saturated': False,
    'swir1_saturated': False,
    'swir2_saturated': False
}


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super(self.__class__, self).__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:  # pylint: disable=bare-except
            self.handleError(record)

_LOG = logging.getLogger('stack_solar_days')


def use_tqdm_root_logger():
    root = logging.getLogger()
    old_handler = root.handlers[-1]
    new_handler = TqdmLoggingHandler()
    new_handler.formatter = old_handler.formatter
    root.removeHandler(old_handler)

    new_handler.setLevel(logging.INFO)
    logging.root.setLevel(logging.DEBUG)

    root.addHandler(new_handler)


def load_masked_data(dc, name, **kwargs):
    data = dc.load(product='%s_nbar_albers' % name, group_by='solar_day', **kwargs)
    pq = dc.load(product='%s_pq_albers' % name, like=data, fuse_func=ga_pq_fuser, group_by='solar_day')
    mask = make_mask(pq.pixelquality, ga_good_pixel=True)
    data = data.where(mask)
    data.attrs['crs'] = pq.attrs['crs']
    return data


def dt64_to_str(dt64, strformat='%Y-%m-%d'):
    t = pd.to_datetime(str(dt64))
    return t.strftime(strformat)

NBAR_PRODS = ('ls5_nbar_albers', 'ls7_nbar_albers', 'ls8_nbar_albers')
PQ_PRODS = tuple([prod.replace('nbar', 'pq') for prod in NBAR_PRODS])


@click.command(short_help="Stack bands into a multiband dataset.")
@click.option('--date-from')
@click.option('--date-to')
@click.option('--x', nargs=2)
@click.option('--y', nargs=2)
@click.option('--nd-bands', nargs=2, help="Normalised Difference Bands")
@click.argument('filename')
@ui.global_cli_options
@ui.pass_index(app_name='agdc-stack_solar_days')
def stack(index, date_from, date_to, x, y, nd_bands, filename):
    use_tqdm_root_logger()
    _LOG.info("Stacking Normalised Difference of %s of solar days from %s to %s", nd_bands, date_from, date_to)
    dc = Datacube(index=index)
    query = dict(time=(date_from, date_to), x=x, y=y, crs='EPSG:4326')
    query['measurements'] = nd_bands

    _LOG.info("Querying database for available data")

    nbar_sources = _load_sources(dc, query, NBAR_PRODS, 'nbar')

    pq_sources = _load_sources(dc, query, PQ_PRODS, 'pq')

    if len(nbar_sources) != len(pq_sources):
        difference = set(pq_sources.time.values) ^ set(nbar_sources.time.values)
        _LOG.info('Excluding NBAR/PQ Mismatched dates: %s', difference)

    _LOG.info("Merging into one time series")
    nbar_pq = xr.merge([nbar_sources, pq_sources], join='inner')

    grid_spec = dc.index.products.get_by_name('ls5_nbar_albers').grid_spec
    geobox = GeoBox.from_geopolygon(query_geopolygon(**query), grid_spec.resolution,
                                    grid_spec.crs, grid_spec.alignment)

    _LOG.info("Writing to: %s", filename)
    write_dataarray_geotiff(dc, filename, nbar_pq, geobox=geobox, measurements=nd_bands)


def _load_sources(dc, query, prods, name):
    group_by = query_group_by(group_by='solar_day')
    observations = sum([dc.product_observations(product=prod, **query) for prod in prods], [])
    _LOG.info("Loading %s observations. Found %s", name, len(observations))
    sources = dc.product_sources(observations, group_by)
    sources.name = name
    _LOG.info("Grouped into %s days.", len(sources))
    return sources


def write_dataarray_geotiff(dc, filename, dataarray_sources, geobox, measurements):
    """
    Write an xarray dataset to a geotiff

    :attr bands: ordered list of dataset names
    :attr time_index: time index to write to file
    :attr dataset: xarray dataset containing multiple bands to write to file
    :attr profile_override: option dict, overrides rasterio file creation options.
    """
    profile = DEFAULT_PROFILE.copy()
    profile.update({
        'width': geobox.width,
        'height': geobox.height,
        'affine': geobox.affine,
        'crs': geobox.crs.crs_str,
        'count': len(dataarray_sources.time),
    })
    band1, band2 = measurements
    measurements = [dict(name=measurement, dtype='int16', nodata=-999, units='1') for measurement in measurements]

    pq_measurements = dc.index.products.get_by_name('ls5_pq_albers').measurements.values()

    with rasterio.open(filename, mode='w', **profile) as dst:
        for i in trange(len(dataarray_sources.time), desc='time slices written', unit='slice'):
            # Load data
            data = dc.product_data(dataarray_sources.nbar[i:i + 1], geobox, measurements=measurements)
            # Mask with PQ
            pq = dc.product_data(dataarray_sources.pq[i:i + 1], geobox,
                                 measurements=pq_measurements, fuse_func=ga_pq_fuser)['pixelquality']
            mask = make_mask(pq, **MASK_FLAGS)
            del pq

            data = data.where(mask)
            del mask

            data = (data[band1] - data[band2]) / (data[band1] + data[band2])

            time_str = dt64_to_str(dataarray_sources.time[i].values)
            platform = dataarray_sources.nbar[i].item()[0].type.name

            dst.write(data.values[0].astype('float32'), i + 1)

            dst.update_tags(i + 1, time=time_str, platform=platform)


if __name__ == '__main__':
    stack()
