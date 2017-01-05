"""
Provide some classes for writing data out to files on disk.

The `NetcdfOutputDriver` will write multiple variables into a single file.

The `RioOutputDriver` writes a single __band__ of data per file.
"""
import abc
import logging
import operator
from collections import OrderedDict
from functools import reduce as reduce_
from pathlib import Path

import numpy
import rasterio
import xarray

from datacube.model import Coordinate, Variable, GeoPolygon
from datacube.model.utils import make_dataset, xr_apply, datasets_to_doc
from datacube.storage import netcdf_writer
from datacube.storage.storage import create_netcdf_storage_unit
from datacube.utils import unsqueeze_data_array

_LOG = logging.getLogger(__name__)
_NETCDF_VARIABLE__PARAMETER_NAMES = {'zlib',
                                     'complevel',
                                     'shuffle',
                                     'fletcher32',
                                     'contiguous',
                                     'attrs'}


class StatsOutputError(Exception):
    pass


class OutputFileAlreadyExists(Exception):
    def __init__(self, output_file=None):
        self._output_file = output_file

    def __str__(self):
        return 'Output file already exists: {}'.format(self._output_file)

    def __repr__(self):
        return "OutputFileAlreadyExists({})".format(self._output_file)


class OutputDriver(object):
    """
    Handles the creation of output data files for a StatsTask.

    Depending on the implementation, may create one or more files per instance.

    To use, instantiate the class, using it as a context manager, eg.

        with MyOutputDriver(task, storage, output_path):
            output_driver.write_data(prod_name, measure_name, tile_index, values)

    :param StatsTask task: A StatsTask that will be producing data
        A task will contain 1 or more output products, with each output product containing 1 or more measurements
    :param Union(Path, str) output_path: Base directory name to output file/s into
    :param storage: Dictionary describing the _storage format. eg.
        {
          'driver': 'NetCDF CF'
          'crs': 'EPSG:3577'
          'tile_size': {
                  'x': 100000.0
                  'y': 100000.0}
          'resolution': {
                  'x': 25
                  'y': -25}
          'chunking': {
              'x': 200
              'y': 200
              'time': 1}
          'dimension_order': ['time', 'y', 'x']}
    :param app_info:
    """
    __metaclass__ = abc.ABCMeta
    valid_extensions = []

    def __init__(self, task, storage, output_path, app_info=None):
        self._storage = storage

        self._output_path = output_path

        self._app_info = app_info

        self._output_file_handles = {}
        self._output_paths = {}

        #: datacube_stats.models.StatsTask
        self._task = task

        self._geobox = task.geobox
        self._output_products = task.output_products

    def close_files(self, completed_successfully):
        self.__close_files_helper(self._output_file_handles, completed_successfully)

    def __close_files_helper(self, handles_dict, completed_successfully):
        for output_name, output_fh in handles_dict.items():
            if isinstance(output_fh, dict):
                self.__close_files_helper(output_fh, completed_successfully)
            else:
                output_fh.close()

                # Remove '.tmp' suffix
                if completed_successfully:
                    output_path = self._output_paths[output_fh]
                    output_path.rename(output_path.with_suffix(''))

    @abc.abstractmethod
    def open_output_files(self):
        raise NotImplementedError

    @abc.abstractmethod
    def write_data(self, prod_name, measurement_name, tile_index, values):
        if len(self._output_file_handles) <= 0:
            raise StatsOutputError('No files opened for writing.')

    @abc.abstractmethod
    def write_global_attributes(self, attributes):
        raise NotImplementedError

    def _get_dtype(self, out_prod_name, measurement_name=None):
        if measurement_name:
            return self._output_products[out_prod_name].product.measurements[measurement_name]['dtype']
        else:
            dtypes = set(m['dtype'] for m in self._output_products[out_prod_name].product.measurements.values())
            if len(dtypes) == 1:
                return dtypes.pop()
            else:
                raise StatsOutputError('Not all measurements for %s have the same dtype.' % out_prod_name)

    def __enter__(self):
        self.open_output_files()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        completed_successfully = exc_type is None
        self.close_files(completed_successfully)

    def _prepare_output_file(self, stat, **kwargs):
        x, y = self._task.tile_index
        epoch_start, epoch_end = self._task.time_period

        output_path = Path(self._output_path,
                           stat.file_path_template.format(
                               x=x, y=y,
                               epoch_start=epoch_start,
                               epoch_end=epoch_end,
                               **kwargs))

        if output_path.suffix not in self.valid_extensions:
            raise StatsOutputError('Invalid Filename: %s for this Output Driver: %s' % (output_path, self))

        if output_path.exists():
            raise OutputFileAlreadyExists(output_path)

        try:
            output_path.parent.mkdir(parents=True)
        except OSError:
            pass

        tmp_path = output_path.with_suffix(output_path.suffix + '.tmp')
        if tmp_path.exists():
            tmp_path.unlink()

        return tmp_path


class NetcdfOutputDriver(OutputDriver):
    """
    Write data to Datacube compatible NetCDF files

    The variables in the file will be 3 dimensional, with a single time dimension + y,x.
    """

    valid_extensions = ['.nc']

    def open_output_files(self):
        for prod_name, stat in self._output_products.items():
            output_filename = self._prepare_output_file(stat)
            self._output_paths[prod_name] = output_filename
            self._output_file_handles[prod_name] = self._create_storage_unit(stat, output_filename)

    def _create_storage_unit(self, stat, output_filename):
        all_measurement_defns = list(stat.product.measurements.values())

        datasets, sources = _find_source_datasets(self._task, stat, self._geobox, self._app_info,
                                                  uri=output_filename.as_uri())

        variable_params = self._create_netcdf_var_params(stat)
        nco = self._nco_from_sources(sources,
                                     self._geobox,
                                     all_measurement_defns,
                                     variable_params,
                                     output_filename)

        netcdf_writer.create_variable(nco, 'dataset', datasets, zlib=True)
        nco['dataset'][:] = netcdf_writer.netcdfy_data(datasets.values)
        return nco

    def _create_netcdf_var_params(self, stat):
        chunking = self._storage['chunking']
        chunking = [chunking[dim] for dim in self._storage['dimension_order']]

        variable_params = {}
        for measurement in stat.data_measurements:
            name = measurement['name']
            variable_params[name] = {k: v for k, v in stat._definition.items() if
                                     k in _NETCDF_VARIABLE__PARAMETER_NAMES}
            variable_params[name]['chunksizes'] = chunking
            variable_params[name].update(
                {k: v for k, v in measurement.items() if k in _NETCDF_VARIABLE__PARAMETER_NAMES})
        return variable_params

    @staticmethod
    def _nco_from_sources(sources, geobox, measurements, variable_params, filename):
        coordinates = OrderedDict((name, Coordinate(coord.values, coord.units))
                                  for name, coord in sources.coords.items())
        coordinates.update(geobox.coordinates)

        variables = OrderedDict((variable['name'], Variable(dtype=numpy.dtype(variable['dtype']),
                                                            nodata=variable['nodata'],
                                                            dims=sources.dims + geobox.dimensions,
                                                            units=variable['units']))
                                for variable in measurements)

        return create_netcdf_storage_unit(filename, geobox.crs, coordinates, variables, variable_params)

    def write_data(self, prod_name, measurement_name, tile_index, values):
        self._output_file_handles[prod_name][measurement_name][(0,) + tile_index[1:]] = netcdf_writer.netcdfy_data(
            values)
        self._output_file_handles[prod_name].sync()
        _LOG.debug("Updated %s %s", measurement_name, tile_index[1:])

    def write_global_attributes(self, attributes):
        for output_file in self._output_file_handles.values():
            for k, v in attributes.items():
                output_file.attrs[k] = v


class RioOutputDriver(OutputDriver):
    """
    Save data to file/s using rasterio. Eg. GeoTiff

    Writes to a different file per statistic/measurement.

    """
    valid_extensions = ['.tif', '.tiff']
    default_profile = {
        'compress': 'lzw',
        'driver': 'GTiff',
        'interleave': 'band',
        'tiled': True
    }
    _dtype_map = {
        'int8': 'uint8'
    }

    def __init__(self, *args, **kwargs):
        super(RioOutputDriver, self).__init__(*args, **kwargs)

        self._measurement_bands = {}

    def _get_dtype(self, prod_name, measurement_name=None):
        dtype = super(RioOutputDriver, self)._get_dtype(prod_name, measurement_name)
        return self._dtype_map.get(dtype, dtype)

    def _get_nodata(self, prod_name, measurement_name=None):
        dtype = self._get_dtype(prod_name, measurement_name)
        if measurement_name:
            nodata = self._output_products[prod_name].product.measurements[measurement_name]['nodata']
        else:
            nodatas = set(m['nodata'] for m in self._output_products[prod_name].product.measurements.values())
            if len(nodatas) == 1:
                nodata = nodatas.pop()
            else:
                raise StatsOutputError('Not all nodata values for output product "%s" are the same. '
                                       'Must all match for geotiff output' % prod_name)
        if dtype == 'uint8' and nodata < 0:
            # Convert to uint8 for Geotiff
            return 255
        else:
            return nodata

    def open_output_files(self):
        for prod_name, stat in self._output_products.items():
            num_measurements = len(stat.product.measurements)
            if num_measurements == 0:
                raise ValueError('No measurements to record for {}.'.format(prod_name))
            elif num_measurements > 1 and 'var_name' in stat.file_path_template:
                # Multiple files, each a single band geotiff
                for measurement_name, measure_def in stat.product.measurements.items():
                    self._open_single_band_geotiff(prod_name, stat, measurement_name)
            else:
                # One file only, either a single or a multi-band geotiff
                dest_fh, output_filename = self._open_geotiff(prod_name, None, stat, num_measurements)

                for band, (measurement_name, measure_def) in enumerate(stat.product.measurements.items(), start=1):
                    self._set_band_metadata(dest_fh, measurement_name, band=band)
                self._output_paths[dest_fh] = output_filename
                self._output_file_handles[prod_name] = dest_fh

    def _open_single_band_geotiff(self, prod_name, stat, measurement_name=None):
        dest_fh, output_filename = self._open_geotiff(prod_name, measurement_name, stat)
        self._set_band_metadata(dest_fh, measurement_name)
        self._output_paths.setdefault(prod_name, {})[dest_fh] = output_filename
        self._output_file_handles.setdefault(prod_name, {})[measurement_name] = dest_fh

    def _set_band_metadata(self, dest_fh, measurement_name, band=1):
        start_date, end_date = self._task.time_period
        dest_fh.update_tags(band,
                            source_product=self._task.source_product_names(),
                            start_date='{:%Y-%m-%d}'.format(start_date),
                            end_date='{:%Y-%m-%d}'.format(end_date),
                            name=measurement_name)

    def _open_geotiff(self, prod_name, measurement_name, stat, num_bands=1):
        output_filename = self._prepare_output_file(stat, var_name=measurement_name)
        profile = self.default_profile.copy()
        dtype = self._get_dtype(prod_name, measurement_name)
        nodata = self._get_nodata(prod_name, measurement_name)
        profile.update({
            'blockxsize': self._storage['chunking']['x'],
            'blockysize': self._storage['chunking']['y'],

            'dtype': dtype,
            'nodata': nodata,
            'width': self._geobox.width,
            'height': self._geobox.height,
            'affine': self._geobox.affine,
            'crs': self._geobox.crs.crs_str,
            'count': num_bands
        })
        _LOG.debug("Opening %s for writing.", output_filename)
        dest_fh = rasterio.open(str(output_filename), mode='w', **profile)
        dest_fh.update_tags(created=self._app_info)
        return dest_fh, output_filename

    def write_data(self, prod_name, measurement_name, tile_index, values):
        super(RioOutputDriver, self).write_data(prod_name, measurement_name, tile_index, values)

        prod = self._output_file_handles[prod_name]
        if isinstance(prod, dict):
            output_fh = prod[measurement_name]
            band_num = 1
        else:
            output_fh = prod
            stat = self._output_products[prod_name]
            band_num = list(stat.product.measurements).index(measurement_name) + 1

        t, y, x = tile_index
        window = ((y.start, y.stop), (x.start, x.stop))
        _LOG.debug("Updating %s.%s %s", prod_name, measurement_name, window)

        dtype = self._get_dtype(prod_name, measurement_name)

        output_fh.write(values.astype(dtype), indexes=band_num, window=window)

    def write_global_attributes(self, attributes):
        for dest in self._output_file_handles.values():
            dest.update_tags(**attributes)


class TestOutputDriver(OutputDriver):
    def write_global_attributes(self, attributes):
        pass

    def write_data(self, prod_name, measurement_name, tile_index, values):
        pass

    def open_output_files(self):
        pass


def _format_filename(path_template, **kwargs):
    x, y = kwargs['tile_index']
    epoch_start, epoch_end = kwargs['time_period']
    return Path(str(path_template).format(x=x, y=y, epoch_start=epoch_start, epoch_end=epoch_end,
                                          **kwargs))


def _find_source_datasets(task, stat, geobox, app_info, uri=None):
    def _make_dataset(labels, sources_):
        return make_dataset(product=stat.product,
                            sources=sources_,
                            extent=geobox.extent,
                            center_time=labels['time'],
                            uri=uri,
                            app_info=app_info,
                            valid_data=GeoPolygon.from_sources_extents(sources_, geobox))

    def merge_sources(prod):
        if stat.masked:
            all_sources = xarray.align(prod['data'].sources, *[mask_tile.sources for mask_tile in prod['masks']])
            return reduce_(operator.add, (sources_.sum() for sources_ in all_sources))
        else:
            return prod['data'].sources.sum()

    start_time, _ = task.time_period
    sources = reduce_(operator.add, (merge_sources(prod) for prod in task.sources))
    sources = unsqueeze_data_array(sources, dim='time', pos=0, coord=start_time,
                                   attrs=task.time_attributes)

    datasets = xr_apply(sources, _make_dataset, dtype='O')  # Store in DataArray to associate Time -> Dataset
    datasets = datasets_to_doc(datasets)
    return datasets, sources


class XarrayOutputDriver(OutputDriver):
    def write_data(self, prod_name, measurement_name, tile_index, values):
        pass

    def write_global_attributes(self, attributes):
        pass

    def open_output_files(self):
        pass


OUTPUT_DRIVERS = {
    'NetCDF CF': NetcdfOutputDriver,
    'Geotiff': RioOutputDriver,
    'Test': TestOutputDriver
}
