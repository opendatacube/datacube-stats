"""
Provide some classes for writing data out to files on disk.

The `NetcdfOutputDriver` will write multiple variables into a single file.

The `RioOutputDriver` writes a single __band__ of data per file.
"""
import abc
from collections import OrderedDict
import logging
from functools import reduce as reduce_
from pathlib import Path
import operator

import numpy
import rasterio
import xarray
from datacube.model import Coordinate, Variable, GeoPolygon
from datacube.model.utils import make_dataset, xr_apply, datasets_to_doc
from datacube.storage import netcdf_writer
from datacube.storage.storage import create_netcdf_storage_unit
from datacube.utils import unsqueeze_data_array

_LOG = logging.getLogger(__name__)
NETCDF_VARIABLE_PARAMETER_NAMES = {'zlib',
                                   'complevel',
                                   'shuffle',
                                   'fletcher32',
                                   'contiguous',
                                   'attrs'}


class OutputDriver(object):
    """
    Handles the creation of output data files for a StatsTask.

    Depending on the implementation, may create one or more files per instance.

    To use, instantiate the class, then use as a context manager, eg.

        output_driver = MyOutputDriver(storage, task, output_path)
        with output_driver:
            output_driver.write_data(prod_name, measure_name, tile_index, values)

    :param StatsTask task: A StatsTask that will be producing data
    :param output_path: Directory name to output file/s into
    :param storage: Dictionary structure describing the _storage format
    :param app_info:
    """
    __metaclass__ = abc.ABCMeta

    # TODO: Add check for valid filename extensions in each driver
    def __init__(self, storage, task, output_path, app_info=None):
        self._task = task
        self._output_path = output_path
        self._storage = storage

        self._geobox = task.geobox
        self._output_products = task.output_products

        self._output_files = {}
        self._app_info = app_info

    def close_files(self):
        for output_file in self._output_files.values():
            output_file.close()

    @abc.abstractmethod
    def open_output_files(self):
        raise NotImplementedError

    @abc.abstractmethod
    def write_data(self, prod_name, measurement_name, tile_index, values):
        raise NotImplementedError

    @abc.abstractmethod
    def write_global_attributes(self, attributes):
        raise NotImplementedError

    def _get_dtype(self, out_prod_name, measurement_name):
        return self._output_products[out_prod_name].product.measurements[measurement_name]['dtype']

    def __enter__(self):
        self.open_output_files()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_files()


class NetcdfOutputDriver(OutputDriver):
    """
    Write data to Datacube compatible NetCDF files
    """

    valid_extensions = ['nc']

    def open_output_files(self):
        for prod_name, stat in self._output_products.items():
            filename_template = str(Path(self._output_path, stat.file_path_template))
            output_filename = _format_filename(filename_template, **self._task)
            self._output_files[prod_name] = self._create_storage_unit(stat, output_filename)

    def _create_storage_unit(self, stat, output_filename):
        geobox = self._geobox
        all_measurement_defns = list(stat.product.measurements.values())

        datasets, sources = _find_source_datasets(self._task, stat, geobox, self._app_info,
                                                  uri=output_filename.as_uri())

        variable_params = self._create_netcdf_var_params(stat)
        nco = self._nco_from_sources(sources,
                                     geobox,
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
            variable_params[name] = {k: v for k, v in stat._definition.items() if k in NETCDF_VARIABLE_PARAMETER_NAMES}
            variable_params[name]['chunksizes'] = chunking
            variable_params[name].update({k: v for k, v in measurement.items() if k in NETCDF_VARIABLE_PARAMETER_NAMES})
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
        self._output_files[prod_name][measurement_name][(0,) + tile_index[1:]] = netcdf_writer.netcdfy_data(values)
        self._output_files[prod_name].sync()
        _LOG.debug("Updated %s %s", measurement_name, tile_index[1:])

    def write_global_attributes(self, attributes):
        for output_file in self._output_files.values():
            for k, v in attributes.items():
                output_file.attrs[k] = v


class RioOutputDriver(OutputDriver):
    """
    Save data to file/s using rasterio. Eg. GeoTiff

    Writes to a different file per statistic/measurement.

    """
    valid_extensions = ['tif', 'tiff']
    default_profile = {
        'compress': 'lzw',
        'driver': 'GTiff',
        'interleave': 'band',
        'tiled': True
    }

    def open_output_files(self):
        for prod_name, stat in self._output_products.items():
            for measurename, measure_def in stat.product.measurements.items():
                filename_template = str(Path(self._output_path, stat.file_path_template))

                output_filename = _format_filename(filename_template,
                                                   var_name=measurename,
                                                   **self._task)
                try:
                    output_filename.parent.mkdir(parents=True)
                except OSError:
                    pass

                profile = self.default_profile.copy()

                profile.update({
                    'blockxsize': self._storage['chunking']['x'],
                    'blockysize': self._storage['chunking']['y'],

                    'dtype': measure_def['dtype'],
                    'nodata': measure_def['nodata'],
                    'width': self._geobox.width,
                    'height': self._geobox.height,
                    'affine': self._geobox.affine,
                    'crs': self._geobox.crs.crs_str,
                    'count': 1
                })

                output_name = prod_name + measurename

                _LOG.debug("Opening %s for writing.", output_filename)

                dest = rasterio.open(str(output_filename), mode='w', **profile)
                # dest.update_tags(created=self._app_info) # TODO record creation metadata
                dest.update_tags(1, platform=self._task.sources[0]['data'].product.name,
                                 date='{:%Y-%m-%d}'.format(self._task.time_period[0]))
                self._output_files[output_name] = dest

    def write_data(self, prod_name, measurement_name, tile_index, values):
        output_name = prod_name + measurement_name
        y, x = tile_index[1:]
        window = ((y.start, y.stop), (x.start, x.stop))
        _LOG.debug("Updating %s.%s %s", prod_name, measurement_name, window)

        dtype = self._get_dtype(prod_name, measurement_name)

        self._output_files[output_name].write(values.astype(dtype), indexes=1, window=window)

    def write_global_attributes(self, attributes):
        for dest in self._output_files.values():
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
    return Path(str(path_template).format(x=x, y=y, epoch_start=epoch_start, epoch_end=epoch_end))


def _find_source_datasets(task, stat, geobox, app_info, uri=None):
    def _make_dataset(labels, sources):
        return make_dataset(product=stat.product,
                            sources=sources,
                            extent=geobox.extent,
                            center_time=labels['time'],
                            uri=uri,
                            app_info=app_info,
                            valid_data=GeoPolygon.from_sources_extents(sources, geobox))

    def merge_sources(prod):
        if stat.masked:
            all_sources = xarray.align(prod['data'].sources, *[mask_tile.sources for mask_tile in prod['masks']])
            return reduce_(operator.add, (sources.sum() for sources in all_sources))
        else:
            return prod['data'].sources.sum()

    start_time, _ = task.time_period
    sources = reduce_(operator.add, (merge_sources(prod) for prod in task.sources))
    sources = unsqueeze_data_array(sources, dim='time', pos=0, coord=start_time,
                                   attrs=task.time_attributes)

    datasets = xr_apply(sources, _make_dataset, dtype='O')  # Store in DataArray to associate Time -> Dataset
    datasets = datasets_to_doc(datasets)
    return datasets, sources
