"""
Provide some classes for writing data out to files on disk.
"""
import abc
import logging
import operator
import subprocess
import tempfile
import pydash
from collections import OrderedDict
from functools import reduce as reduce_
from pathlib import Path
from typing import Iterable, Tuple, List

import numpy
import rasterio
import xarray
from boltons import fileutils
from datacube.model.utils import make_dataset, xr_apply, datasets_to_doc
from datacube.utils import unsqueeze_data_array, geometry
from datacube.drivers.netcdf import (
    Variable,
    writer as netcdf_writer,
    create_netcdf_storage_unit)
from six import with_metaclass

from .models import OutputProduct
from .utils import prettier_slice

_LOG = logging.getLogger(__name__)
_NETCDF_VARIABLE__PARAMETER_NAMES = {'zlib',
                                     'complevel',
                                     'shuffle',
                                     'fletcher32',
                                     'contiguous',
                                     'attrs'}

OUTPUT_DRIVERS = {}


def polygon_from_sources_extents(sources, geobox):
    sources_union = geometry.unary_union(source.extent.to_crs(geobox.crs) for source in sources)

    # TODO: remove ._geom check once datacube-core is fixed older versions of
    #       datacube returned unusable Geometry object instead of None on
    #       failure
    if sources_union is None or sources_union._geom is None:  # pylint: disable=protected-access
        _LOG.warning('Failed to compute "valid_region" from a set of datasets')
        return geobox.extent

    valid_data = geobox.extent.intersection(sources_union)
    resolution = min([abs(x) for x in geobox.resolution])
    return valid_data.simplify(tolerance=resolution * 0.01)


class RegisterDriver(abc.ABCMeta):
    """
    A metaclass which registers all sub-classes of :class:`OutputDriver` into the OUTPUT_DRIVERS dictionary.
    """
    # pylint: disable=bad-mcs-classmethod-argument
    def __new__(mcs, name, bases, namespace, **kwargs):
        # pylint: disable=bad-mcs-classmethod-argument
        cls = type.__new__(mcs, name, bases, namespace, **kwargs)
        if hasattr(cls, '_driver_name'):
            # pylint: disable=protected-access
            OUTPUT_DRIVERS[cls._driver_name] = cls
        return cls


def get_driver_by_name(name):
    """Search for an output driver, ignoring case and spaces."""
    for driver_name, driver_class in OUTPUT_DRIVERS.items():
        if driver_name.lower().replace(' ', '') == name.lower().replace(' ', ''):
            return driver_class
    raise NoSuchOutputDriver()


class NoSuchOutputDriver(Exception):
    """The requested output driver is not available."""


class StatsOutputError(Exception):
    """Something went wrong while writing to output files."""


class OutputFileAlreadyExists(Exception):
    def __init__(self, output_file=None):
        super(OutputFileAlreadyExists, self).__init__()
        self._output_file = output_file

    def __str__(self):
        return 'Output file already exists: {}'.format(self._output_file)

    def __repr__(self):
        return "OutputFileAlreadyExists({})".format(self._output_file)


def _walk_dict(file_handles, func):
    """

    :param file_handles:
    :param func: returns iterable
    :return:
    """
    for _, output_fh in file_handles.items():
        if isinstance(output_fh, dict):
            for out in _walk_dict(output_fh, func):
                yield out
        else:
            try:
                yield func(output_fh)
            except TypeError as te:
                _LOG.debug('Error running %s: %s', func, te)


class OutputDriver(with_metaclass(RegisterDriver)):
    """
    Handles the creation of output data files for a StatsTask.

    Depending on the implementation, may create one or more files per instance.

    To use, instantiate the class, using it as a context manager, eg.

        with MyOutputDriver(task, storage, output_path):
            output_driver.write_data(prod_name, measure_name, chunk, values)

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
    valid_extensions: List[str] = []

    def __init__(self, task, storage, output_path, app_info=None, global_attributes=None, var_attributes=None):
        self._storage = storage

        self._output_path = output_path

        self._app_info = app_info

        #: Maps from prod_name to Output File Handle
        self._output_file_handles = {}

        #: Map temporary file it's target final filename
        self.output_filename_tmpname = {}

        #: datacube_stats.models.StatsTask
        self._task = task

        self._geobox = task.geobox
        self._output_products = task.output_products

        #: dict of str to str
        self.global_attributes = global_attributes

        #: dict of str to dict of str to str
        self.var_attributes = var_attributes if var_attributes is not None else {}

    def close_files(self, completed_successfully: bool) -> Iterable[Path]:
        # Turn file_handles into paths
        written_paths = list(_walk_dict(self._output_file_handles, self._handle_to_path))

        # Close Files, need to iterate with list()  since the _walk_dict() generator is lazy
        list(_walk_dict(self._output_file_handles, lambda fh: fh.close()))

        # Rename to final filename
        if completed_successfully:
            destinations = [self.output_filename_tmpname[path] for path in written_paths]
            for tmp, dest in zip(written_paths, destinations):
                atomic_rename(tmp, dest)

            return destinations
        else:
            return written_paths

    @classmethod
    def format_name(cls):
        return cls._driver_name

    def _handle_to_path(self, file_handle) -> Path:
        return Path(file_handle.name)

    @abc.abstractmethod
    def open_output_files(self):
        raise NotImplementedError

    def write_chunk(self, prod_name, chunk, result):
        for var_name, var in result.data_vars.items():
            self.write_data(prod_name, var_name, chunk, var.values)

    @abc.abstractmethod
    def write_data(self, prod_name, measurement_name,
                   chunk: Tuple[slice, slice, slice],
                   values: numpy.ndarray) -> None:
        if len(self._output_file_handles) <= 0:
            raise StatsOutputError('No files opened for writing.')

    @abc.abstractmethod
    def write_global_attributes(self, attributes):
        raise NotImplementedError

    def __enter__(self):
        self.open_output_files()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        completed_successfully = exc_type is None
        self.close_files(completed_successfully)

    def _prepare_output_file(self, output_product: OutputProduct, **kwargs) -> Path:
        """
        Format the output filename for the current task,
        make sure it is valid and doesn't already exist
        Make sure parent directories exist
        Switch it around for a temporary filename.

        :return: Path to write output to
        """
        output_path = self._generate_output_filename(output_product, **kwargs)

        if output_path.suffix not in self.valid_extensions:
            raise StatsOutputError('Invalid Filename: %s for this Output Driver: %s' % (output_path, self))

        if output_path.exists():
            raise OutputFileAlreadyExists(output_path)

        # Ensure target directory exists
        try:
            output_path.parent.mkdir(parents=True)
        except OSError:
            pass

        with tempfile.NamedTemporaryFile(dir=str(output_path.parent)) as tmpfile:
            pass
        tmp_path = Path(tmpfile.name)
        self.output_filename_tmpname[tmp_path] = output_path

        return tmp_path

    def _generate_output_filename(self, output_product: OutputProduct, **kwargs) -> Path:
        # Fill parameters from config file filename specification
        params = self._task.spatial_id.copy()

        params['epoch_start'], params['epoch_end'] = self._task.time_period
        params['name'] = output_product.name
        params['stat_name'] = output_product.stat_name
        params.update(output_product.extras)
        params.update(kwargs)

        output_path = Path(self._output_path,
                           output_product.file_path_template.format(**params))
        return output_path

    def _find_source_datasets(self, stat: OutputProduct, uri: str = None,
                              band_uris: dict = None) -> xarray.DataArray:
        """
        Find all the source datasets for a task

        Put them in order so that they can be assigned to a stacked output aligned against it's time dimension

        :return: (datasets, sources)

        datasets is a bunch of strings to dump, indexed on time
        sources is more structured. An x-array of lists of dataset sources, indexed on time
        """
        task = self._task
        geobox = self._task.geobox
        app_info = self._app_info

        def add_all(iterable):
            return reduce_(operator.add, iterable)

        def merge_sources(prod):
            # Merge data sources and mask sources
            # Align the data `Tile` with potentially many mask `Tile`s along their time axis
            all_sources = xarray.align(prod.data.sources,
                                       *[mask_tile.sources for mask_tile in prod.masks if mask_tile])

            # TODO: The following can fail if prod.data and prod.masks have different times
            # Which can happen in the case of a missing PQ Scene, where there is a scene overlap
            # ie. Two overlapped NBAR scenes, One PQ scene (the later)
            return add_all(sources_.sum() for sources_ in all_sources)

        sources = add_all(merge_sources(prod) for prod in task.sources)

        def unique(index, dataset_tuple):
            return tuple(set(dataset_tuple))

        sources = xr_apply(sources, unique, dtype='O')

        # Sources has no time at this point, so insert back in the start of our stats epoch
        start_time, _ = task.time_period
        sources = unsqueeze_data_array(sources, dim='time', pos=0, coord=start_time,
                                       attrs=task.time_attributes)

        if not sources:
            raise StatsOutputError('No valid sources found, or supplied sources do not align to the same time.\n'
                                   'Unable to write dataset metadata.')

        def _make_dataset(labels, sources_):
            return make_dataset(product=stat.product,
                                sources=sources_,
                                extent=geobox.extent,
                                center_time=labels['time'],
                                uri=uri,
                                band_uris=band_uris,
                                app_info=app_info,
                                valid_data=polygon_from_sources_extents(sources_, geobox))
        datasets = xr_apply(sources, _make_dataset, dtype='O')  # Store in DataArray to associate Time -> Dataset
        datasets = datasets_to_doc(datasets)
        return datasets


class NetCDFCFOutputDriver(OutputDriver):
    """
    Write data to Datacube compatible NetCDF files

    The variables in the file will be 3 dimensional, with a single time dimension + y,x.
    """
    _driver_name = 'NetCDF CF'

    valid_extensions = ['.nc']

    @classmethod
    def format_name(cls):
        return 'NetCDF'

    def open_output_files(self):
        for prod_name, stat in self._output_products.items():
            output_filename = self._prepare_output_file(stat)
            self._output_file_handles[prod_name] = self._create_storage_unit(stat, output_filename)

    def _handle_to_path(self, file_handle) -> Path:
        return Path(file_handle.filepath())

    def _create_storage_unit(self, stat: OutputProduct, output_filename: Path):
        all_measurement_defns = list(stat.product.measurements.values())

        datasets = self._find_source_datasets(stat, uri=output_filename.as_uri())

        variable_params = self._create_netcdf_var_params(stat)
        nco = self._nco_from_sources(datasets,
                                     self._geobox,
                                     all_measurement_defns,
                                     variable_params,
                                     output_filename)

        netcdf_writer.create_variable(nco, 'dataset', datasets, zlib=True)
        nco['dataset'][:] = netcdf_writer.netcdfy_data(datasets.values)
        return nco

    def _create_netcdf_var_params(self, stat: OutputProduct):
        def build_attrs(name, attrs):
            return pydash.assign(dict(long_name=name,
                                      coverage_content_type='modelResult'),  # defaults
                                 attrs,                                      # Defined by plugin
                                 self.var_attributes.get(name, {}))          # From config, highest priority

        chunking = self._storage['chunking']
        chunking = [chunking[dim] for dim in self._storage['dimension_order']]

        variable_params = {}
        for measurement in stat.data_measurements:
            name = measurement['name']

            if stat.output_params is None:
                v_params = {}
            else:
                v_params = stat.output_params.copy()
            v_params['chunksizes'] = chunking
            v_params.update(
                {k: v for k, v in measurement.items() if k in _NETCDF_VARIABLE__PARAMETER_NAMES})

            v_params['attrs'] = build_attrs(name, v_params.get('attr', {}))

            variable_params[name] = v_params
        return variable_params

    def _nco_from_sources(self, sources, geobox, measurements, variable_params, filename):

        coordinates = OrderedDict((name, geometry.Coordinate(coord.values, coord.units))
                                  for name, coord in sources.coords.items())
        coordinates.update(geobox.coordinates)

        variables = OrderedDict((variable['name'], Variable(dtype=numpy.dtype(variable['dtype']),
                                                            nodata=variable['nodata'],
                                                            dims=sources.dims + geobox.dimensions,
                                                            units=variable['units']))
                                for variable in measurements)

        return create_netcdf_storage_unit(filename, crs=geobox.crs, coordinates=coordinates,
                                          variables=variables, variable_params=variable_params,
                                          global_attributes=self.global_attributes)

    def write_data(self, prod_name, measurement_name, chunk, values):
        self._output_file_handles[prod_name][measurement_name][(0,) + chunk[1:]] = netcdf_writer.netcdfy_data(
            values)
        self._output_file_handles[prod_name].sync()
        _LOG.debug("Updated %s %s", measurement_name,
                   "({})".format(", ".join(prettier_slice(x) for x in chunk[1:])))

    def write_global_attributes(self, attributes):
        for output_file in self._output_file_handles.values():
            for k, v in attributes.items():
                output_file.attrs[k] = v


class GeoTiffOutputDriver(OutputDriver):
    """
    Save data to file/s using rasterio. Eg. GeoTIFF

    Con write all statistics to the same output file, or each statistic to a different file.
    """
    _driver_name = 'GeoTIFF'
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
        super(GeoTiffOutputDriver, self).__init__(*args, **kwargs)

    def _get_dtype(self, out_prod_name, measurement_name=None):
        if measurement_name:
            dtype = self._output_products[out_prod_name].product.measurements[measurement_name]['dtype']
        else:
            dtypes = set(m['dtype'] for m in self._output_products[out_prod_name].product.measurements.values())
            if len(dtypes) == 1:
                dtype = dtypes.pop()
            else:
                raise StatsOutputError('Not all measurements for %s have the same dtype.'
                                       'For GeoTIFF output they must ' % out_prod_name)
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
                # Output each statistic product into a separate single band geotiff file
                tmp_filenames = [self._open_single_band_geotiff(prod_name, stat, measurement_name)
                                 for measurement_name, measure_def in stat.product.measurements.items()]

                yaml_filename = str(self._generate_output_filename(stat, var_name='').with_suffix('.yaml'))
                self.write_yaml(stat, tmp_filenames, yaml_filename, multiband=False)

            else:
                # Output all statistics into a single geotiff file, with as many bands
                # as there are output statistic products
                output_filename = self._prepare_output_file(stat)

                self.write_yaml(stat, [output_filename] * len(stat.product.measurements))

                dest_fh = self._open_geotiff(prod_name, None, output_filename, num_measurements)

                for band, (measurement_name, measure_def) in enumerate(stat.product.measurements.items(), start=1):
                    self._set_band_metadata(dest_fh, measurement_name, band=band)
                self._output_file_handles[prod_name] = dest_fh

    def write_yaml(self, stat: OutputProduct, tmp_filenames: List[Path], yaml_filename=None, multiband=True):
        output_filenames = [self.output_filename_tmpname[tmp_filename]
                            for tmp_filename in tmp_filenames]

        uris = [output_filename.absolute().as_uri()
                for output_filename in output_filenames]

        def layer(index):
            if multiband:
                return index + 1
            return 1

        band_uris = {name: {'layer': layer(index), 'path': uris[index]}
                     for index, name in enumerate(stat.product.measurements)}

        datasets = self._find_source_datasets(stat, uri=None, band_uris=band_uris)

        if yaml_filename is None:
            yaml_filename = str(output_filenames[0].with_suffix('.yaml'))

        # Write to Yaml
        if len(datasets) == 1:  # I don't think there should ever be more than 1 dataset in here...
            _LOG.info('writing dataset yaml for %s to %s', stat, yaml_filename)
            with fileutils.atomic_save(yaml_filename) as yaml_dst:
                yaml_dst.write(datasets.values[0])
        else:
            _LOG.error('Unexpected more than 1 dataset %r being written at once, '
                       'investigate!', datasets)

    def _open_single_band_geotiff(self, prod_name, stat, measurement_name=None):
        output_filename = self._prepare_output_file(stat, var_name=measurement_name)
        dest_fh = self._open_geotiff(prod_name, measurement_name, output_filename)
        self._set_band_metadata(dest_fh, measurement_name)
        self._output_file_handles.setdefault(prod_name, {})[measurement_name] = dest_fh
        return output_filename

    def _set_band_metadata(self, dest_fh, measurement_name, band=1):
        start_date, end_date = self._task.time_period
        dest_fh.update_tags(band,
                            source_product=self._task.source_product_names(),
                            start_date='{:%Y-%m-%d}'.format(start_date),
                            end_date='{:%Y-%m-%d}'.format(end_date),
                            name=measurement_name)

    def _open_geotiff(self, prod_name, measurement_name, output_filename, num_bands=1):
        profile = self.default_profile.copy()
        dtype = self._get_dtype(prod_name, measurement_name)
        nodata = self._get_nodata(prod_name, measurement_name)
        x_block_size = self._storage['chunking']['x'] if 'x' in self._storage['chunking'] else \
            self._storage['chunking']['longitude']
        y_block_size = self._storage['chunking']['y'] if 'y' in self._storage['chunking'] else \
            self._storage['chunking']['latitude']
        profile.update({
            'blockxsize': x_block_size,
            'blockysize': y_block_size,

            'dtype': dtype,
            'nodata': nodata,
            'width': self._geobox.width,
            'height': self._geobox.height,
            'transform': self._geobox.affine,
            'crs': self._geobox.crs.crs_str,
            'count': num_bands
        })
        _LOG.debug("Opening %s for writing.", output_filename)
        dest_fh = rasterio.open(str(output_filename), mode='w', **profile)
        dest_fh.update_tags(created=self._app_info)
        return dest_fh

    def write_data(self, prod_name, measurement_name, chunk, values):
        super(GeoTiffOutputDriver, self).write_data(prod_name, measurement_name, chunk, values)

        prod = self._output_file_handles[prod_name]
        if isinstance(prod, dict):
            output_fh = prod[measurement_name]
            band_num = 1
        else:
            output_fh = prod
            stat = self._output_products[prod_name]
            band_num = list(stat.product.measurements).index(measurement_name) + 1

        t, y, x = chunk
        window = ((y.start, y.stop), (x.start, x.stop))
        _LOG.debug("Updating %s.%s %s", prod_name, measurement_name,
                   "({})".format(", ".join(prettier_slice(x) for x in chunk[1:])))

        dtype = self._get_dtype(prod_name, measurement_name)

        output_fh.write(values.astype(dtype), indexes=band_num, window=window)

    def write_global_attributes(self, attributes):
        for dest in self._output_file_handles.values():
            dest.update_tags(**attributes)


class TestOutputDriver(OutputDriver):
    def write_global_attributes(self, attributes):
        pass

    def write_data(self, prod_name, measurement_name, chunk, values):
        pass

    def open_output_files(self):
        pass


def atomic_rename(src, dest):
    """Wrap boltons.fileutils.atomic_rename to allow passing  `str` or `pathlib.Path`"""
    _LOG.info('renaming %s to %s', src, dest)
    fileutils.replace(str(src), str(dest))
