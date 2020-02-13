import xarray as xr
import numpy as np
import dask.array as da
from datacube.model import Measurement
from datacube.virtual.impl import Transformation
from osgeo import ogr
from osgeo import gdal
from osgeo import osr
import logging

_LOG = logging.getLogger(__name__)

class MaskByValue(Transformation):
    '''
    '''
    def __init__(self, mask_measurement_name, greater_than=None, smaller_than=None):
        self.greater_than = greater_than
        self.smaller_than = smaller_than
        if self.greater_than is not None and self.smaller_than is not None:
            if self.greater_than > self.smaller_than:
                raise Exception("greater_than should smaller than smaller_than")
        self.mask_measurement_name = mask_measurement_name

    def compute(self, data):
        if self.greater_than is not None:
            results = data[self.mask_measurement_name].where(
                            data[self.mask_measurement_name] > self.greater_than, -9999)
        else:
            results = data[self.mask_measurement_name]

        if self.smaller_than is not None:
            results = results.where(results < self.smaller_than, -9999)

        results = results > -9999
        results.attrs['crs'] = data.attrs['crs']
        return results

    def measurements(self, input_measurements):
        if self.mask_measurement_name not in list(input_measurements.keys()):
            raise Exception("have to mask by the band in product")

        return {self.mask_measurement_name: Measurement(name=self.mask_measurement_name,
                                                        dtype='bool', nodata=0, units=1)}


class TCIndex(Transformation):
    '''
    '''
    def __init__(self, category='wetness', coeffs=None):
        self.category = category
        if coeffs is None:
            self.coeffs = {
                 'brightness': {'blue': 0.2043, 'green': 0.4158, 'red': 0.5524, 'nir': 0.5741,
                                'swir1': 0.3124, 'swir2': 0.2303},
                 'greenness': {'blue': -0.1603, 'green': -0.2819, 'red': -0.4934, 'nir': 0.7940,
                               'swir1': -0.0002, 'swir2': -0.1446},
                 'wetness': {'blue': 0.0315, 'green': 0.2021, 'red': 0.3102, 'nir': 0.1594,
                             'swir1': -0.6806, 'swir2': -0.6109}
            }
        else:
            self.coeffs = coeffs
        self.var_name = f'TC{category[0].upper()}'

    def compute(self, data):
        tci_var = 0
        for var, key in zip(data.data_vars, self.coeffs[self.category].keys()):
            nodata = getattr(data[var], 'nodata', -1)
            data[var] = data[var].where(data[var] > nodata)
            tci_var += data[var] * self.coeffs[self.category][key]
        tci_var.data[da.isnan(tci_var.data)] = -9999
        tci_var = tci_var.astype(np.float32)
        tci_var.name = self.var_name
        tci_var.attrs = dict(nodata=-9999, units=1, crs=data.attrs['crs'])
        tci_var = tci_var.to_dataset()
        tci_var.attrs['crs'] = data.attrs['crs']
        return tci_var

    def measurements(self, input_measurements):
        return {self.var_name: Measurement(name=self.var_name, dtype='float32', nodata=-9999, units='1')}


class MangroveCC(Transformation):
    def __init__(self, thresholds, shape_file, bands=None):
        self.thresholds = thresholds
        if bands is None:
            self.bands = ['extent', 'canopy_cover_class']
        else:
            self.bands = bands
        self.shape_file = shape_file

    def measurements(self, input_measurements):
        output_measurements = dict()
        for band in self.bands:
            output_measurements[band] = Measurement(name=band, dtype='int16', nodata=-1, units='1')
        return output_measurements

    def compute(self, data):
        var_name = list(data.data_vars.keys())[0]

        rast_data = data[var_name].where(self.generate_rasterize(data[var_name]) == 1, -9999)

        cover_extent = rast_data.astype(np.int16).copy(True)
        cover_extent.data[rast_data.data <= self.thresholds[0]] = -1
        cover_extent.data[rast_data.data > self.thresholds[0]] = 1
        cover_extent.data[da.logical_and(rast_data.data > -9999, data[var_name].data == -2)] = 0
        cover_extent.attrs = dict(nodata=-1, units=1, crs=data.attrs['crs'])

        cover_type = rast_data.astype(np.int16).copy(True)
        cover_type.data[rast_data.data <= self.thresholds[0]] = -1
        level_threshold = 1
        for s_t in self.thresholds:
            cover_type.data[rast_data.data > s_t] = level_threshold
            level_threshold += 1
        cover_type.data[da.logical_and(rast_data.data > -9999, data[var_name].data == -2)] = 0
        cover_type.attrs = dict(nodata=-1, units=1, crs=data.attrs['crs'])

        outputs = {}
        outputs[self.bands[0]] = cover_extent
        outputs[self.bands[1]] = cover_type
        return xr.Dataset(outputs, attrs=dict(crs=data.crs))

    def generate_rasterize(self, data):
        source_ds = ogr.Open(self.shape_file)
        source_layer = source_ds.GetLayer()

        yt, xt = data.shape[1:]
        xres = 25
        yres = -25
        no_data = 0

        xcoord = data.coords['x'].min()
        ycoord = data.coords['y'].max()
        geotransform = (xcoord - (xres*0.5), xres, 0, ycoord - (yres*0.5), 0, yres)

        target_ds = gdal.GetDriverByName('MEM').Create('', xt, yt, gdal.GDT_Byte)
        target_ds.SetGeoTransform(geotransform)
        albers = osr.SpatialReference()
        albers.ImportFromEPSG(3577)
        target_ds.SetProjection(albers.ExportToWkt())
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(no_data)

        gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])
        dask_array = da.asarray(band.ReadAsArray()).rechunk(data.data.chunksize[1:])
        return dask_array

class ndvi_clim_mean(Transformation):
    """
    Calculate rolling quarterly NDVI mean climatolgies

    """
    def __init__(self):

        self.quarter= {'JFM': [1,2,3],
                       'FMA': [2,3,4],
                       'MAM': [3,4,5],
                       'AMJ': [4,5,6],
                       'MJJ': [5,6,7],
                       'JJA': [6,7,8],
                       'JAS': [7,8,9],
                       'ASO': [8,9,10],
                       'SON': [9,10,11],
                       'OND': [10,11,12],
                       'NDJ': [11,12,1],
                       'DJF': [12,1,2],
                      }
        self.var_names = list(self.quarter.keys())

    def compute(self, data):
        _LOG.info("length of input data:%s", str(len(data.time.values)))
        def attrs_reassign(da, dtype=np.float32):
            da_attr = data.attrs
            da = da.assign_attrs(**da_attr)
            return da

        data = data.where(data != -999)
        _LOG.info("data array %s", data)

        ndvi = xr.Dataset(data_vars={'ndvi': (data.nbart_nir - data.nbart_red) / (data.nbart_nir + data.nbart_red)},
                              coords=data.coords,
                              attrs=dict(crs=data.crs))

        ndvi_var = []
        for q in self.quarter.keys():
            ix = ndvi['time.month'].isin(self.quarter[q])
            ndvi_clim_mean = ndvi.where(ix, drop = True).mean(dim='time')
            ndvi_clim_mean = ndvi_clim_mean.to_array(name=q).drop('variable').squeeze()
            ndvi_var.append(ndvi_clim_mean)

        q_clim_mean = xr.merge(ndvi_var)
        #assign back attributes
        q_clim_mean.attrs = data.attrs
        q_clim_mean = q_clim_mean.apply(attrs_reassign, keep_attrs=True)
        _LOG.info("results %s", q_clim_mean)

        return q_clim_mean

    def measurements(self, input_measurements):
        output_measurements = dict()
        for m_name in self.var_names:
            output_measurements[m_name] = Measurement(name=m_name, dtype='float32', nodata=-999, units='1')

        return output_measurements
