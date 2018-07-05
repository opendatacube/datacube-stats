import xarray as xr
import numpy as np
from osgeo import ogr
from osgeo import gdal
from osgeo import osr
from datacube.model import Measurement
from .statistics import Statistic


class MangroveCC(Statistic):
    def __init__(self, thresholds, shape_file, bands=None):
        super().__init__()
        self.thresholds = thresholds
        if bands is None:
            self.bands = ['extent', 'type']
        else:
            self.bands = bands
        self.shape_file = shape_file

    def measurements(self, input_measurements):
        return [Measurement(name=band, dtype='int32', nodata=0, units='1') for band in self.bands]

    def compute(self, data):
        var_name = list(data.data_vars.keys())[0]
        rast_data = data[var_name].where(self.generate_rasterize(data[var_name]) == 1)
        rast_data.data[np.isnan(rast_data.data)] = 0

        cover_extent = rast_data.copy(True)
        cover_extent.data = np.zeros(cover_extent.shape)
        cover_extent.data[rast_data.data > self.thresholds[0]] = 1

        cover_type = rast_data.copy(True)
        cover_type.data = np.zeros(cover_type.shape)
        level_threshold = 1
        for s_t in self.thresholds:
            cover_type.data[rast_data.data > s_t] = level_threshold
            level_threshold += 1

        outputs = {}
        outputs[self.bands[0]] = cover_extent
        outputs[self.bands[1]] = cover_type
        return xr.Dataset(outputs, attrs=dict(crs=data.crs))

    def generate_rasterize(self, data):
        source_ds = ogr.Open(self.shape_file)
        source_layer = source_ds.GetLayer()

        yt, xt = data[0].shape
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
        return band.ReadAsArray()
