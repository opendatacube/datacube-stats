from shapely.geometry import Polygon, mapping
import fiona
import numpy as np
import sys
from fiona.crs import from_epsg


def main(input_txt, output_geojson, stride=1):
    output_driver = 'GeoJSON'
    tile_size = 60000.
    tile_schema = {'geometry': 'Polygon',
                   'properties': {'ID': 'int',
                                  'X_MAX': 'float',
                                  'X_MIN': 'float',
                                  'Y_MAX': 'float',
                                  'Y_MIN': 'float',
                                  'label': 'str'}}
    tile_shape = {}
    tiles = np.genfromtxt(input_txt, dtype=np.int32)
    with fiona.open(output_geojson, 'w', crs=from_epsg(3577), driver=output_driver, schema=tile_schema) as f:
        j = 0
        for e in tiles:
            j += 1
            tile_shape['geometry'] = mapping(Polygon([(e[0]*tile_size, e[1]*tile_size),
                                                      ((e[0]+stride)*tile_size, e[1]*tile_size),
                                                      ((e[0]+stride)*tile_size, (e[1]-stride)*tile_size),
                                                      (e[0]*tile_size, (e[1]-stride)*tile_size)]))
            tile_shape['properties'] = {'ID': j,
                                        'X_MIN': e[0]*tile_size,
                                        'X_MAX': (e[0]+stride)*tile_size,
                                        'Y_MIN': (e[1]-stride)*tile_size,
                                        'Y_MAX': e[1]*tile_size,
                                        'label': str(e[0])+','+str(e[1])}
            f.write(tile_shape)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('usage: write_tiles_shapefile.py input_list_file output_geojson stride(e.g., 1, 2,...)')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
