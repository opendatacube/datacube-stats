"""
Create tile indices for datacube-stats.

CRS: EPSG:3577

Tile size: 60X60KM^2 for sentinel 2
120X120KM^2 for LSAT

Rules of indices:
    x = topleft.x / 60,000
    y = topleft.y / 60,000
Index origin:
    (0, 0) = (0., 0.)

spatial bound for Australia(see https://cmi.ga.gov.au/node/185)
    Minimum X value
    -1943830.00
    Maximum X value
    2170690.00
    Minimum Y value
    -1119030.00
    Maximum Y value
    -4856630.00
"""

import numpy as np
import click
from os import path

# [[tl.x, tl.y],
#  [br.x, br.y]]

AU_EXTENT = np.array([[-1943830., -1119030.], [2170690., -4856630.]])
TILE_SIZE = 60000.


def make_tile_indices(product, output, extent=AU_EXTENT, tile_size=TILE_SIZE):
    x_indices = np.floor(extent[:, 0]/tile_size).astype(np.int)
    y_indices = np.ceil(extent[:, 1]/tile_size).astype(np.int)
    tl_index = np.array([x_indices[0], y_indices[0]])
    br_index = np.array([x_indices[1], y_indices[1]])
    click.echo('Generate tiles for product {}'.format(product))
    click.echo('Output tiles to {}'.format(output))
    if product == 'sentinel2':
        # two tiles buffer
        indices = np.array(np.meshgrid(np.arange(tl_index[0], br_index[0]+1+2),
                                       np.arange(tl_index[1]+2, br_index[1]-1-2, -1))).T.reshape(-1, 2)
        # save in the text file for now
        np.savetxt(path.join(output, 'sentinel2_tiles.txt'), indices.astype(np.int), fmt='%s')
    elif product == 'landsat':
        tl_index += tl_index % 2 * np.array([-1, 1])
        br_index += br_index % 2 * np.array([-1, 1])
        # two tiles buffer
        indices = np.array(np.meshgrid(np.arange(tl_index[0], br_index[0]+2+2, 2),
                                       np.arange(tl_index[1]+2, br_index[1]-2-2, -2))).T.reshape(-1, 2)
        # save in the text file for now
        np.savetxt(path.join(output, 'landsat_tiles.txt'), indices.astype(np.int), fmt='%s')
    else:
        click.echo('Dont know the product {}'.format(product))


# pylint: disable=broad-except
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
@click.command(name='make-tile-indices')
@click.option('--product', type=str, help='Override output location in configuration file')
@click.option('--output-location', type=str, help='Override output location in configuration file')
def main(product, output_location):
    make_tile_indices(product, output_location)


if __name__ == '__main__':
    main()
