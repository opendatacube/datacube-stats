"""
rio stack equivalent


"""


import collections
import logging

import click
from cligj import files_inout_arg, format_opt

import rasterio
from rasterio.rio import options
from rasterio.rio.helpers import resolve_inout

_LOG = logging.getLogger('stacker')


@click.command(short_help="Stack bands into a multiband dataset.")
@files_inout_arg
@options.output_opt
@format_opt
@options.force_overwrite_opt
@options.creation_options
def stack(files, output, driver, force_overwrite,
          creation_options):
    """Stack all bands from one or more input files into a multiband dataset.

    Input datasets must be of a kind: same data type, dimensions, etc. The
    output is cloned from the first input.

    stacker will take all bands from each input and write them
    in same order to the output.

    Examples, using the Rasterio testing dataset, which produce a copy.

      stack RGB.byte.tif -o stacked.tif
    """
    try:
        output, files = resolve_inout(files=files, output=output,
                                      force_overwrite=force_overwrite)
        output_count = 0
        indexes = []
        for path in files:
            with rasterio.open(path) as src:
                indexes.append(src.indexes)
                output_count += len(src.indexes)

        with rasterio.open(files[0]) as first:
            kwargs = first.meta
            kwargs.update(**creation_options)

        kwargs.update(
            driver=driver,
            count=output_count)

        with rasterio.open(output, 'w', **kwargs) as dst:
            dst_idx = 1
            for path, index in zip(files, indexes):
                with rasterio.open(path) as src:
                    if isinstance(index, int):
                        data = src.read(index)
                        dst.write(data, dst_idx)
                        dst.update_tags(dst_idx, filename=path, description=path)
                        dst_idx += 1
                    elif isinstance(index, collections.Iterable):
                        data = src.read(index)
                        dst.write(data, range(dst_idx, dst_idx + len(index)))
                        for idx in range(dst_idx, dst_idx + len(index)):
                            dst.update_tags(idx, filename=path, description=path)
                        dst_idx += len(index)

    except Exception:
        _LOG.exception("Exception caught during processing")
        raise click.Abort()

if __name__ == '__main__':
    stack()
