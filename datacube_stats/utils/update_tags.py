#!/usr/bin/env python
import click
import rasterio


@click.command()
@click.argument('output-file')
@click.argument('file-names', nargs=-1)
def copy_tags(output_file, file_names):
    click.echo("Updating tags in {}".format(output_file))

    with rasterio.open(output_file, 'r+') as dest:
        assert dest.count == len(file_names)

        with click.progressbar(file_names, label='Copying band tags') as bar:
            for i, name in enumerate(bar):
                with rasterio.open(name) as src:
                    dest.update_tags(i + 1, **src.tags(1))


if __name__ == '__main__':
    copy_tags()