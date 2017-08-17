#!/usr/bin/env python

from __future__ import print_function
import click
import pickle
import time


def ds_to_key(ds, cell_idx, solar_day):
    return '{}|{:+03},{:+03}'.format(solar_day, *cell_idx)


def flat_map_ds(proc, obs):
    from datacube_stats.utils import tile_flatten_sources
    from datacube.api.query import solar_day

    for cell_idx, tile in obs.items():
        for ds in tile_flatten_sources(tile):
            yield proc(ds, cell_idx=cell_idx, solar_day=solar_day(ds))


def flat_foreach_ds(proc, obs):
    for _ in flat_map_ds(proc, obs):
        pass


@click.command()
@click.argument('products', nargs=-1, type=str)
@click.option('--year', type=int)
@click.option('--month', type=int)
@click.option('--save', type=str, nargs=1)
def main(products, year, month, save):
    from datacube_stats.utils.query import multi_product_list_cells
    import datacube
    from datacube.api import GridWorkflow

    query = {}
    if year is not None:
        if month is not None:
            query['time'] = ('{}-{}-01'.format(year, month),
                             '{}-{}-01'.format(year, month+1))
        else:
            query['time'] = ('{}-01-01'.format(year),
                             '{}-12-31'.format(year))

    dc = datacube.Datacube(app='dbg')
    gw = GridWorkflow(product=products[0],
                      index=dc.index)

    click.echo('## Starting to run query', err=True)
    t_start = time.time()
    co_common, co_unmatched = multi_product_list_cells(products, gw, **query)
    t_took = time.time() - t_start
    click.echo('## Completed in {} seconds'.format(t_took), err=True)

    if save is not None:
        click.echo('## Saving data to {}'.format(save), err=True)
        with open(save, 'wb') as f:
            pickle.dump(dict(co_common=co_common, co_unmatched=co_unmatched), f)
            f.close()
        click.echo(' done')

    click.echo('## Processing results,  ...wait', err=True)

    coverage = set(flat_map_ds(ds_to_key, co_common[0]))
    um = set(flat_map_ds(ds_to_key, co_unmatched[0]))

    # These tiles have both matched and unmatched data on the same solar day
    # It's significant cause these are the ones that will interfere with
    # masking if masking is done the "usual way"
    um_with_siblings = um - (um - coverage)

    click.echo('## Found {} matched records and {} unmatched'.format(len(coverage), len(um)))
    click.echo('##   Of {} unmatched records {} are "dangerous" for masking'.
               format(len(um), len(um_with_siblings)))
    click.echo('##')

    def dump_unmatched_ds(ds, cell_idx, solar_day):
        k = ds_to_key(ds, cell_idx, solar_day)
        flag = '!' if k in coverage else '.'
        click.echo('{} {} {} {}'.format(k, flag, ds.id, ds.local_path))

    for (idx, product) in enumerate(products):
        click.echo('## unmatched ###########################')
        click.echo('## {}'.format(product))
        click.echo('########################################')
        flat_foreach_ds(dump_unmatched_ds, co_unmatched[idx])


if __name__ == '__main__':
    main()
