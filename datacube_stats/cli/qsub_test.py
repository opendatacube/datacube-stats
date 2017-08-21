#!/usr/bin/env python
from __future__ import print_function

import click
from .qsub import with_qsub


@click.command(help='TODO')
@click.argument('products', nargs=-1, type=str)
@with_qsub
@click.option('--pbs-celery', is_flag=True, help='Launch worker pool when running on PBS')
def main(products, qsub=None, pbs_celery=False):
    import datacube
    import datacube_stats
    import pathlib
    from ..utils import pbs

    if qsub:
        qsub.dump_options()
        return qsub('--pbs-celery', *products)

    qsize = 100

    if pbs.is_under_pbs():
        qsize = pbs.preferred_queue_size()

    click.echo(datacube.__file__)
    click.echo(datacube_stats.__file__)
    click.echo('PWD:' + str(pathlib.Path('.').absolute()))
    click.echo('celery_flag:{}'.format('Y' if pbs_celery else 'N'))
    click.echo('queue size: {}'.format(qsize))
    click.echo('  > ' + ' '.join(products))

    return 0


if __name__ == '__main__':
    main()
