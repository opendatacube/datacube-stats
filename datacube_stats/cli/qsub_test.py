#!/usr/bin/env python
from __future__ import print_function

import click
from collections import namedtuple
import pathlib

import datacube
import datacube_stats
from datacube.executor import SerialExecutor, _get_concurrent_executor
from datacube.ui.task_app import run_tasks, wrap_task

from .qsub import with_qsub
from ..utils.pbs import _hostname
from ..utils import pbs

Task = namedtuple('Task', 'val'.split(' '))
Result = namedtuple('Result', 'result val op worker'.split(' '))


def random_sleep(amount_secs=0.1, prop=0.5):
    """emulate processing time variance"""
    from time import sleep
    from random import uniform

    if uniform(0, 1) < prop:
        sleep(amount_secs)


def task_generator(num_tasks):
    for i in range(num_tasks):
        click.echo('Generating task: {}'.format(i))
        yield Task(i)._asdict()


def run_task(task, op):
    """ Runs across multiple cpus/nodes
    """
    from math import sqrt

    if type(task) != Task:
        task = Task(**task)

    host = _hostname()

    ops = {'sqrt': sqrt,
           'pow2': lambda x: x*x}

    random_sleep(1, 0.1)  # Sleep for 1 second 10% of the time

    val = task.val

    if val == 666:
        click.echo('Injecting failure')
        raise IOError('Fake IO Error')

    result = ops[op](val)
    click.echo('{} => {}'.format(val, result))

    return Result(result=result,
                  val=val,
                  op=op,
                  worker=host)


def log_completed_task(result):
    click.echo('From [{worker}]: {val} => {op} => {result}'.format(**result._asdict()))


@click.command(help='TODO')
@click.argument('app_config', nargs=1, type=str)
@click.option('--op', help='Configure dummy task: sqrt|pow2', default='sqrt')
@with_qsub
@click.option('--parallel', type=int, help='Run in parallel on local machine')
@click.option('--pbs-celery', is_flag=True, help='Launch worker pool when running on PBS')
def main(app_config, op, qsub=None, parallel=None, pbs_celery=False):
    if qsub:
        qsub.dump_options()
        return qsub('--pbs-celery',
                    '--op', op,
                    app_config)

    shutdown = None
    executor = None
    qsize = 100

    if pbs.is_under_pbs():
        qsize = pbs.preferred_queue_size()

    try:
        num_tasks = int(app_config)
    except ValueError:
        num_tasks = qsize*10

    if pbs_celery:
        click.echo('Launching Redis worker pool')
        try:
            executor, shutdown = pbs.launch_redis_worker_pool()
        except RuntimeError:
            raise click.ClickException('Failed to launch redis worker pool')
    elif parallel is not None:
        executor = _get_concurrent_executor(parallel, use_cloud_pickle=True)
    else:
        executor = SerialExecutor()

    click.echo(datacube.__file__)
    click.echo(datacube_stats.__file__)
    click.echo('PWD:' + str(pathlib.Path('.').absolute()))
    click.echo('celery_flag: {}'.format('Y' if pbs_celery else 'N'))
    click.echo('queue size: {}'.format(qsize))
    click.echo('cfg: ' + app_config)

    do_task = wrap_task(run_task, op)

    run_tasks(task_generator(num_tasks),
              executor,
              do_task,
              log_completed_task,
              qsize)

    if shutdown is not None:
        click.echo('Calling shutdown hook')
        shutdown()

    click.echo('All done!')
    return 0


if __name__ == '__main__':
    main()
