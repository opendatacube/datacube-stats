#!/usr/bin/env python

from __future__ import print_function

import subprocess
from pathlib import Path
import shutil

import click

CPUS_PER_NODE = 16
MEMORY_PER_NODE = 62


@click.command(short_help='Submit a stats job to qsub',
               help='Submit a job to qsub for '
                    'an app_config (from the list command).')
@click.option('--queue', '-q', default='normal',
              type=click.Choice(['normal', 'express']))
@click.option('--project', '-P', default='v10')
@click.option('--nodes', '-n', required=True,
              help='Number of nodes to request',
              type=click.IntRange(1, 100))
@click.option('--walltime', '-t', default=10,
              help='Number of hours to request',
              type=click.IntRange(1, 48))
@click.option('--name', help='Job name to use')
@click.option('--config', help='Datacube config file',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--env', help='Node environment setup script',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--no-confirm', required=False, help="Don't ask for confirmation or perform file check",
              is_flag=True)
@click.option('--load-tasks', type=click.Path(exists=True, dir_okay=False))
@click.argument('app_config', type=click.Path(exists=True, dir_okay=False))
def qsub(app_config, queue, project, nodes, walltime, name, no_confirm, load_tasks, config=None, env=None):
    confirm = not no_confirm
    app_config = Path(app_config).absolute()

    config_arg = '-C "%s"' % config if config else ''
    env_arg = '--env "%s"' % env if env else ''

    do_system_check(config_arg)

    do_qsub(name, nodes, walltime, queue, project, config_arg, env_arg,
            app_config, load_tasks, confirm)


def do_system_check(config_arg):
    """Checks the connection to the database"""
    check_cmd = 'datacube %(config_arg)s -v system check' % dict(
        config_arg=config_arg,
    )
    subprocess.check_call(check_cmd, shell=True)


def do_qsub(name, nodes, walltime, queue, project, config_arg, env_arg, app_config,
            load_tasks=None, confirm=True):
    """Submits the job to qsub"""
    name = name
    app_cmd = ('datacube-stats -v {config_arg} '
               '--queue-size {queue_size} '
               '--dask DSCHEDULER {lt} {app_config}'.format(
                   config_arg=config_arg,
                   queue_size=nodes * CPUS_PER_NODE * 2,
                   app_config=app_config,
                   lt='--load-tasks {}'.format(load_tasks) if load_tasks else ''
               ))  # 'DSCHEDULER' is replaced by distributed.sh with the host/port for the dask scheduler

    distr_cmd = '"%(distr)s" %(env_arg)s --ppn 16 %(app_cmd)s' % dict(
        distr=shutil.which('launch-distributed-pbs'),
        env_arg=env_arg,
        app_cmd=app_cmd,
    )

    l_args = 'ncpus=%(ncpus)d,mem=%(mem)dgb,walltime=%(walltime)d:00:00,other=%(other)s' % dict(
        ncpus=nodes * CPUS_PER_NODE,
        mem=nodes * MEMORY_PER_NODE,
        walltime=walltime,
        other='gdata1:gdata2',
    )

    qsub = 'qsub -q %(queue)s -N %(name)s -P %(project)s -l %(l_args)s -- /bin/bash %(distr_cmd)s'
    cmd = qsub % dict(queue=queue,
                      name=name,
                      project=project,
                      l_args=l_args,
                      distr_cmd=distr_cmd,
                      )

    if not confirm or click.confirm('\n' + cmd + '\nRUN?', default=True):
        subprocess.check_call(cmd, shell=True)


if __name__ == '__main__':
    qsub()
