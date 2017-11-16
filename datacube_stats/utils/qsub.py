import click
import logging
from time import sleep

from functools import update_wrapper

from datacube.executor import (SerialExecutor,
                               mk_celery_executor,
                               _get_concurrent_executor,
                               _get_distributed_executor)

from datacube import _celery_runner as cr

from digitalearthau import pbs
from digitalearthau.qsub import get_current_obj
from digitalearthau.qsub import HostPort
from digitalearthau.qsub import run_tasks
from digitalearthau.qsub import QSubParamType

_LOG = logging.getLogger(__name__)


def launch_redis_worker_pool(port=6379, **redis_params):
    redis_port = port
    redis_host = pbs.hostname()
    redis_password = cr.get_redis_password(generate_if_missing=True)

    redis_shutdown = cr.launch_redis(redis_port, redis_password, **redis_params)

    _LOG.info('Launched Redis at %s:%d', redis_host, redis_port)

    if not redis_shutdown:
        raise RuntimeError('Failed to launch Redis')

    for i in range(5):
        if cr.check_redis(redis_host, redis_port, redis_password) is False:
            sleep(0.5)

    executor = cr.CeleryExecutor(
        redis_host,
        redis_port,
        password=redis_password)

    worker_env = pbs.get_env()
    worker_procs = []

    for node in pbs.nodes():
        nprocs = node.num_cores
        if node.is_main:
            nprocs = max(1, nprocs - 2)

        celery_worker_script = 'exec datacube-worker --executor celery {}:{} --nprocs {} >/dev/null 2>/dev/null'.format(
            redis_host, redis_port, nprocs)
        proc = pbs.pbsdsh(node.offset, celery_worker_script, env=worker_env)
        worker_procs.append(proc)

    def shutdown():
        cr.app.control.shutdown()

        _LOG.info('Waiting for workers to quit')

        # TODO: time limit followed by kill
        for p in worker_procs:
            p.wait()

        _LOG.info('Shutting down redis-server')
        redis_shutdown()

    return executor, shutdown


class TaskRunner(object):
    def __init__(self, kind='serial', opts=None):
        self._kind = kind
        self._opts = opts
        self._executor = None
        self._shutdown = None
        self._queue_size = None
        self._user_queue_size = None

    def __repr__(self):
        args = '' if self._opts is None else '-{}'.format(str(self._opts))
        return '{}{}'.format(self._kind, args)

    def set_qsize(self, qsize):
        self._user_queue_size = qsize

    def start(self):
        def noop():
            pass

        def mk_pbs_celery():
            qsize = pbs.preferred_queue_size()
            port = 6379  # TODO: randomise
            maxmemory = "4096mb"  # TODO: compute maxmemory from qsize
            executor, shutdown = launch_redis_worker_pool(port=port, maxmemory=maxmemory)
            return (executor, qsize, shutdown)

        def mk_dask():
            qsize = 100
            executor = _get_distributed_executor(self._opts)
            return (executor, qsize, noop)

        def mk_celery():
            qsize = 100
            executor = mk_celery_executor(*self._opts)
            return (executor, qsize, noop)

        def mk_multiproc():
            qsize = 100
            executor = _get_concurrent_executor(self._opts)
            return (executor, qsize, noop)

        def mk_serial():
            qsize = 10
            executor = SerialExecutor()
            return (executor, qsize, noop)

        mk = dict(pbs_celery=mk_pbs_celery,
                  celery=mk_celery,
                  dask=mk_dask,
                  multiproc=mk_multiproc,
                  serial=mk_serial)

        try:
            (self._executor,
             default_queue_size,
             self._shutdown) = mk.get(self._kind, mk_serial)()
        except RuntimeError:
            return False

        if self._user_queue_size is not None:
            self._queue_size = self._user_queue_size
        else:
            self._queue_size = default_queue_size

    def stop(self):
        if self._shutdown is not None:
            self._shutdown()
            self._executor = None
            self._queue_size = None
            self._shutdown = None

    def __call__(self, tasks, run_task, on_task_complete=None):
        if self._executor is None:
            if self.start() is False:
                raise RuntimeError('Failed to launch worker pool')

        return run_tasks(tasks, self._executor, run_task, on_task_complete, self._queue_size)


def with_qsub_runner():
    """
    Will add the following options

    --parallel <int>
    --dask 'host:port'
    --celery 'host:port'|'pbs-launch'
    --queue-size <int>
    --qsub <qsub-params>

    Will populate variables
      qsub   - None | QSubLauncher
      runner - None | TaskRunner
    """

    arg_name = 'runner'
    o_key = '_qsub_state'

    class State:
        def __init__(self):
            self.runner = None
            self.qsub = None
            self.qsize = None

    def state(ctx=None):
        obj = get_current_obj(ctx)
        if o_key not in obj:
            obj[o_key] = State()
        return obj[o_key]

    def add_multiproc_executor(ctx, param, value):
        if value is None:
            return
        state(ctx).runner = TaskRunner('multiproc', value)

    def add_dask_executor(ctx, param, value):
        if value is None:
            return
        state(ctx).runner = TaskRunner('dask', value)

    def add_celery_executor(ctx, param, value):
        if value is None:
            return
        if value[0] == 'pbs-launch':
            state(ctx).runner = TaskRunner('pbs_celery')
        else:
            state(ctx).runner = TaskRunner('celery', value)

    def add_qsize(ctx, param, value):
        if value is None:
            return
        state(ctx).qsize = value

    def capture_qsub(ctx, param, value):
        state(ctx).qsub = value
        return value

    def decorate(f):
        opts = [
            click.option('--parallel',
                         type=int,
                         help='Run locally in parallel',
                         expose_value=False,
                         callback=add_multiproc_executor),
            click.option('--dask',
                         type=HostPort(),
                         help='Use dask.distributed backend for parallel computation. ' +
                         'Supply address of dask scheduler.',
                         expose_value=False,
                         callback=add_dask_executor),
            click.option('--celery',
                         type=HostPort(),
                         help='Use celery backend for parallel computation. ' +
                         'Supply redis server address, or "pbs-launch" to launch redis ' +
                         'server and workers when running under pbs.',
                         expose_value=False,
                         callback=add_celery_executor),
            click.option('--queue-size',
                         type=int,
                         help='Overwrite defaults for queue size',
                         expose_value=False,
                         callback=add_qsize),
            click.option('--qsub',
                         type=QSubParamType(),
                         callback=capture_qsub,
                         help='Launch via qsub, supply comma or new-line separated list of parameters.' +
                         ' Try --qsub=help.'),
        ]

        for o in opts:
            f = o(f)

        def finalise_state():
            s = state()
            if s.runner is None and s.qsub is None:
                s.runner = TaskRunner()

            if s.runner is not None and s.qsize is not None:
                s.runner.set_qsize(s.qsize)

            if s.qsub is not None and s.qsize is not None:
                s.qsub.add_internal_args('--queue-size', s.qsize)

        def extract_runner(*args, **kwargs):
            finalise_state()
            kwargs.update({arg_name: state().runner})
            return f(*args, **kwargs)

        return update_wrapper(extract_runner, f)
    return decorate
