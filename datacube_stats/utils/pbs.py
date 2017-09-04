from collections import namedtuple, OrderedDict
Node = namedtuple('Node', 'name num_cores offset is_main'.split(' '))

_nodes = None


def _hostname():
    import platform
    return platform.node()


def is_under_pbs():
    import os
    return 'PBS_NODEFILE' in os.environ


def parse_nodes_file(fname=None):
    import os

    if fname is None:
        fname = os.environ.get('PBS_NODEFILE')
        if fname is None:
            raise RuntimeError("Can't find PBS node file")

    def load_lines(fname):
        with open(fname, 'r') as f:
            ll = [l.strip() for l in f.readlines()]
            return [l for l in ll if len(l) > 0]

    hostname = _hostname()
    nodes = OrderedDict()

    for idx, l in enumerate(load_lines(fname)):
        if l in nodes:
            nodes[l]['num_cores'] += 1
        else:
            nodes[l] = dict(
                name=l,
                num_cores=1,
                offset=idx,
                is_main=(
                    hostname == l))

    return [Node(**x) for x in nodes.values()]


def nodes():
    global _nodes
    if _nodes is None:
        _nodes = parse_nodes_file()
    return _nodes


def total_cores():
    total = 0
    for n in nodes():
        total += n.num_cores
    return total


def preferred_queue_size():
    return total_cores()*2


def get_env(extras=[], **more_env):
    import os
    import re

    pass_envs = set(['PATH', 'LANG', 'LD_LIBRARY_PATH', 'HOME', 'USER'])
    REGEXES = ['^PYTHON.*', '^GDAL.*', '^LC.*', '^DATACUBE.*']
    rgxs = [re.compile(r) for r in REGEXES]

    def need_this_env(k):
        if k in pass_envs or k in extras:
            return True
        for rgx in rgxs:
            if rgx.match(k):
                return True
        return False

    ee = dict((k, v) for k, v in os.environ.items() if need_this_env(k))
    ee.update(**more_env)
    return ee


def mk_exports(env):
    return '\n'.join('export {}="{}"'.format(k, v) for k, v in env.items())


def generate_env_header(extras=[], **more_env):
    return mk_exports(get_env(extras, **more_env))


def wrap_script(script):
    from base64 import b64encode
    b64s = b64encode(script.encode('ascii')).decode('ascii')
    return 'eval "$(echo {}|base64 -d)"'.format(b64s)


def pbsdsh(cpu_num, script, env=None, test_mode=False):
    import subprocess

    if env is None:
        env = get_env()

    hdr = mk_exports(env) + '\n\n'

    if test_mode:
        args = "env -i bash --norc -c".split(' ')
    else:
        args = "pbsdsh -n {} -- bash -c".format(cpu_num).split(' ')

    args.append(wrap_script(hdr + script))
    return subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)


def launch_redis_worker_pool(port=6379):
    from time import sleep
    from datacube import _celery_runner as cr

    redis_port = port
    redis_host = _hostname()
    redis_password = cr.get_redis_password(generate_if_missing=True)

    redis_shutdown = cr.launch_redis(redis_port, redis_password)

    print('Launched redis at {}:{}'.format(redis_host, redis_port))

    if not redis_shutdown:
        raise RuntimeError('Failed to launch Redis')

    for i in range(5):
        if cr.check_redis(redis_host, redis_port, redis_password) is False:
            sleep(0.5)

    executor = cr.CeleryExecutor(
        redis_host,
        redis_port,
        password=redis_password)

    worker_env = get_env()
    worker_procs = []

    for node in nodes():
        nprocs = node.num_cores
        if node.is_main:
            nprocs = max(1, nprocs - 2)

        celery_worker_script = 'exec datacube-worker --executor celery {}:{} --nprocs {} >/dev/null 2>/dev/null'.format(
            redis_host, redis_port, nprocs)
        proc = pbsdsh(node.offset, celery_worker_script, env=worker_env)
        worker_procs.append(proc)

    def shutdown():
        cr.app.control.shutdown()

        print('Waiting for workers to quit')

        # TODO: time limit followed by kill
        for p in worker_procs:
            p.wait()

        print('Shutting down redis-server')
        redis_shutdown()

    return executor, shutdown
