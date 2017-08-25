import click

NUM_CPUS_PER_NODE = 16
_l_flags = 'mem ncpus walltime wd'.split(' ')

_pass_thru_keys = 'name project queue env_vars wd noask _internal'.split(' ')
_valid_keys = _pass_thru_keys + 'walltime nodes mem extra_qsub_args'.split(' ')


class QsubLauncher(object):
    def __init__(self, params, internal_args=None):
        self._internal_args = internal_args
        self._params = params

    def __repr__(self):
        import yaml
        return yaml.dump(dict(qsub=self._params))

    def __call__(self, *args):
        if self._internal_args is not None:
            args = args + tuple(self._internal_args)

        r, output = qsub_self_launch(self._params, *args)
        click.echo(output)
        return r


class QSubParamType(click.ParamType):
    name = 'opts'

    def convert(self, value, param, ctx):
        from .qsub import norm_qsub_params, parse_comma_args

        try:
            p = parse_comma_args(value, _valid_keys)

            if 'wd' not in p:
                p['wd'] = True

            p = norm_qsub_params(p)
            return QsubLauncher(p, ('--qsub', '_internal=celery'))
        except ValueError:
            self.fail('Failed to parse: {}'.format(value), param, ctx)


def parse_comma_args(s, valid_keys=None):
    import re

    def parse_one(a):
        kv = tuple(s.strip() for s in re.split(' *[=:] *', a))
        if len(kv) == 1:
            kv = (kv[0], True)

        if len(kv) != 2:
            raise ValueError('Bad option: ' + a)

        if valid_keys:
            k = kv[0]
            if k not in valid_keys:
                raise ValueError('Unexpected key:'+k)

        return kv

    return dict(parse_one(a) for a in re.split('[,;\n]', s) if a != '')


def normalise_walltime(x):
    import re

    if x is None or x.find(':') >= 0:
        return x

    m = re.match('^([0-9]+) *([hms]|min|minutes|hours)?$', x)
    if m is None:
        return None

    aliases = {'hours': 'h',
               None: 'h',
               'min': 'm',
               'minutes': 'm'}

    scale = dict(h=60*60,
                 m=60,
                 s=1)

    def fmt(secs):
        h = secs//(60*60)
        m = (secs//60) % 60
        s = secs % 60
        return '{}:{:02}:{:02}'.format(h, m, s)

    v, units = m.groups()
    units = aliases.get(units, units)
    return fmt(int(v)*scale[units])


def normalise_mem(x):
    import re

    named = dict(small=2,
                 medium=4,
                 large=7.875)

    if x in named:
        return named[x]

    m = re.match('^ *([0-9]+) *([g|G][bB]*)* *$', x)
    if m is None:
        return None
    return int(m.groups()[0])


def norm_qsub_params(p):
    from pydash import pick

    nodes = int(p.get('nodes', 1))
    ncpus = nodes*NUM_CPUS_PER_NODE

    mem = normalise_mem(p.get('mem', 'small'))
    mem = int((mem*NUM_CPUS_PER_NODE*1024 - 512)*nodes)
    mem = '{}MB'.format(mem)

    walltime = normalise_walltime(p.get('walltime'))

    extra_qsub_args = p.get('extra_qsub_args', [])
    if type(extra_qsub_args) == str:
        extra_qsub_args = extra_qsub_args.split(' ')

    pp = dict(ncpus=ncpus,
              mem=mem,
              walltime=walltime,
              extra_qsub_args=extra_qsub_args)

    pp.update(pick(p, _pass_thru_keys))

    return pp


def build_qsub_args(**p):
    args = []

    flags = dict(project='-P',
                 queue='-q',
                 name='-N')

    def add_l_arg(n):
        v = p.get(n)
        if v is not None:
            if type(v) is bool:
                if v:
                    args.append('-l{}'.format(n))
            else:
                args.append('-l{}={}'.format(n, v))

    def add_arg(n):
        v = p.get(n)
        if v is not None:
            flag = flags[n]
            args.extend([flag, v])

    for n in _l_flags:
        add_l_arg(n)

    for n in flags.keys():
        add_arg(n)

    args.extend(p.get('extra_qsub_args', []))

    # TODO: deal with env_vars!

    return args


def self_launch_args(*args):
    """
    Build tuple in the form (current_python, current_script, *args)
    """
    import sys
    from pathlib import Path

    py_file = str(Path(sys.argv[0]).absolute())
    return (sys.executable, py_file) + args


def generate_self_launch_script(*args):
    from ..utils import pbs
    s = "#!/bin/bash\n\n"
    s += pbs.generate_env_header()
    s += '\n\nexec ' + ' '.join("'{}'".format(s) for s in self_launch_args(*args))
    return s


def qsub_self_launch(qsub_opts, *args):
    from subprocess import Popen, PIPE

    script = generate_self_launch_script(*args)
    qsub_args = build_qsub_args(**qsub_opts)

    noask = qsub_opts.get('noask', False)

    if not noask:
        click.echo('Args: ' + ' '.join(map(str, args)))
        confirmed = click.confirm('Submit to pbs?')
        if not confirmed:
            return (0, 'Aborted by user')

    proc = Popen(['qsub'] + qsub_args, stdin=PIPE, stdout=PIPE)
    proc.stdin.write(script.encode('utf-8'))
    proc.stdin.close()
    out_txt = proc.stdout.read().decode('utf-8')
    exit_code = proc.wait()

    return exit_code, out_txt


class TaskRunner(object):
    def __init__(self, kind='serial', opts=None):
        self._kind = kind
        self._opts = opts
        self._executor = None
        self._shutdown = None
        self._queue_size = None

    def __repr__(self):
        args = '' if self._opts is None else '-{}'.format(str(self._opts))
        return '{}{}'.format(self._kind, args)

    def start(self):
        from ..utils import pbs
        from datacube.executor import SerialExecutor, _get_concurrent_executor

        def noop():
            pass

        def mk_pbs_celery():
            qsize = pbs.preferred_queue_size()
            executor, shutdown = pbs.launch_redis_worker_pool()
            return (executor, qsize, shutdown)

        def mk_multiproc():
            qsize = 100
            executor = _get_concurrent_executor(self._opts)
            return (executor, qsize, noop)

        def mk_serial():
            qsize = 10
            executor = SerialExecutor()
            return (executor, qsize, noop)

        mk = dict(pbs_celery=mk_pbs_celery,
                  multiproc=mk_multiproc,
                  serial=mk_serial)

        try:
            (self._executor,
             self._queue_size,
             self._shutdown) = mk.get(self._kind, mk_serial)()
        except RuntimeError:
            return False

    def stop(self):
        if self._shutdown is not None:
            self._shutdown()
            self._executor = None
            self._queue_size = None
            self._shutdown = None

    def __call__(self, tasks, run_task, on_task_complete=None):
        from datacube.ui.task_app import run_tasks

        if self._executor is None:
            if self.start() is False:
                raise RuntimeError('Failed to launch worker pool')

        return run_tasks(tasks, self._executor, run_task, on_task_complete, self._queue_size)


def get_current_obj(ctx=None):
    if ctx is None:
        ctx = click.get_current_context()

    if ctx.obj is None:
        ctx.obj = {}
    return ctx.obj


def with_qsub_runner():
    """
    Will add the following options

    --parallel <int>
    --qsub <qsub-params>

    Will populate variables
      qsub   - None | QsubLauncher
      runner - None | TaskRunner
    """

    from functools import update_wrapper

    rkey = '_runner'
    qkey = '_qsub'

    arg_name = 'runner'

    def add_multiproc_executor(ctx, param, value):
        if value is None:
            return
        obj = get_current_obj(ctx)
        obj[rkey] = TaskRunner('multiproc', value)

    def check_qsub(ctx, param, value):
        if value is None:
            return

        obj = get_current_obj(ctx)

        if '_internal' in value._params:
            obj[rkey] = TaskRunner('pbs_celery')
            return None  # erase

        obj[qkey] = value
        return value

    def decorate(f):
        opts = [
           click.option('--parallel',
                        type=int,
                        help='Run locally in parallel',
                        callback=add_multiproc_executor,
                        expose_value=False),
           click.option('--qsub',
                        type=QSubParamType(),
                        help='Launch via qsub',
                        callback=check_qsub),
        ]

        for o in opts:
            f = o(f)

        def get_runner():
            obj = get_current_obj()
            r = obj.get(rkey)
            if r is None:
                if qkey in obj:
                    return None

                return TaskRunner()
            return r

        def extract_runner(*args, **kwargs):
            kwargs.update({arg_name: get_runner()})
            return f(*args, **kwargs)

        return update_wrapper(extract_runner, f)
    return decorate
