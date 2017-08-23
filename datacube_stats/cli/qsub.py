import click

NUM_CPUS_PER_NODE = 16
_l_flags = 'mem ncpus walltime wd'.split(' ')

_pass_thru_keys = 'name project queue env_vars wd noask'.split(' ')
_valid_keys = _pass_thru_keys + 'walltime nodes mem extra_qsub_args'.split(' ')


class QsubLauncher(object):
    def __init__(self, params):
        self._params = params

    def dump_options(self):
        import yaml
        click.echo(yaml.dump(dict(qsub=self._params)))

    def __call__(self, *args):
        from .qsub import qsub_self_launch
        r, output = qsub_self_launch(self._params, *args)
        click.echo(output)
        return r


class QSubParamType(click.ParamType):
    name = 'QSUB params'

    def convert(self, value, param, ctx):
        from .qsub import norm_qsub_params, parse_comma_args

        try:
            p = parse_comma_args(value, _valid_keys)

            if 'wd' not in p:
                p['wd'] = True

            p = norm_qsub_params(p)
            return QsubLauncher(p)
        except ValueError:
            self.fail('Failed to parse: {}'.format(value), param, ctx)


QSUB = QSubParamType()

with_qsub = click.option('--qsub', type=QSUB,
                         help='Launch via qsub')


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
        click.echo('Args: ' + ' '.join(args))
        confirmed = click.confirm('Submit to pbs?')
        if not confirmed:
            return (0, 'Aborted by user')

    proc = Popen(['qsub'] + qsub_args, stdin=PIPE, stdout=PIPE)
    proc.stdin.write(script.encode('utf-8'))
    proc.stdin.close()
    out_txt = proc.stdout.read().decode('utf-8')
    exit_code = proc.wait()

    return exit_code, out_txt
