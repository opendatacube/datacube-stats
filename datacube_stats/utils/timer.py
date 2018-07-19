import time
import psutil
from contextlib import contextmanager
from collections import defaultdict


class MultiTimer:
    def __init__(self):
        self._start_times = {}
        self.run_times = defaultdict(int)
        self.max_rss = defaultdict(int)
        self._proc = psutil.Process()

    @contextmanager
    def time(self, name):
        self.start(name)
        yield
        self.pause(name)

    def start(self, name):
        self._start_times[name] = time.time()
        return self

    def pause(self, name):
        self.run_times[name] += time.time() - self._start_times[name]
        rss = self._proc.memory_info().rss
        if rss > self.max_rss[name]:
            self.max_rss[name] = rss

    def __str__(self):
        formatted_sizes = {k: sizeof_fmt(v) for k, v in self.max_rss.items()}
        formatted_times = {k: '{:.0f}m {:.0f}s'.format(*divmod(v, 60)) for k, v in self.run_times.items()}
        return 'Run times: {}, Max RSS: {}'.format(prettier_dict(formatted_times), prettier_dict(formatted_sizes))


def prettier_dict(d):
    return "({})".format(", ".join("{}={}".format(key, value) for key, value in d.items()))


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def wrap_in_timer(func, timer, name):
    """
    If timer is none, return func, else returns a wrapper function that calls supplied function inside the timer.

    with timer.time(name):
       return func(...)
    """
    if timer is None:
        return func

    def wrapped(*args, **kwargs):
        with timer.time(name):
            return func(*args, **kwargs)

    return wrapped
