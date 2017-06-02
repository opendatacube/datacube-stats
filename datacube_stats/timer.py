import time
import psutil
from contextlib import contextmanager
from collections import defaultdict


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: %f ms' % self.msecs)


class MultiTimer(object):
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
        return 'Run times: {}, Max RSS: {}'.format(formatted_times, formatted_sizes)


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)
