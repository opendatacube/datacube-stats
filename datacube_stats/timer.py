import time
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

    def start(self, name):
        self._start_times[name] = time.time()

    def pause(self, name):
        self.run_times[name] += self._start_times[name] - time.time()


