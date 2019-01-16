import os
import sys
import logging
from pydoc import locate
from typing import Iterable

import xarray

from datacube.model import Measurement
from datacube_stats.statistics.core import Statistic, StatsProcessingError


LOG = logging.getLogger(__name__)


class ExternalPlugin(Statistic):
    """
    Run externally defined plugin.
    """

    def __init__(self, impl, *args, **kwargs):
        # Temporarily, add current path
        sys.path.insert(0, os.getcwd())

        LOG.debug('Looking for external plugin `%s` in %s', impl, sys.path)
        impl_class = locate(impl)

        # Remove the path that was added
        sys.path = sys.path[1:]

        if impl_class is None:
            raise StatsProcessingError("Failed to load external plugin: '{}'".format(impl))
        else:
            LOG.debug('Found external plugin `%s`', impl)

        self.impl = impl_class(*args, **kwargs)

    def is_iterative(self) -> bool:
        return self.impl.is_iterative()

    def make_iterative_proc(self):
        return self.impl.make_iterative_proc()

    def measurements(self, input_measurements: Iterable[Measurement]) -> Iterable[Measurement]:
        return self.impl.measurements(input_measurements)

    def compute(self, data: xarray.Dataset) -> xarray.Dataset:
        return self.impl.compute(data)

    # caused trouble in unpickle stream
    # def __getattr__(self, name):
    #     # If attribute not on current object or on Statistic, try to find it on self.impl
    #     return getattr(self.impl, name)
