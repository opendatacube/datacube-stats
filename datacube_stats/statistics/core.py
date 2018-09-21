import abc
from typing import Iterable

import xarray

from datacube.model import Measurement


class StatsConfigurationError(RuntimeError):
    pass


class StatsProcessingError(RuntimeError):
    pass


class Statistic(abc.ABC):
    @abc.abstractmethod
    def compute(self, data: xarray.Dataset) -> xarray.Dataset:
        """
        Compute a statistic on the given Dataset.

        # FIXME: Explain a little bit better, Dataset in, Dataset out, measurements match measurements()

        :param xarray.Dataset data:
        :return: xarray.Dataset
        """

    def measurements(self, input_measurements: Iterable[Measurement]) -> Iterable[Measurement]:
        """
        Turn a list of input measurements into a list of output measurements.

        Base implementation simply copies input measurements to output_measurements.

        # FIXME: Explain the purpose of this

        :rtype: List[Measurement]
        """
        return input_measurements

    def is_iterative(self) -> bool:
        """
        Should return True if class supports iterative computation one time slice at a time.

        :rtype: Bool
        """
        return False

    def make_iterative_proc(self):
        """
        Should return `None` if `is_iterative()` returns `False`.

        Should return processing function `proc` that closes over internal
        state that get updated one time slice at time, if `is_iterative()`
        returns `True`.

        proc(dataset_slice)  # Update internal state, called many times
        result = proc()  # Extract final result, called once


        See `incremental_stats.assemble_updater`

        """


class PerPixelMetadata(abc.ABC):
    def __init__(self, var_name='observed'):
        self._var_name = var_name

    @abc.abstractmethod
    def compute(self, data, selected_indexes):
        """
        Return a variable name and :class:`xarray.Variable` to add in to the dataset.
        """

    @abc.abstractmethod
    def measurement(self):
        """
        Return band information on the per-pixel metadata.
        """


class SimpleStatistic(Statistic):
    """
    Describes the outputs of a statistic and how to calculate it

    :param stat_func:
        callable to compute statistics. Should both accept and return a :class:`xarray.Dataset`.
    """

    def __init__(self, stat_func):
        self.stat_func = stat_func

    def compute(self, data):
        return self.stat_func(data)
