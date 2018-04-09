from ..schema import stats_schema
from datacube.ui import click as ui
from datacube.ui.click import to_pathlib
from datacube.utils import read_documents

from .util import Config, TaskProducer, TaskConsumer


def load_config(config_file):
    _, config = next(read_documents(config_file))
    stats_schema(config)
    config_obj = Config(config)
    return config_obj

#
# maybe later
#


def generate_tasks(config):
    pass


def load_tasks(tasks_file):
    pass


def run_tasks(config, tasks, slice_no=None, dump=False):
    pass
