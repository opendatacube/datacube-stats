"""
This module is based on the main() method code from FC/NDVI/ingest for running
tasks using `task_app` and extracts it into a reusable function called `run_task`.

Hopefully this can be pushed back into `task_app` to reduce duplication, and allow
any bugs or improvements to be fixed in only one place.
"""
from __future__ import absolute_import, print_function
import logging

_LOG = logging.getLogger(__name__)


def add_dataset_to_db(index, datasets):
    for dataset in datasets.values:
        index.datasets.add(dataset, skip_sources=True)
        _LOG.info('Dataset added')
