"""
This module is based on the main() method code from FC/NDVI/ingest for running
tasks using `task_app` and extracts it into a reusable function called `run_taks`.

Hopefully this can be pushed back into `task_app` to reduce duplication, and allow
any bugs or improvements to be fixed in only one place.
"""
from __future__ import absolute_import, print_function

import click
import itertools

import logging

_LOG = logging.getLogger(__name__)


def add_dataset_to_db(index, datasets):
    for dataset in datasets.values:
        index.datasets.add(dataset, skip_sources=True)
        _LOG.info('Dataset added')


def run_tasks(tasks, executor, run_task, process_result, queue_size=50):
    """

    :param tasks: iterable of tasks. Usually a generator to create them as required.
    :param executor: a datacube executor, similar to `distributed.Client` or `concurrent.futures`
    :param run_task: the function used to run a task. Expects a single argument of one of the tasks
    :param process_result: a function to do something based on the result of a completed task. It
                           takes a single argument, the return value from `run_task(task)`
    :param queue_size: How large the queue of tasks should be. Will depend on how fast tasks are
                       processed, and how much memory is available to buffer them.
    """
    _LOG.debug('Starting running tasks...')
    results = []
    task_queue = itertools.islice(tasks, queue_size)
    for task in task_queue:
        _LOG.info('Running task: %s', task['tile_index'])
        results.append(executor.submit(run_task, task=task))

        _LOG.debug('Task queue filled, waiting for first result...')

    successful = failed = 0
    while results:
        result, results = executor.next_completed(results, None)

        # submit a new _task to replace the one we just finished
        task = next(tasks, None)
        if task:
            _LOG.info('Running _task: %s', task['tile_index'])
            results.append(executor.submit(run_task, task=task))

        # Process the result
        try:
            actual_result = executor.result(result)
            process_result(actual_result)
            successful += 1
        except Exception as err:  # pylint: disable=broad-except
            _LOG.exception('Task failed: %s', err)
            failed += 1
            continue
        finally:
            # Release the _task to free memory so there is no leak in executor/scheduler/worker process
            executor.release(result)

    _LOG.info('%d successful, %d failed' % (successful, failed))
