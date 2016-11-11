from __future__ import absolute_import, print_function

import click
import itertools

import logging

_LOG = logging.getLogger('agdc-fc')


def add_dataset_to_db(index, datasets):
    for dataset in datasets.values:
        index.datasets.add(dataset, skip_sources=True)
        _LOG.info('Dataset added')


def run_tasks(tasks, executor, run_task, process_result, queue_size=50):
    click.echo('Starting processing...')
    results = []
    task_queue = itertools.islice(tasks, queue_size)
    for task in task_queue:
        _LOG.info('Running task: %s', task['tile_index'])
        results.append(executor.submit(run_task, task=task))

    click.echo('Task queue filled, waiting for first result...')

    successful = failed = 0
    while results:
        result, results = executor.next_completed(results, None)

        # submit a new task to replace the one we just finished
        task = next(tasks, None)
        if task:
            _LOG.info('Running task: %s', task['tile_index'])
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
            # Release the task to free memory so there is no leak in executor/scheduler/worker process
            executor.release(result)

    click.echo('%d successful, %d failed' % (successful, failed))
