from __future__ import absolute_import, print_function

import click
from datacube.ui.task_app import task_app, task_app_options, check_existing_files

_LOG = logging.getLogger('agdc-fc')


@click.command(name=APP_NAME)
@ui.pass_index(app_name=APP_NAME)
@click.option('--dry-run', is_flag=True, default=False, help='Check if output files already exist')
@click.option('--year', callback=validate_year, help='Limit the process to a particular year')
@click.option('--queue-size', '--backlog', type=click.IntRange(1, 100000), default=3200,
              help='Number of tasks to queue at the start')
@task_app_options
@task_app(make_config=make_fc_config, make_tasks=make_fc_tasks)
def fc_app(index, config, tasks, executor, dry_run, queue_size, *args, **kwargs):
    click.echo('Starting Stats processing...')

    if dry_run:
        check_existing_files((task['filename'] for task in tasks))
    else:
        results = []
        task_queue = itertools.islice(tasks, queue_size)
        for task in task_queue:
            _LOG.info('Running task: %s', task['tile_index'])
            results.append(executor.submit(do_fc_task, config=config, task=task))

        click.echo('Task queue filled, waiting for first result...')

        successful = failed = 0
        while results:
            result, results = executor.next_completed(results, None)

            # submit a new task to replace the one we just finished
            task = next(tasks, None)
            if task:
                _LOG.info('Running task: %s', task['tile_index'])
                results.append(executor.submit(do_fc_task, config=config, task=task))

            # Process the result
            try:
                datasets = executor.result(result)
                for dataset in datasets.values:
                    index.datasets.add(dataset, skip_sources=True)
                    _LOG.info('Dataset added')
                successful += 1
            except Exception as err:  # pylint: disable=broad-except
                _LOG.exception('Task failed: %s', err)
                failed += 1
                continue
            finally:
                # Release the task to free memory so there is no leak in executor/scheduler/worker process
                executor.release(result)

        click.echo('%d successful, %d failed' % (successful, failed))
