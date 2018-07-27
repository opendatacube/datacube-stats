import multiprocessing
from functools import partial

import yaml
from datacube_stats import StatsApp
from datacube import Datacube


def execute_task(app, task):
    output = app.execute_task(task)
    print('result for {}:\n{}'.format(task.tile_index,
                                      output.result['nbar_mean']))


def main():
    config_yaml = """
    sources:
      - product: ls8_nbar_albers
        measurements: [red, green, blue]
        group_by: solar_day

    date_ranges:
        start_date: 2014-06-01
        end_date: 2014-07-01

    storage:
        # this driver enables in-memory computation
        driver: xarray

        crs: EPSG:3577
        tile_size:
            x: 40000.0
            y: 40000.0
        resolution:
            x: 25
            y: -25
        chunking:
            x: 200
            y: 200
            time: 1
        dimension_order: [time, y, x]

    computation:
        chunking:
            x: 800
            y: 800

    input_region:
          tiles:
            - [15, -41]
            - [15, -40]
            - [15, -41]
            - [15, -40]


    output_products:
        - name: nbar_mean
          statistic: simple
          statistic_args:
               reduction_function: mean
    """

    # or manually creating a config dictionary works too
    config = yaml.load(config_yaml)

    print(yaml.dump(config, indent=4))

    app = StatsApp(config)

    print('generating tasks')
    dc = Datacube(app='api-example')
    tasks = app.generate_tasks(dc.index)

    print('running tasks')
    pool = multiprocessing.Pool(4)
    pool.map(partial(execute_task, app), tasks)


if __name__ == '__main__':
    main()
