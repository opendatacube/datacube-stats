import yaml
from datacube_stats.main import StatsApp
from datacube import Datacube


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
          tile: [15, -41]

    output_products:
        - name: nbar_mean
          statistic: simple
          statistic_args:
               reduction_function: mean
    """

    # or manually creating a config dictionary works too
    config = yaml.load(config_yaml)

    print(yaml.dump(config, indent=4))

    dc = Datacube()
    app = StatsApp(config, dc.index)

    print('generating tasks')
    tasks = app.generate_tasks()

    print('running tasks')
    for task in tasks:
        # this method is only available for the xarray output driver
        output = app.execute_task(task)
        print('result for {}'.format(task.tile_index))
        print(output.result['nbar_mean'])


if __name__ == '__main__':
    main()
