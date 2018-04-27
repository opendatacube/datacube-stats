import yaml
from datacube_stats.main import read_config, normalize_config, StatsApp
from datacube import Datacube

config_yaml = """
sources:
  - product: ls8_nbar_albers
    measurements: [red, green, blue]
    group_by: solar_day

date_ranges:
    start_date: 2014-06-01
    end_date: 2014-07-01

storage:
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

config = yaml.load(config_yaml)
print(yaml.dump(config, indent=4))

dc = Datacube()
app = StatsApp(config, dc.index)
app.validate()

print('generating tasks')
tasks = app.generate_tasks()

print('running tasks')
app.run_tasks(tasks)

# this is only available for the None output driver
nbar_mean = app.output_driver.result['nbar_mean']
print('result of computation')
print(nbar_mean)
