from datacube_stats import api
from datacube import Datacube
import matplotlib.pyplot as plt

dc = Datacube()
f_name = 'config.yaml'
config_obj = api.load_config(f_name)
print(config_obj.config)
config_obj.normalize_config()
task_generator = api.TaskProducer(dc, config_obj)
task_generator.validate()
tasks = task_generator.produce_tasks()
print('finish producing tasks')

consumer = api.TaskConsumer(config_obj)
consumer.validate()
source, results = consumer.consume_tasks(tasks)
print(source)
print(results)

prod = config_obj.config['output_products'][0]['name']
print(prod)

rgb = source['source']
rgb = rgb.to_array(dim='color')
rgb = rgb.transpose(*(rgb.dims[1:]+rgb.dims[:1]))
fake_saturation = 4000
rgb = rgb.where((rgb <= fake_saturation).all(dim='color'))
rgb /= fake_saturation  # scale to [0, 1] range for imshow
print(rgb)

rgb.plot.imshow(x='x', y='y', col='time')

rgb = results[prod]
rgb = rgb.to_array(dim='color')
rgb = rgb.transpose(*(rgb.dims[1:]+rgb.dims[:1]))
fake_saturation = 4000
rgb = rgb.where((rgb <= fake_saturation).all(dim='color'))
rgb /= fake_saturation  # scale to [0, 1] range for imshow
print(rgb)

rgb.plot.imshow(x='x', y='y', col='time')
