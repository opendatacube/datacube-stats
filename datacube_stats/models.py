from datacube.model import DatasetType
from datacube_stats.statistics import STATS


class StatsTask(object):
    """
    Contains everything a task runner requires to produce a single statistical output.
    Including:
      - Reference to all the source datasets
      - A list of `StatsProduct`s to create
    """

    def __init__(self, time_period, tile_index=None, sources=None, output_products=None, extras=None):
        #: Start date - End date as a datetime tuple
        self.time_period = time_period

        self.tile_index = tile_index  # Only used for file naming... I think

        #: List of source datasets, required masking datasets, and details on applying them
        self.sources = sources if sources is not None else []

        #: Defines which files will be output, and what operations are done
        #: dict(product_name: OutputProduct)
        self.output_products = output_products if output_products is not None else []

        #: A dictionary of extra arguments to be used through the processing chain
        #: Will be available as named argument when producing the output filename
        self.extras = extras or {}

    @property
    def geobox(self):
        return self.sources[0]['data'].geobox

    @property
    def sample_tile(self):
        return self.sources[0]['data']

    @property
    def time_attributes(self):
        return self.sources[0]['data'].sources.time.attrs

    def data_sources_length(self):
        return sum(len(d['data'].sources) for d in self.sources)

    def source_product_names(self):
        return ', '.join(source['data'].product.name for source in self.sources)

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return getattr(self, item)

    def __str__(self):
        return "StatsTask(time_period={}, tile_index={})".format(self.time_period, self.tile_index)

    def __repr__(self):
        return self.__str__()


class OutputProduct(object):
    """
    Defines an 'output_product' statistical product.
    Including:
      - Name
      - Statistical operation, ie max, mean, median, medoid. An implementation of `ValueStat`
      - Output product definition
      - Input measurements
    """

    def __init__(self, metadata_type, input_measurements, storage, name, file_path_template,
                 stat_name, statistic, output_params=None, extras=None):
        #: The product name.
        self.name = name

        self.file_path_template = file_path_template

        #: The name of the statistic. Eg, mean, max, medoid, percentile_10
        self.stat_name = stat_name

        #: The implementation of a statistic. See :class:`Statistic`.
        #: Will provide `compute` and `measurements` functions.
        self.statistic = statistic

        self.data_measurements = statistic.measurements(input_measurements)

        self.product = self._create_product(metadata_type, self.data_measurements, storage)
        self.output_params = output_params

        #: A dictionary of extra arguments to be used through the processing chain
        #: Will be available as named argument when producing the output filename
        self.extras = extras or {}

    @classmethod
    def from_json_definition(cls, metadata_type, input_measurements, storage, definition):
        return cls(metadata_type, input_measurements, storage,
                   name=definition['name'],
                   file_path_template=definition.get('file_path_template'),
                   stat_name=definition['statistic'],
                   statistic=STATS[definition['statistic']](**definition.get('statistic_args', {})),
                   output_params=definition.get('output_params'))

    @property
    def compute(self):
        return self.statistic.compute

    def _create_product(self, metadata_type, data_measurements, storage):
        product_definition = {
            'name': self.name,
            'description': 'Description for ' + self.name,
            'metadata_type': 'eo',
            'metadata': {
                'format': 'NetCDF',
                'product_type': self.stat_name,
            },
            'storage': storage,
            'measurements': data_measurements
        }
        DatasetType.validate(product_definition)
        return DatasetType(metadata_type, product_definition)

    def __str__(self):
        return "OutputProduct<name={}, stat_name={}>".format(self.name, self.stat_name)

    def __repr__(self):
        return self.__str__()
