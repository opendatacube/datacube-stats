from __future__ import absolute_import
try:
    from datacube.model import Product
except ImportError:
    from datacube.model import DatasetType as Product
from datacube_stats.statistics import STATS
import warnings


class StatsTask(object):
    """
    Contains everything a task runner requires to produce a single statistical output.
    Including:
      - Reference to all the source datasets
      - A list of `StatsProduct`s to create

    :param time_period: (start datetime, end datetime) tuple
    :param sources: List[dict] describing data/masks
    :param output_products: dict(product_name: OutputProduct)
    """

    def __init__(self, time_period, tile_index=None, sources=None, output_products=None, geom=None,
                 extra_fn_params=None):
        #: Start date - End date as a datetime tuple
        self.time_period = time_period

        self.tile_index = tile_index  # Only used for file naming... I think

        #: List of source datasets, required masking datasets, and details on applying them
        self.sources = sources if sources is not None else []

        #: Defines which files will be output, and what operations are done
        #: dict(product_name: OutputProduct)
        self.output_products = output_products if output_products is not None else []

        #: Optional geometry. Can be used to mask the loaded data
        self.geom = geom

        #: A dictionary of extra arguments to be used through the processing chain
        #: Will be available as named argument when producing the output filename
        self.extra_fn_params = extra_fn_params or {}

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

    def get(self, item, default=None):
        return getattr(self, item, default)

    def __str__(self):
        return "StatsTask(time_period={}, tile_index={})".format(self.time_period, self.tile_index)

    def __repr__(self):
        return self.__str__()


class DataSource(object):
    """A source was originally a dictionary containing:

      * data - Tile, Loadable by GridWorkflow
      * masks - List[Tile], loadable by GridWorkflow
      * spec - Source specification. Dictionary copy of the specification from the config file,
               containing details about which bands to load and how to apply masks.
    """
    def __init__(self, data, masks, spec):
        #: :type: Tile
        self.data = data

        #: :type: List[Tile]
        self.masks = masks

        #: Original Specification from configuration file
        #: :type: dict
        self.spec = spec

    def __getitem__(self, item):
        warnings.warn("Stop using dictionary based access for DataSource")
        return getattr(self, item)


class OutputProduct(object):
    """
    Defines an 'output_product' statistical product.
    Including:
      - Name
      - Statistical operation, ie max, mean, median, medoid. An implementation of `ValueStat`
      - Output product definition
      - Input measurements

    :param str product_type: Just a string tag, labelling the type of product
    :param dict stats_metadata: This will be copied to `metadata.stats` subtree in the product definition
    """

    def __init__(self, metadata_type, product_type, input_measurements, storage, name, file_path_template,
                 stat_name, statistic, output_params=None, extras=None, stats_metadata=None):

        #: The product name.
        self.name = name

        self.file_path_template = file_path_template

        #: The name of the statistic. Eg, mean, max, medoid, percentile_10
        self.stat_name = stat_name

        #: The implementation of a statistic. See :class:`Statistic`.
        #: Will provide `compute` and `measurements` functions.
        self.statistic = statistic

        self.data_measurements = statistic.measurements(input_measurements)

        #: The ODC Product (formerly DatasetType)
        self.product = self._create_product(metadata_type, product_type, self.data_measurements, storage,
                                            stats_metadata=stats_metadata or {})

        self.output_params = output_params

        #: A dictionary of extra arguments to be used through the processing chain
        #: Will be available as named argument when producing the output filename
        self.extras = extras or {}

    @classmethod
    def from_json_definition(cls, metadata_type, input_measurements, storage, definition, stats_metadata):
        return cls(metadata_type,
                   definition.get('product_type', '!!NOTSET!!'),
                   input_measurements, storage,
                   name=definition['name'],
                   file_path_template=definition.get('file_path_template'),
                   stat_name=definition['statistic'],
                   statistic=STATS[definition['statistic']](**definition.get('statistic_args', {})),
                   output_params=definition.get('output_params'),
                   stats_metadata=stats_metadata)

    @property
    def compute(self):
        return self.statistic.compute

    @property
    def is_iterative(self):
        return self.statistic.is_iterative

    @property
    def make_iterative_proc(self):
        return self.statistic.make_iterative_proc

    def _create_product(self, metadata_type, product_type, data_measurements, storage, stats_metadata):
        product_definition = {
            'name': self.name,
            'description': 'Description for ' + self.name,
            'metadata_type': metadata_type.name,
            'metadata': {
                # TODO: Setting format here is wrong. does it have to be set? Hopefully it's optional
                'format': {
                    'name': 'NetCDF'
                },
                'product_type': product_type,
                'statistics': stats_metadata,
            },
            'storage': storage,
            'measurements': data_measurements
        }
        Product.validate(product_definition)
        return Product(metadata_type, product_definition)

    def __str__(self):
        return "OutputProduct<name={}, stat_name={}>".format(self.name, self.stat_name)

    def __repr__(self):
        return self.__str__()
