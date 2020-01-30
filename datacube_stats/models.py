from datacube.model import Measurement

import warnings


class OutputProduct:
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
    # pylint: disable=too-many-arguments
    def __init__(self, metadata_type, product_type, virtual_datasets, virtual_product,
                 storage, name, file_path_template,
                 output_params=None, extras=None, custom_metadata=None):

        #: The product name.
        self.name = name

        self.file_path_template = file_path_template

        #: The ODC Product (formerly DatasetType)
        self.datasets = virtual_datasets
        self.product = virtual_product
        output_measurements = [output for output in
                               virtual_product.output_measurements(virtual_datasets.product_definitions).values()]

        self.product_definition = self._create_product(metadata_type, product_type,
                                                       output_measurements, storage, custom_metadata)
        self.output_params = output_params

        #: A dictionary of extra arguments to be used through the processing chain
        #: Will be available as named argument when producing the output filename
        self.extras = extras or {}

    @classmethod
    def from_json_definition(cls, metadata_type, virtual_datasets, virtual_product, storage, definition, extras):
        return cls(metadata_type,
                   definition.get('product_type', '!!NOTSET!!'),
                   virtual_datasets, virtual_product, storage,
                   name=definition['name'],
                   file_path_template=definition.get('file_path_template'),
                   output_params=definition.get('output_params'),
                   custom_metadata=definition.get('metadata'),
                   extras=extras)

    @property
    def compute(self):
        return self.product.fetch

    def _create_product(self, metadata_type, product_type, data_measurements, storage,
                        custom_metadata):
        product_definition = {
            'name': self.name,
            'description': 'Description for ' + self.name,
            'metadata_type': metadata_type.name,
            'metadata': {
                'product_type': product_type,
                **custom_metadata
            },
            'storage': storage,
            'measurements': data_measurements
        }
        try:
            from datacube.model import Product
        except ImportError:
            from datacube.model import DatasetType as Product

        # Disable this due to a json schema validation bug in datacube-core
        # Product.validate(product_definition)
        return Product(metadata_type, product_definition)

    def __str__(self):
        return "OutputProduct<name={}>".format(self.name)

    def __repr__(self):
        return self.__str__()
