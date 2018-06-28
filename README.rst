Data Cube Statistics Tools
##########################

|Build Status| |Coverage Status| |CodeCov Status|

Data Cube Statistics is a an application used to calculate large scale temporal statistics
on data stored using an `Open Data Cube`_ (`ODC`_) installation. It provides a
command line application which uses a `YAML <https://en.wikipedia.org/wiki/YAML>`_ configuration
file to specify the data range and statistics to calculate.


.. contents::

.. .. section-numbering::


Main features
=============

* Calculate continental scale statistics over a 30+ year time range
* Simple yet powerful `Configuration format`_
* Scripts supplied for running in a HPC PBS based environment
* Flexible tiling and chunking for efficient execution of 100+Tb jobs
* Track and store full provenance record of operations
* Round-trip workflow from `ODC`_ back into ODC
* Supports saving to `NetCDF`_, `GeoTIFF`_ or other GDAL supported format
* Optional per-pixel metadata tracking
* Out of the box support for most common statistics - see `Available statistics`_
* Able to create user defined `Custom statistics`_
* Able to handle any `CRS`_ and resolution combination (through the power of the `ODC`_)


Installation
============

.. code-block:: bash

    $ pip install https://github.com/GeoscienceAustralia/datacube-stats/

Usage
=====

Simplest case
-------------

At it's simplest, Data Cube Statistics only requires specifying a configuration file:

.. code-block:: bash

    $ datacube-stats example-configuration.yaml

If a configuration file is not provided and a file named ``config.yaml`` is found in the 
current directory, then it will be used automatically.

More detailed usage information is also available:

.. code-block:: bash

    $ datacube-stats --help


Parallel processing
-------------------

Parallel operation is provided by executors in ODC. For example to run across 4 cores:

.. code-block:: bash

    $ datacube-stats --parallel 4 example-configuration.yaml

Overrides for testing
---------------------

For tiled jobs, you can specify a single tile as a test run:

.. code-block:: bash

    $ datacube-stats --tile-index [X] [Y] example-configuration.yaml


Also useful when testing stats configurations, you can override the output directory:

.. code-block:: bash

    $ datacube-stats --output-location /home/user/example_folder/ example-configuration.yaml

Listing available Statistics
----------------------------

.. code-block:: bash

    $ datacube-stats --list-statistics


Configuration format
====================

Sources
-------

Specify the product/s of interest, measurements of interest, and any masks to be applied.

A simple example loading a single measurement from a single product:

.. code-block:: yaml

    sources:
      - product: old_wofs
        measurements: [water]
        group_by: solar_day

A (much) more complicated example which combines Landsat 5 and Landsat 7 data,
with filtering based on particular flags in a Pixel Quality layer, as well as
eliminating data which doesn't meet the minimum required spatial accuracy:

.. code-block:: yaml

    sources:
      - product: ls5_nbar_albers
        measurements: [blue, green, red, nir, swir1, swir2]
        group_by: solar_day
        source_filter:
          product: ls5_level1_scene
          gqa_iterative_mean_xy: [0, 1]
        masks:
          - product: ls5_pq_albers
            measurement: pixelquality
            group_by: solar_day
            fuse_func: datacube.helpers.ga_pq_fuser
            flags:
              contiguous: True
              cloud_acca: no_cloud
              cloud_fmask: no_cloud
              cloud_shadow_acca: no_cloud_shadow
              cloud_shadow_fmask: no_cloud_shadow
              blue_saturated: False
              green_saturated: False
              red_saturated: False
              nir_saturated: False
              swir1_saturated: False
              swir2_saturated: False
      - product: ls7_nbar_albers
        measurements: [blue, green, red, nir, swir1, swir2]
        group_by: solar_day
        source_filter:
          product: ls7_level1_scene
          gqa_iterative_mean_xy: [0, 1]
        masks:
          - product: ls7_pq_albers
            measurement: pixelquality
            group_by: solar_day
            fuse_func: datacube.helpers.ga_pq_fuser
            flags:
              contiguous: True
              cloud_acca: no_cloud
              cloud_fmask: no_cloud
              cloud_shadow_acca: no_cloud_shadow
              cloud_shadow_fmask: no_cloud_shadow
              blue_saturated: False
              green_saturated: False
              red_saturated: False
              nir_saturated: False
              swir1_saturated: False
              swir2_saturated: False


Discrete Values / No-Data Masking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, null or no-data values are automatically masked out, according to their definition in the Data Cube Product they are loaded from. In some cases this doesn't make sense, and can be disabled by specifying ``mask_nodata: False``. For example bitfield data like PQ and WOfS Extents that have a more complicated representation of no-data, which will be handled by the *statistic* being run over them.



.. code-block:: yaml

    sources:
      - product: ls5_pq_albers
        group_by: solar_day
        mask_nodata: False
        fuse_func: datacube.helpers.ga_pq_fuser
        group_by: solar_day



Date ranges
-----------

Single duration
~~~~~~~~~~~~~~~

Perform statistics over a single time range. The first date is inclusive and
the last date is exclusive.

.. code-block:: yaml

    date_ranges:
      start_date: 2000-01-01
      end_date: 2016-01-01

Multiple durations
~~~~~~~~~~~~~~~~~~

Or over a sequence of time steps, for example, an output for each year over
a 15 year period:

.. code-block:: yaml

    date_ranges:
      start_date: 2000-01-01
      end_date: 2016-01-01
      stats_duration: 1y
      step_size: 1y

Winter seasons in the southern hemisphere over the same 15 year period:

.. code-block:: yaml

    date_ranges:
      start_date: 2000-06-01
      end_date: 2016-09-01
      stats_duration: 3m
      step_size: 1y


Output location
---------------

Specify the base output directory where files will be written:

.. code-block:: yaml

    location: /home/user/mystats_outputs/


Output storage format
---------------------

NetCDF
~~~~~~

Able to write fully compliant `NetCDF-CF`_, either projected or unprojected spatially, with optional `Extra metadata attributes`_.

For example, to output 100×100km tiles, with 25m per pixel:

.. code-block:: yaml

    storage:
      driver: NetCDF CF

      crs: EPSG:3577
      tile_size:
          x: 100000.0
          y: 100000.0
      resolution:
          x: 25
          y: -25
      chunking:
          x: 200
          y: 200
          time: 1
      dimension_order: [time, y, x]

GeoTIFF
~~~~~~~

Write GeoTIFF files for each defined output. Side car `dataset metadata documents`_ in YAML format will be written which include
the provenance information and allow re-indexing into the Data Cube.

Output 1°×1° tiles, with 4000×4000 pixels per tile:

.. code-block:: yaml

    storage:
      driver: GeoTIFF

      crs: EPSG:4326
      tile_size:
          longitude: 1.0
          latitude: 1.0
      resolution:
          longitude: 0.00025
          latitude: -0.00025
      chunking:
          longitude: 400
          latitude: 400
          time: 1
      dimension_order: [time, latitude, longitude]


Computation/memory usage
------------------------

Adjust the size of the spatial chunks that are loaded into memory. This setting can be adjusted depending on the time depth
being processed, the available memory on the processing machine, and how many simultaneous tasks are being run on the machine.

.. code-block:: yaml

    computation:
      chunking:
        longitude: 1000
        latitude: 1000

Input area of interest (optional)
---------------------------------

Shapefile
~~~~~~~~~

An easy way to create a Shapefile is to use `GeoJSON.io <http://geojson.io>`_, draw your region of interest,
then from the top menu ``Save -> Shapefile`` to download the zipped Shapefile.

.. code-block:: yaml

    input_region:
      from_file: /home/user/mdb_floodplan/mdb_floodplain.shp

Whether the output will be gridded (tile-based, default) or not (feature-based) may be specified by
setting ``gridded: true`` or ``gridded: false`` respectively. The features to generate output for
may also be specified (in which case the output is feature-based),

.. code-block:: yaml

    input_region:
      from_file: /home/user/mdb_floodplan/mdb_floodplain.shp
      feature_id: [39]


Tile index
~~~~~~~~~~

The tiling regime is determined by the ``tile_size`` parameter of the `Output storage format`_ section.
A list of tiles can be passed on to ``tiles`` parameter.

.. code-block:: yaml

    input_region:
      tiles:
        - [16, -39]
        - [17, -39]


Period of interest
~~~~~~~~~~~~~~~~~~

The time period can be specified for individual sensors to include only datasets for this period.
This can be sometime useful to exclude datasets for Landsat 7 due to SLC failure.

.. code-block:: yaml

    sources:
      product: ls7_nbar_albers
      name: intertidal_low
      measurements: [blue, green, red, nir, swir1, swir2]
      group_by: solar_day
      time: [1986-01-01, 2003-05-01]


Spatial extents
~~~~~~~~~~~~~~~

Specify the maximum and minimum spatial range. You must also specify the `CRS`_ to use, normally with an EPSG code,
this alters whether you are specifying x/y or latitude/longitude.


.. code-block:: yaml

    input_region:
       crs: EPSG:4326
       longitude: [147.1, 147.9]
       latitude: [-33, -34]

.. note::

    This method of specifying extents will output a *single* large file, *not* a set of tiles.


Everywhere
~~~~~~~~~~

Don't specify an ``input_region`` to process all available data.

GeoJSON
~~~~~~~

Use http://geojson.io/ to draw out a region of interest. Copy the ``geometry`` portion of the GeoJSON and paste
it into your configuration file under ``input_region``. An `example for Australia <http://bl.ocks.org/d/e3b2cb64c170c6e306cf272cf9a60e41>`_:

.. code-block:: yaml

    input_region:
        "geometry": {
            "type": "Polygon",
            "coordinates": [ [
                [ 143.26171875, -9.88227549342994 ],
                [ 129.7265625, -9.96885060854611 ],
                [ 125.859375, -12.554563528593656 ],
                [ 119.35546875000001, -18.22935133838667 ],
                [ 111.005859375, -22.350075806124853 ],
                [ 113.818359375, -36.17335693522159 ],
                [ 117.94921874999999, -36.52729481454623 ],
                [ 130.78125, -33.06392419812064 ],
                [ 135.966796875, -37.43997405227057 ],
                [ 147.041015625, -44.59046718130883 ],
                [ 154.248046875, -34.234512362369856 ],
                [ 154.775390625, -24.5271348225978 ],
                [ 143.26171875, -9.88227549342994 ]
              ] ]
          }

Output products/which statistics
--------------------------------

This section of the configuration file specifies both which statistics to calculate, and which files to write them out to.

For many statistics workflows, it takes longer to load the data into memory than it does to compute the result. For these cases
it makes sense to perform multiple computations on the same set of data, and so ``output_products`` is a list of outputs, but at
a minimum it only needs one definition.

Name
~~~~

Define the name of the output product. eg:

.. code-block:: yaml

    name: landsat_yearly_mean

Product type
~~~~~~~~~~~~

Optional field allows to specify ``product_type`` field of the output product.
Defaults to ``!!NOTSET!!``. This is needed when output is to be indexed into the
data cube.

.. code-block:: yaml

        product_type: seasonal_stats


Statistic/calculation
~~~~~~~~~~~~~~~~~~~~~

Specify which statistic to use, and optionally any arguments. For example, a simple mean:

.. code-block:: yaml

    statistic: simple
    statistic_args:
      reduction_function: mean

Output parameters
~~~~~~~~~~~~~~~~~

Any extra arguments to pass to the output driver for an individual output band:

.. code-block:: yaml

       output_params:
         zlib: True
         fletcher32: True

Filter product
~~~~~~~~~~~~~~

    **NOTE**: This feature is being deprecated. We expect to remove it in the next release
    after release 0.9b1 and replace it with something more general.

To filter out sources that correspond to any derived products. It currently supports two methods
to filter out list of dates. Filtering in hydrological months ('by_hydrological_months'), can be
used to filter months from July to November for the year after the dry or wet years collected from
the polygon. Specific month range can also be specified. Second method of filtering is 'by_tide_height',
which uses OTPS model to get tide_height:

.. code-block:: yaml

       filter_product:
         method: by_tide_height
         args:
           tide_range: 10
           tide_percent: 20

.. code-block:: yaml

       filter_product:
         method: by_hydrological_months
         args:
           type: dry
           months: ['07', '11']




File naming
~~~~~~~~~~~

Specify a template string used to name the output files. Uses the python ``format()`` string syntax, with the following placeholders available:


==============  ==============
  Placeholder    Description
==============  ==============
x                X Tile Index
y                Y Tile Index
epoch_start      Start date of the epoch, format using `strftime syntax`_
epoch_end        End date of the epoch, format using `strftime syntax`_
name             The product name given to this output product
stat_name        The name of the statistic used to compute this product
==============  ==============

For example:

.. code-block:: yaml

       file_path_template: '{y}_{x}/LS_PQ_COUNT_3577_{y}_{x}_{epoch_start:%Y-%m-%d}_{epoch_end:%Y-%m-%d}.nc'

Will output filenames similar to:

.. code-block:: bash

    10_15/LS_PQ_COUNT_3577_10_15_2010-01-01_2011-01-01.nc


Custom metadata
~~~~~~~~~~~~~~~

Specify arbitrary custom metadata to attach to the produced datasets.
This is useful to resolve product ambiguity when indexing the datasets
back to the datacube.

.. code-block:: yaml

      metadata:
          platform:
              code: LANDSAT-8


Complete example
~~~~~~~~~~~~~~~~

.. code-block:: yaml

    output_products:
     - name: landsat_seasonal_mean
       product_type: seasonal_stats
       statistic: simple
       statistic_args:
         reduction_function: mean
       output_params:
         zlib: True
         fletcher32: True
       file_path_template: 'SR_N_MEAN/SR_N_MEAN_3577_{x:02d}_{y:02d}_{epoch_start:%Y%m%d}.nc'

     - name: landsat_seasonal_medoid
       product_type: seasonal_stats
       statistic: medoid
       output_params:
         zlib: True
         fletcher32: True
       file_path_template: 'SR_N_MEDOID/SR_N_MEDOID_3577_{x:02d}_{y:02d}_{epoch_start:%Y%m%d}.nc'

     - name: landsat_seasonal_percentile_10
       product_type: seasonal_stats
       statistic: percentile
       statistic_args:
         q: 10
       output_params:
         zlib: True
         fletcher32: True
       file_path_template: 'SR_N_PCT_10/SR_N_PCT_10_3577_{x:02d}_{y:02d}_{epoch_start:%Y%m%d}.nc'


Extra metadata attributes
-------------------------

Additional metadata can be specified which will be written as
``global attributes`` into the output NetCDF file. For example:

.. code-block:: yaml

    global_attributes:
      institution: Commonwealth of Australia (Geoscience Australia)
      instrument: OLI
      keywords: AU/GA,NASA/GSFC/SED/ESD/LANDSAT,ETM+,TM,OLI,EARTH SCIENCE
      keywords_vocabulary: GCMD
      platform: LANDSAT-8
      publisher_email: earth.observation@ga.gov.au
      publisher_name: Section Leader, Operations Section, NEMO, Geoscience Australia
      publisher_url: http://www.ga.gov.au
      license: CC BY Attribution 4.0 International License
      coverage_content_type: physicalMeasurement
      cdm_data_type: Grid
      product_suite: Pixel Quality 25m




Available statistics
====================

* Any `reduction operation <http://xarray.pydata.org/en/stable/api.html#computation>`_ supported by `xarray <http://xarray.pydata.org>`_. eg:

    - mean
    - median
    - percentile

* High-dimensional medians implemented by the `hdmedians python package`_

    - Medoid
    - Geometric median

* Normalised difference statistics. eg. NDVI + statistic
* `Custom statistics`_

Custom statistics
=================

Statistics operations in Data Cube Statistics are implemented as Python Classes, which extends ``datacube_stats.statistics.Statistic``. Two
methods should be implemented, ``measurements()`` and ``compute()``.

measurements()
    Takes a list of measurements provided by the input product type, and returns a list
    of measurements that this class will produce when asked to compute a statistic over some data.

compute()
    Takes a ``xarray.Dataset`` containing some data that has been loaded, and returns another ``xarray.Dataset`` after doing some computation.
    The variables on the returned dataset must match the types specified by ``measurements()``.

For example, the following implementation requires its input data to contain a
variable named ``water``, and outputs data with a single variable named
``count_wet`` of type ``int16``. In the configuration file, we will need to pass
a list of values for ``water`` that indicate "wetness" as an argument named
``wet_values`` to the statistic.

.. code-block:: python

    import xarray
    from datacube_stats.statistics import Statistic

    class CountWet(Statistic):
        def __init__(self, wet_values):
            # list of values of 'water' that we count as "wet"
            assert len(wet_values) > 0, 'no wet values provided'

            self.wet_values = wet_values

        def compute(self, data):
            wet = xarray.zeros_like(data.water)

            for val in self.wet_values:
                wet += data.water == val

            return xarray.Dataset({'count_wet': wet.sum(dim='time')},
                                  attrs={'crs': data.crs})

        def measurements(self, input_measurements):
            assert 'water' in [m['name'] for m in input_measurements]

            wet = {'name': 'count_wet',
                   'dtype': 'int16',
                   'nodata': -1,
                   'units': '1'}

            return [wet]

Suppose the package that contains this implementation is called ``pseudo.example``,
and it is available in the Python path (with the current directory added). Then 
the configuration file could look like (eliding ``location``, ``computation``, 
and ``storage`` specifications)

.. code-block:: yaml

   sources:
     - product: wofs_albers
       name: wofs_dry
       measurements: [water]
       group_by: solar_day

   date_ranges:
     start_date: 2014-01-01
     end_date: 2014-02-01
     stats_duration: 1m
     step_size: 1m

   output_products:
    - name: wet_count_summary
      product_type: wofs_statistical_summary
      statistic: external
      statistic_args:
        impl: pseudo.example.CountWet

        # ignoring PQ sea mask that excludes a lot of useful data
        wet_values:
           - 128 # clear and wet
           - 132 # clear and wet and masked for sea
      output_params:
        zlib: True
        fletcher32: True
      file_path_template: 'WOFS_COUNT/{x}_{y}/WOFS_COUNT_3577_{x}_{y}_{epoch_start:%Y%m%d}_{epoch_end:%Y%m%d}.nc'


Running with PBS job scheduler
==============================

To submit a job to PBS, run ``datacube-stats`` like

.. code-block:: bash

    $ datacube-stats --qsub="project=u46,nodes=100,walltime=5h,mem=large,queue=normal" example.yaml

The ``mem`` specification can be ``small``, ``medium``, or ``large``, for 2GB, 4GB, or 8GB
memory per core respectively. For more details, run

.. code-block:: bash

    $ datacube-stats --qsub=help

Release Notes
=============

0.9b2 release
-------------
- The GeoTIFF output driver name has changed to ``GeoTIFF`` from ``GeoTiff``

0.9b1 release
-------------
    **Note:** We expect several backwards incompatible changes to the ``datacube-stats`` package
    in the near future. Release 0.9b1 is intended to be the last release fully supporting
    configurations from earlier releases.

- Add tasseled cap indices statistic
- Fix GeoTiff output driver
- Preliminary Python API and an ``xarray`` output driver to produce in-momery results
- Support default configurations (``config.yaml`` in the current working folder)
- Support discoverable external plugins (in the current working folder)

0.9a9 release
-------------
- Fix ``xarray`` sorting bug
- Add ability to specify ``num_threads`` to the ``new_geomedian`` statistic
- Add ability to attach custom metadata to generated datasets

0.9a8 release
-------------
- Add ability for feature-based task generation from a shapefile
- Fix issue with ``hdmedian`` GeoMedian statistics

0.9a7 release
-------------
- Move task execution code to the ``digitalearthau`` repository
- ITEM and low/high tide composite, FC percentile
- Schema-validated configuration

0.9a6 release
-------------
- Time filters on individual source products

0.9 release
-----------

* User documentation!
* List available statistics from the command line ``datacube-stats --list-statistics``




.. _DEA: http://www.ga.gov.au/about/projects/geographic/digital-earth-australia
.. _ODC: https://github.com/opendatacube/datacube-core
.. _Open Data Cube: https://github.com/opendatacube/datacube-core
.. _NetCDF-CF: http://cfconventions.org/
.. _CRS: https://en.wikipedia.org/wiki/Spatial_reference_system
.. _dataset metadata documents: http://datacube-core.readthedocs.io/en/stable/ops/config.html#dataset-metadata-document
.. _strftime syntax: http://strftime.org/
.. _hdmedians python package: https://github.com/daleroberts/hdmedians
.. |Build Status| image:: https://travis-ci.org/GeoscienceAustralia/agdc_statistics.svg?branch=master
   :target: https://travis-ci.org/GeoscienceAustralia/agdc_statistics
.. |Coverage Status| image:: https://coveralls.io/repos/github/GeoscienceAustralia/agdc_statistics/badge.svg?branch=master
   :target: https://coveralls.io/github/GeoscienceAustralia/agdc_statistics?branch=master
.. |CodeCov Status| image:: https://codecov.io/gh/GeoscienceAustralia/agdc_statistics/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/GeoscienceAustralia/agdc_statistics
