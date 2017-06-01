Data Cube Statistics Tools
##########################

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

    $ pip install https://github.com/GeoscienceAustralia/agdc_statistics/

Usage
=====

Simplest case
-------------

At it's simplest, Data Cube Statistics only requires specifying a configuration file:

.. code-block:: bash

    $ datacube-stats example-configuration.yaml

Parallel processing
-------------------

Parallel operation is provided by executors in ODC. For example to run across 4 cores:

.. code-block:: bash

    $ datacube-stats --executor multiproc 4 example-configuration.yaml

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
          gqa: [-1, 1]
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
          gqa: [-1, 1]
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

Perform statistics over a single time range. For now it is required to specify
``stats_duration`` and ``step_size`` to cover the entire period.

.. code-block:: yaml

    date_ranges:
      start_date: 2000-01-01
      end_date: 2016-01-01
      stats_duration: 15y
      step_size: 15y


Or over a sequence of time steps, for example, an output for each year over a 15 year period:

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
      driver: NetCDFCF

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
      driver: Geotiff

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

.. code-block:: yaml

    input_region:
      from_file: /home/user/mdb_floodplan/mdb_floodplain.shp



Tile index
~~~~~~~~~~

The tiling regime is determined by the ``tile_size`` parameter of the `Output storage format`_ section.

.. code-block:: yaml

    input_region:
      tile: [16, -39]

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

    This input region does not perform tile based processing, and will result in a single output for the region.


Everywhere
~~~~~~~~~~

Don't specify any ``input_region`` to process all available data.

GeoJSON
~~~~~~~


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

Complete example
~~~~~~~~~~~~~~~~

.. code-block:: yaml

    output_products:
     - name: landsat_seasonal_mean
       statistic: mean
       output_params:
         zlib: True
         fletcher32: True
       file_path_template: 'SR_N_MEAN/SR_N_MEAN_3577_{x:02d}_{y:02d}_{epoch_start:%Y%m%d}.nc'

     - name: landsat_seasonal_medoid
       statistic: medoid
       output_params:
         zlib: True
         fletcher32: True
       file_path_template: 'SR_N_MEDOID/SR_N_MEDOID_3577_{x:02d}_{y:02d}_{epoch_start:%Y%m%d}.nc'

     - name: landsat_seasonal_percentile_10
       statistic: percentile_10
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

For example, the following implementation requires it's input data to contain a variable named ``water``, and outputs datasets with a single variable
named ``count_wet`` of type ``int16``. When passed appropriate data it counts the number of times that 132 or 128 occur.

.. code-block:: python

    class CountWet(Statistic):
        def compute(self, data):
            # 128 == clear and wet, 132 == clear and wet and masked for sea
            # The PQ sea mask that we use is dodgy and should be ignored. It excludes lots of useful data
            wet = ((data.water == 128) + (data.water == 132)).sum(dim='time')
            return xarray.Dataset({'count_wet': wet,
                                   attrs={'crs':data.crs})

        def measurements(self, input_measurements):
            measurement_names = set(m['name'] for m in input_measurements)
            assert 'water' in measurement_names

            wet = {'name': 'count_wet',
                   'dtype': 'int16',
                   'nodata': -1,
                   'units': '1'}
            return [wet]




Running with PBS job scheduler
==============================


Installation onto Raijin
========================

This section is only relevant for `DEA`_ deployment managers

Run the following after logging into ``raijin``.

.. code-block:: bash

    $ cd ansible
    $ ansible-playbook -v -v -i "localhost," -c local install-stats-module.yml

-v                Show verbose output
-i <hosts list>   Which hosts to run on, trailing ',' indicates list of one
-c                Connection type. local: run commands locally, not over SSH



Release Notes
=============

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
