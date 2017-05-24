Data Cube Statistics Tools
##########################

Data Cube Statistics is a an application used to calculate large scale temporal statistics
on data stored using an `Open Data Cube <https://github.com/>`_ (ODC) installation. It provides a
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
* Full support for saving to `NetCDF-CF`_, GeoTIFF or other GDAL supported format
* Optional per-pixel metadata tracking
* Out of the box support for most common statistics - `Available statistics`_
* Able to create `Custom statistics`_
* Able to handle any CRS and resolution combination (through the power of the `ODC`_)


Installation
============


Usage
=====

At it's simplest, Data Cube Statistics only requires specifying a configuration file:

.. code-block:: bash

    $ datacube-stats example-configuration.yaml


Parallel operation is provided by executors in ODC. For example to run across 4 cores:

.. code-block:: bash

    $ datacube-stats --executor multiproc 4 example-configuration.yaml


For tiled jobs, you can specify a single tile as a test run:

.. code-block:: bash

    $ datacube-stats --tile-index [X] [Y] example-configuration.yaml


Also useful when testing stats configurations, you can override the output directory:

.. code-block:: bash

    $ datacube-stats --output-location /home/user/example_folder/ example-configuration.yaml



Configuration format
====================

Sources
-------

Specify the product/s of interest, measurements of interest, and any masks to be applied.

.. code-block:: yaml

    sources:
      - product: old_wofs
        name: wofs_dry
        measurements: [water]
        group_by: solar_day


Date ranges
-----------

Perform statistics over a single time range.

.. code-block:: yaml

    date_ranges:
      start_date: 2000-01-01
      end_date: 2016-01-01
      stats_duration: 15y
      step_size: 15y


Or over a sequence of time steps, for example, yearly:

.. code-block:: yaml

    date_ranges:
      start_date: 2000-01-01
      end_date: 2016-01-01
      stats_duration: 1y
      step_size: 1y

Winter seasons in the southern hemisphere:

.. code-block:: yaml

    date_ranges:
      start_date: 2000-06-01
      end_date: 2016-09-01
      stats_duration: 3m
      step_size: 1y


Output location
---------------

Specify the base output directory where files will be written.

.. code-block:: yaml

    location: /home/user/mystats_outputs/


Output storage format
---------------------

* `NetCDF-CF`_ in projected or unprojected, with custom resolution
* GeoTIFF

Computation/memory usage
------------------------

Adjust the size of the spatial chunks that are loaded into memory.

.. code-block:: yaml

    computation:
      chunking:
        longitude: 1000
        latitude: 1000

Input area of interest (optional)
---------------------------------

* Shapefile
* Everywhere
* geojson
* tile index
* Spatial extents + CRS

Output products/which statistics
--------------------------------




Available statistics
====================


Custom statistics
=================


.. _ODC: https://github.com/opendatacube/datacube-core
.. _NetCDF-CF: http://cfconventions.org/
