""" Module specifically to hold version information.  The reason this
exists is the version information is needed in both setup.py for install and
also in datacube_stats/__init__.py when generating results.  If these values were
defined in datacube_stats/__init__.py then install would fail because there are other
dependencies imported in datacube_stats/__init__.py that are not present until after
install. Do not import anything into this module."""

__version__ = '0.9a5'
