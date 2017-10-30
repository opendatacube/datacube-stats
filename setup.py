from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))


# bring in __version__ from version.py for install.
with open(path.join(here, 'datacube_stats', 'version.py')) as h:
    __version__ = None
    exec(h.read())

setup(
    name='datacube-stats',
    version=__version__,
    packages=find_packages(),
    url='https://github.com/GeoscienceAustralia/agdc_statistics/',
    license='Apache',
    author='Geoscience Australia',
    author_email='datacube@ga.gov.au',
    description='Perform statistics operations on a Data Cube',
    install_requires=['xarray', 'click', 'pandas', 'numpy', 'datacube', 'rasterio', 'pyyaml',
                      'cloudpickle', 'boltons', 'pydash', 'python-dateutil', 'fiona'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'mock', 'hypothesis', 'voluptuous', 'otps'],
    entry_points={
        'console_scripts': [
            'datacube-stats = datacube_stats.main:main',
            'datacube-tile-check = datacube_stats.cli.tile_check:main',
        ],
        'datacube.stats': [
            'wofs-summary = datacube_stats.statistics:WofsStats'
        ]
    },
)
