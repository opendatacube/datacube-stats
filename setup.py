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
    url='https://github.com/data-cube/',
    license='Apache',
    author='Geosience Australia',
    author_email='datacube@ga.gov.au',
    description='Perform statistics operations on a Data Cube',
    install_requires=['xarray', 'click', 'pandas', 'numpy', 'datacube', 'rasterio', 'pyyaml',
                      'cloudpickle', 'boltons', 'pydash'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'mock'],
    entry_points={
        'console_scripts': [
            'datacube-stats = datacube_stats.main:main',
            'datacube-stats-qsub = datacube_stats.cli.datacube_stats_qsub:qsub',
            'datacube-tile-check = datacube_stats.cli.tile_check:main',
            'dc-qsub-test = datacube_stats.cli.qsub_test:main',
        ],
        'datacube.stats': [
            'wofs-summary = datacube_stats.statistics:WofsStats'
        ]
    },
    scripts=['scripts/launch-distributed-pbs'],
)
