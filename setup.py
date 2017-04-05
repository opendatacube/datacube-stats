from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))


# bring in __version__ from version.py for install.
with open(path.join(here, 'datacube_stats', 'version.py')) as h:
    __version__ = None
    exec(h.read())

setup(
    name='datacube-stats',
    version=__version__,
    packages=['datacube_stats'],
    url='https://github.com/data-cube/',
    license='Apache',
    author='Geosience Australia',
    author_email='datacube@ga.gov.au',
    description='Perform statistics operations on a Data Cube',
    install_requires=['xarray', 'click', 'pandas', 'numpy', 'datacube', 'rasterio', 'pyyaml', 'cloudpickle', ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'mock'],
    entry_points={
        'console_scripts': [
            'datacube-stats = datacube_stats.main:main'
        ],
        'datacube.stats': [
            'wofs-summary = datacube_stats.statistics:WofsStats'
        ]
    },
)
