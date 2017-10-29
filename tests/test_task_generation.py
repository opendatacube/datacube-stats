from mock import MagicMock

from datacube.utils.geometry import Geometry, CRS
from datacube_stats.main import NonGriddedTaskGenerator, ArbitraryTileMaker, GriddedTaskGenerator

EXAMPLE_STORAGE = {'crs': 'EPSG:4326', 'resolution': {'latitude': 0.1, 'longitude': 0.1},
                   'tile_size': {'latitude': 1, 'longitude': 1}}
EXAMPLE_SOURCES_SPEC = [{'product': 'fake_product'}]
EXAMPLE_DATE_RANGE = ('2000-01-01', '2001-01-01')

BIG_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [
                106.875,
                -45.58328975600631
            ],
            [
                160.3125,
                -45.58328975600631
            ],
            [
                160.3125,
                -6.664607562172573
            ],
            [
                106.875,
                -6.664607562172573
            ],
            [
                106.875,
                -45.58328975600631
            ]
        ]
    ]
}

class FakeDataset:
    extent = Geometry(BIG_POLYGON, crs=CRS('EPSG:4326'))
    center_time = object()

def test_gridded_task_generation_no_data(mock_index):
    gen = GriddedTaskGenerator(storage=EXAMPLE_STORAGE)

    tasks = gen(mock_index, EXAMPLE_SOURCES_SPEC, EXAMPLE_DATE_RANGE)

    assert list(tasks) == []


def test_gridded_task_generation_with_datasets(mock_index):
    mock_index.datasets.search_eager.return_value = [FakeDataset()]
    gen = GriddedTaskGenerator(storage=EXAMPLE_STORAGE)

    tasks = gen(mock_index, EXAMPLE_SOURCES_SPEC, EXAMPLE_DATE_RANGE)

    tasks = list(tasks)
    assert tasks

def xtest_non_gridded_task_generation():
    gen = NonGriddedTaskGenerator()



def test_arbitrary_tile_maker(mock_index):
    input_region = {
        'crs': 'EPSG:4326',
        'latitude': [-30, -31],
        'longitude': [137, 138.5]
    }
    atm = ArbitraryTileMaker(mock_index, input_region, storage=EXAMPLE_STORAGE)

    tile = atm('product', EXAMPLE_DATE_RANGE, group_by='time')
    assert tile.dims == ('time', 'latitude', 'longitude')
    assert tile.geobox.width == 15
    assert tile.geobox.height == 10
    assert len(tile.sources.time) == 0


def xtest_arbitrary_tile_maker_with_sources():
    mock_index = MagicMock()
    mock_index.datasets.get_field_names.return_value = ['time']  # Check is performed validating the name of query fields
    mock_index.datasets.search_eager.return_value = [set('source1'), set('source2')] # TODO: Expects actual datasets with .crs and .extents
    input_region = {
        'crs': 'EPSG:4326',
        'latitude': [-30, -31],
        'longitude': [137, 138.5]
    }
    atm = ArbitraryTileMaker(mock_index, input_region, storage={'crs': 'EPSG:4326', 'resolution': {'latitude': 0.1, 'longitude': 0.1}})

    tile = atm('product', ('2000-01-01', '2001-01-01'), group_by='time')
    assert tile.dims == ('time', 'latitude', 'longitude')
    assert tile.geobox.width == 15
    assert tile.geobox.height == 10
    assert len(tile.sources.time) == 2



