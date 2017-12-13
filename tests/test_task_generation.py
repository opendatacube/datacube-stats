from mock import MagicMock

from datacube.utils.geometry import Geometry, CRS
from datacube_stats.tasks import NonGriddedTaskGenerator, ArbitraryTileMaker, GriddedTaskGenerator, \
    select_task_generator
from dateutil.parser import parse

EXAMPLE_STORAGE = {'crs': 'EPSG:4326', 'resolution': {'latitude': 0.1, 'longitude': 0.1},
                   'tile_size': {'latitude': 1, 'longitude': 1}}
EXAMPLE_SOURCES_SPEC = [{'product': 'fake_product'}]
EXAMPLE_DATE_RANGE = [(parse('2000-01-01'), parse('2001-01-01'))]

BIG_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [
                100,
                -40
            ],
            [
                100,
                -50
            ],
            [
                150,
                -50
            ],
            [
                150,
                -40
            ],
            [
                100,
                -40
            ]
        ]
    ]
}


class FakeDataset:
    extent = Geometry(BIG_POLYGON, crs=CRS('EPSG:4326'))
    center_time = object()
    crs = CRS('EPSG:4326')


def test_gridded_task_generation_no_data(mock_index):
    gen = GriddedTaskGenerator(storage=EXAMPLE_STORAGE)

    tasks = gen(mock_index, EXAMPLE_SOURCES_SPEC, EXAMPLE_DATE_RANGE)

    assert list(tasks) == []


def test_gridded_task_generation_with_datasets(mock_index):
    mock_index.datasets.search_eager.return_value = [FakeDataset()]
    gridded_generator = GriddedTaskGenerator(storage=EXAMPLE_STORAGE)

    tasks = gridded_generator(mock_index, EXAMPLE_SOURCES_SPEC, EXAMPLE_DATE_RANGE)

    tasks = list(tasks)
    assert tasks
    assert len(tasks) == 500
    assert set(range(-50, -40)) == set(task.tile_index[1] for task in tasks)


def test_non_gridded_task_generation(mock_index):
    mock_index.datasets.search_eager.return_value = [FakeDataset()]
    mock_index.datasets.search.return_value = [FakeDataset()]
    input_region = {
        'crs': 'EPSG:4326',
        'latitude': [-41, -42],
        'longitude': [137, 138.5]
    }
    non_gridded_task_generator = NonGriddedTaskGenerator(input_region, filter_product=None, geopolygon=None,
                                                         feature=None, storage=EXAMPLE_STORAGE)

    tasks = non_gridded_task_generator(mock_index, EXAMPLE_SOURCES_SPEC, EXAMPLE_DATE_RANGE)

    tasks = list(tasks)
    assert len(tasks) == 1

    tile = task.sources[0].data
    assert tile.dims == ('time', 'latitude', 'longitude')
    assert tile.geobox.width == 15
    assert tile.geobox.height == 10
    assert len(tile.sources.time) == 0


def xtest_arbitrary_tile_maker_with_sources():
    mock_index = MagicMock()

    # Check is performed validating the name of query fields
    mock_index.datasets.get_field_names.return_value = ['time']
    # TODO: Expects actual datasets with .crs and .extents
    mock_index.datasets.search_eager.return_value = [set('source1'), set('source2')]
    input_region = {
        'crs': 'EPSG:4326',
        'latitude': [-30, -31],
        'longitude': [137, 138.5]
    }
    atm = ArbitraryTileMaker(mock_index, input_region,
                             storage={'crs': 'EPSG:4326', 'resolution': {'latitude': 0.1, 'longitude': 0.1}})

    tile = atm('product', ('2000-01-01', '2001-01-01'), group_by='time')
    assert tile.dims == ('time', 'latitude', 'longitude')
    assert tile.geobox.width == 15
    assert tile.geobox.height == 10
    assert len(tile.sources.time) == 2

TEST_GEOJSON = '''{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[145,-34.4],[146,-34.4],[146,-34.1],[145,-34.1],[145,-34.4]]]}}]}'''
def test_select_gridded_task_generator_from_extenal_geojson(tmpdir):
    shapefile = tmpdir.join('myfile.geojson')
    shapefile.write(TEST_GEOJSON)
    input_region = {'from_file': str(shapefile)}
    task_generator = select_task_generator(input_region, EXAMPLE_STORAGE, None)

    assert isinstance(task_generator, GriddedTaskGenerator)


def test_non_gridded_task_generator_when_specifying_spatial_extents():
    input_region = {'crs': 'EPSG:4326', 'latitude': [-33, -34], 'longitude': [147.1, 147.9]}
    task_generator = select_task_generator(input_region, None, None)

    assert isinstance(task_generator, NonGriddedTaskGenerator)


def test_gridded_task_generation_when_no_input_region():
    input_region = None
    task_generator = select_task_generator(input_region, EXAMPLE_STORAGE, None)

    assert isinstance(task_generator, GriddedTaskGenerator)


def test_gridded_task_generation_when_empty_input_region():
    input_region = {}
    task_generator = select_task_generator(input_region, EXAMPLE_STORAGE, None)

    assert isinstance(task_generator, GriddedTaskGenerator)
