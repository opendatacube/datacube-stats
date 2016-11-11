from datacube_stats.output_drivers import RioOutputDriver


def test_rio():

    with RioOutputDriver(task, storage, output_path) as output_driver:

        output_driver.write_data()
