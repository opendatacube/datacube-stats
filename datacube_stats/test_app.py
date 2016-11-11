"""
Test that the stats app can generate tasks and run them.
"""

from .main import StatsApp


def test_stats_app():
    stats_app = StatsApp()
    assert stats_app is not None


def test_generate_gridded_tasks():
    assert False


def test_generate_non_gridded_tasks():
    assert False
