from datetime import datetime

import numpy as np
import pytest
from hypothesis import given, strategies as st

from datacube.utils.dates import date_sequence
from datacube_stats.utils import filter_time_by_source, datetime64_to_inttime


def parse(*dates):
    return tuple(datetime.strptime(d, "%Y-%m-%d") for d in dates)


@pytest.mark.parametrize("source_interval,epoch_interval,expected_interval", [
    (('2017-02-02', '2017-03-03'), parse('2017-01-01', '2018-01-01'), parse('2017-02-02', '2017-03-03')),
    (('2015-01-01', '2015-02-02'), parse('2017-01-01', '2018-01-01'), None),
    (('2010-01-01', '2017-03-03'), parse('2017-01-01', '2018-01-01'), parse('2017-01-01', '2017-03-03')),
], ids=['interval inside', 'non overlapping', 'overlapping'])
def test_overlapping_dates(epoch_interval, source_interval, expected_interval):
    result = filter_time_by_source(source_interval, epoch_interval)

    assert result == expected_interval


@given(st.lists(st.datetimes(datetime(1970, 1, 1), datetime(2100, 1, 1))).map(lambda dates: np.array(dates)))
def test_inttime(dates):
    inttimes = datetime64_to_inttime(dates)

    assert all(f'{date:%Y%m%d}' == str(inttime) for date, inttime in zip(dates, inttimes))


reasonable_date = st.datetimes(datetime(1990, 1, 1), datetime(2100, 1, 1))
start_end = st.lists(reasonable_date, min_size=2, max_size=2).map(sorted)


@st.composite
def durations(draw):
    num = draw(st.integers(1, 100))
    suffix = draw(st.sampled_from('ymd'))
    return f'{num}{suffix}'


@given(start_end, durations(), durations())
def test_date_sequence(start_end, stats_duration, step_size):
    start, end = start_end
    sequence = date_sequence(start, end, stats_duration, step_size)

    assert iter(sequence) == sequence

    all_ranges = list(sequence)

    assert all(s < e for s, e in all_ranges)
