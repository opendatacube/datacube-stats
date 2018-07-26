"""
Date utility functions to be used by statistics apps


"""
import logging
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta
from dateutil.rrule import YEARLY, MONTHLY, DAILY, rrule

_LOG = logging.getLogger(__name__)

FREQS = {'y': YEARLY, 'm': MONTHLY, 'd': DAILY}
DURATIONS = {'y': 'years', 'm': 'months', 'd': 'days', 'M': 'microseconds'}
HYDRO_START_CAL = '01/07/'
HYDRO_END_CAL = '30/11/'


def date_sequence(start, end, stats_duration, step_size):
    """
    Generate a sequence of time span tuples

    :seealso:
        Refer to `dateutil.parser.parse` for details on date parsing.

    :param str start: Start date of first interval
    :param str end: End date. The end of the last time span may extend past this date.
    :param str stats_duration: What period of time should be grouped
    :param str step_size: How far apart should the start dates be
    :return: sequence of (start_date, end_date) tuples
    """
    step_size, freq = parse_interval(step_size)
    stats_duration = parse_duration(stats_duration)
    #
    # datacube query returns the data on [start_date, end_date], which is a behaviou we don't like to see.
    # We expect data on [start_date, end_dated).
    #
    exclude_duration = parse_duration('1M')
    for start_date in rrule(freq, interval=step_size, dtstart=start, until=end):
        end_date = start_date + stats_duration - exclude_duration
        if end_date <= end:
            yield start_date, end_date


def parse_interval(interval):
    count, units = split_duration(interval)
    try:
        return count, FREQS[units]
    except KeyError:
        raise ValueError('Invalid interval "{}", units not in of: {}'.format(interval, FREQS.keys))


def parse_duration(duration):
    count, units = split_duration(duration)
    try:
        delta = {DURATIONS[units]: count}
    except KeyError:
        raise ValueError('Duration "{}" not in months or years'.format(duration))

    return relativedelta(**delta)


def split_duration(duration):
    return int(duration[:-1]), duration[-1:]


def filter_time_by_source(source_interval, epoch_interval):
    """
    Override date ranges if sensor specific time is within the time_period range

    Parses the source_interval into dates in the form YYYY-MM-DD, then
    returns the intersection of the two ranges, or None if they don't overlap.
    """
    if not source_interval:
        return epoch_interval

    epoch_start, epoch_end = epoch_interval
    source_start, source_end = [datetime.strptime(v, "%Y-%m-%d") for v in source_interval]

    if source_start > epoch_end or epoch_start > source_end:
        _LOG.debug("No valid time overlap for %s and %s", source_interval, epoch_interval)
        return None

    start_time = max(source_start, epoch_start)
    end_time = min(source_end, epoch_end)

    return start_time, end_time


def get_hydrological_years(all_years, months=None):
    """ This function is used to return a list of hydrological date range for dry wet geomedian
        as per month list passed from config or by default from July to Nov
        :param all_years: a list of input years from polygon
        :param months: a list of hydrological months from config or default values
        :return: a list of dates corresponding to predefined month range or from config
    """
    all_dates = list()
    for k, v in all_years.items():
        year = int(v)

        # No months
        if months is not None:
            st_dt = str(year + 1) + str(months[0]) + '01'
            en_dt = str(year + 1) + str(months[1]) + '30'
        else:
            st_dt = HYDRO_START_CAL + str(year + 1)
            en_dt = HYDRO_END_CAL + str(year + 1)
        date_list = pd.date_range(st_dt, en_dt)
        date_list = date_list.to_datetime().astype(str).tolist()
        all_dates = all_dates + date_list
    return all_dates


def datetime64_to_inttime(var):
    """
    Return an "inttime" representing a datetime64.
    For example, 2016-09-29 as an "inttime" would be 20160929

    An 'inttime' is used in statistics which return an actual
    observation to represent the date that observation happened.
    It is a relatively compact representation of a date, while still being human readable.

    :param var: ndarray of datetime64
    :return: ndarray of ints, representing the given time to the nearest day
    """
    values = getattr(var, 'values', var)
    years = values.astype('datetime64[Y]').astype('int32') + 1970
    months = values.astype('datetime64[M]').astype('int32') % 12 + 1
    days = (values.astype('datetime64[D]') - values.astype('datetime64[M]') + 1).astype('int32')
    return years * 10000 + months * 100 + days
