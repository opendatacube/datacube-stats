"""
Date utility functions to be used by statistics apps


"""
from __future__ import absolute_import

import logging
import pandas as pd
from datetime import datetime
from datacube.api.query import query_group_by

from dateutil.relativedelta import relativedelta
from dateutil.rrule import YEARLY, MONTHLY, DAILY, rrule

_LOG = logging.getLogger(__name__)

FREQS = {'y': YEARLY, 'm': MONTHLY, 'd': DAILY}
DURATIONS = {'y': 'years', 'm': 'months', 'd': 'days'}
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
    for start_date in rrule(freq, interval=step_size, dtstart=start, until=end):
        end_date = start_date + stats_duration
        if end_date <= end:
            yield start_date, start_date + stats_duration


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


def list_poly_dates(dc, boundary_polygon, sources_spec, date_ranges):
    """
     Return all sensor dates related to the feature id, if only pq dataset is available

     :param dc: datacube index
     :param boundary_polygon: for a given geometry
     :param sources_spec:
     :param date_ranges:
     :return: All dates for valid pq datasets within a date range
    """

    all_times = list()
    for source_spec in sources_spec:
        for mask in source_spec.get('masks', []):
            group_by_name = source_spec.get('group_by', 'solar_day')
            ep_range = filter_time_by_source(source_spec.get('time'), date_ranges[0])
            if ep_range is None:
                _LOG.info("Datasets not included for %s and time range for %s", mask['product'], date_ranges[0])
                continue
            ds = dc.find_datasets(product=mask['product'], time=ep_range,
                                  geopolygon=boundary_polygon, group_by=group_by_name)
            group_by = query_group_by(group_by=group_by_name)
            sources = dc.group_datasets(ds, group_by)
            # Here is a data error specific to this date so before adding exclude it
            if len(ds) > 0:
                all_times = all_times + [dd for dd in sources.time.data.astype('M8[s]').astype('O').tolist()]
    return sorted(all_times)


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
        # months = filter_product['args'].get('months')
        # No months
        if months is not None:
            st_dt = str(year+1)+str(months[0])+'01'
            en_dt = str(year+1)+str(months[1])+'30'
        else:
            st_dt = HYDRO_START_CAL + str(year+1)
            en_dt = HYDRO_END_CAL + str(year+1)
        date_list = pd.date_range(st_dt, en_dt)
        date_list = date_list.to_datetime().astype(str).tolist()
        all_dates = all_dates + date_list
    return all_dates
