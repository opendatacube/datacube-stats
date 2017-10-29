import statistics
import sys
import logging
import pandas as pd
import pytest

from datetime import timedelta, datetime
from datacube.api.query import query_group_by
from otps.predict_wrapper import predict_tide
from otps import TimePoint
from operator import itemgetter
from datacube_stats.utils.dates import filter_time_by_source


DERIVED_PRODS = ['dry', 'wet', 'item', 'low', 'high']
FILTER_METHOD = {
    'by_tidal_height': ['item', 'low', 'high'],
    'by_hydrological_months': ['dry', 'wet'],
}
PROD_SUB_LIST = ['e', 'f', 'ph', 'pl']
HYDRO_START_CAL = '01/07/'
HYDRO_END_CAL = '30/11/'

_LOG = logging.getLogger('tide_utility')

MODULE_EXISTS = 'otps' in sys.modules


def geom_from_file(filename, feature_id):
    """ Return the geom for a feature id
    :param filename: passed from input_region from_file parameter
    :param feature_id: It is passed from input_region
    :return:  boundary polygon or none
    """
    import fiona
    from datacube.utils.geometry import CRS, Geometry

    with fiona.open(filename) as input_region:
        for feature in input_region:
            if feature['properties']['ID'] in feature_id:
                geom = feature['geometry']
                crs = CRS(input_region.crs_wkt)
                geom = Geometry(geom, crs)
                return geom
    return None


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


@pytest.mark.xfail(not MODULE_EXISTS, reason="otps module is not available")
def range_tidal_data(all_dates, feature_id, tide_range, per, ln, la):
    """
    This routine is used for ITEM product and it returns a list of dates corresponding to the range interval.
    :param all_dates:  gets all the source dates
    :param feature_id: It is used to have a log information
    :param tide_range: It supports 10 percentage. Can be changed through config file
    :param per: tide percentage to use
    :param ln: model centroid longitude value from polygon feature
    :param la: model centroid lattitude value from polygon feature
    :return:  a list of filtered time
    """
    tp = list()
    tide_dict = dict()
    for dt in all_dates:
        tp.append(TimePoint(ln, la, dt))
    # Calling this routine to get the tide object for each timepoint
    tides = predict_tide(tp)
    if len(tides) == 0:
        raise ValueError("No tide height observed from OTPS model within lat/lon range")
    for tt in tides:
        tide_dict[datetime.strptime(tt.timepoint.timestamp.isoformat()[0:19], "%Y-%m-%dT%H:%M:%S")] = tt.tide_m
    # sorting as per tide heights lowest to highest
    tide_list = sorted(tide_dict.items(), key=lambda x: x[1])
    # This is hard coded to have 9 intervals for tide_range 10 to support ITEM V1
    # if tide_range changes then it needs reworking here to merge top or bottom intervals
    per_range = int(100/tide_range) - 1
    # Extract list of dates that falls within the input range otherwise return empty list
    return input_range_data(per_range, tide_list, feature_id, per)


def input_range_data(per_range, tide_list, feature_id, per):
    """ Returns inter tidal range median values
    :param per_range: range of the inter tidal
    :param tide_list: sorted tide list as per height
    :param feature_id: Used for important debug info
    :param per: tide percentage
    :return:
    """
    inc = tide_list[0][1]
    _LOG.info("id, per, min, max, observation, LOT, HOT, median")
    for i in range(per_range):
        inc_cnt = inc
        perc = 20 if i == per_range-1 else 10
        inc = float("%.3f" % (inc_cnt + (tide_list[-1][1] - tide_list[0][1])*perc*0.01))
        inc = tide_list[-1][1] if i == per_range-1 else inc
        range_value = [[x[0].strftime('%Y-%m-%d'), x[1]] for x in tide_list
                       if x[1] >= inc_cnt and x[1] <= inc]
        median = float("%.3f" % (statistics.median([x[1] for x in range_value])))
        if per == (i+1)*10:
            _LOG.info("MEDIAN INFO " + str(feature_id) + "," + str(per) + "," + str(inc_cnt) + "," +
                      str(inc) + "," + str(len(range_value)) + "," +
                      str(range_value[0][1]) + "," + str(range_value[-1][1]) + "," + str(median))
            return range_value
    return []


@pytest.mark.fail(not MODULE_EXISTS, reason="otps module is not packaged")
def extract_otps_computed_data(dates, date_ranges, per, ln, la):
    """
    This function is used for composite products and also for sub class extraction
    like ebb/flow/peak high/low on the basis of 15 minutes before and after
    :param dates: a list of source dates for valid pq datasets
    :param date_ranges: The date range passed
    :param per:
    :param ln:
    :param la:
    :return:
    """

    tides = list()
    tide_dict = dict()
    new_date_list = list()
    # add 15 min before and after to decide the type of tide for each dates
    for dt in dates:
        new_date_list.append(dt-timedelta(minutes=15))
        new_date_list.append(dt)
        new_date_list.append(dt+timedelta(minutes=15))
    for dt in new_date_list:
        tides.append(TimePoint(ln, la, dt))
    tides = predict_tide(tides)
    if len(tides) == 0:
        raise ValueError("No tide height observed from OTPS model within lat/lon range")
    # collect in ebb/flow list
    for tt in tides:
        tide_dict[datetime.strptime(tt.timepoint.timestamp.isoformat()[0:19], "%Y-%m-%dT%H:%M:%S")] = tt.tide_m
    # Here sort data now as predict_tide can return out of order tide list and
    # send now a list of sorted tidal height, sorted tidal date data, percentage and input date range
    return list_time_otps_data(sorted(sorted(tide_dict.items(), key=lambda x: x[0])[1::3],
                                      key=itemgetter(1)),
                               sorted(tide_dict.items(), key=lambda x: x[0]),
                               per, date_ranges)


def list_time_otps_data(tide_data, tide_dt, per, date_ranges):
    """
    Following logic is based on tide observed 15 min before and 15 min after = ebb tide
    If 15min before < 15min after = flow tide
    If 15min before & 15min after < Observed tide = Peak high
    If 15min before & 15min after > Observed tide = Peak low
    This is tested and worked for ebb flow
    It returns a list of low/high and ebb flow data
    :param tide_data: list of tuples with date and sorted tide height like
     [(datetime.datetime(2013, 8, 17, 0, 41, 44), 0.3)]
    :param tide_dt: list of tuples with sorted date and height
    :param per:
    :param date_ranges:
    :return:
    """
    max_tide_ht = tide_data[-1][1]
    low_tide_ht = tide_data[0][1]
    lowest_tide_dt = date_ranges[0][0]
    highest_tide_dt = date_ranges[0][1]
    # Creates a list of lists of date and ebb flow sub class like [['2013-09-18', 'f'], ['2013-09-25', 'e']]
    tide_dt = [[tide_dt[i+1][0].strftime("%Y-%m-%d"), 'ph']
               if tide_dt[i][1] < tide_dt[i+1][1] and tide_dt[i+2][1] < tide_dt[i+1][1] else
               [tide_dt[i+1][0].strftime("%Y-%m-%d"), 'pl'] if tide_dt[i][1] > tide_dt[i+1][1] and
               tide_dt[i+2][1] > tide_dt[i+1][1] else [tide_dt[i+1][0].strftime("%Y-%m-%d"), 'f']
               if tide_dt[i][1] < tide_dt[i+2][1] else [tide_dt[i+1][0].strftime("%Y-%m-%d"), 'e']
               for i in range(0, len(tide_dt), 3)]
    perc_adj = 25 if per == 50 else per
    # find the low and high tide range from low_tide_ht and max_tide_ht
    lmr = low_tide_ht + (max_tide_ht - low_tide_ht)*perc_adj*0.01   # low tide max range
    hlr = max_tide_ht - (max_tide_ht - low_tide_ht)*perc_adj*0.01   # high tide range
    list_low = list()
    # doing for middle percentage to extract list of dates or for any percentage as per input.
    if perc_adj == 50:
        list_high = sorted([[x[0].strftime('%Y-%m-%d'), x[1]]
                            for x in tide_data if (x[1] >= lmr) & (x[1] <= hlr) &
                            (x[0] >= lowest_tide_dt) & (x[0] <= highest_tide_dt)])
    else:
        list_low = sorted([[x[0].strftime('%Y-%m-%d'), x[1]] for x in tide_data if (x[1] <= lmr) &
                           (x[0] >= lowest_tide_dt) & (x[0] <= highest_tide_dt)])
        list_high = sorted([[x[0].strftime('%Y-%m-%d'), x[1]] for x in tide_data if (x[1] >= hlr) &
                            (x[0] >= lowest_tide_dt) & (x[0] <= highest_tide_dt)])
    # Extract list of dates and type of tide phase within the date ranges for composite products
    ebb_flow = [tt for tt in tide_dt
                if ((datetime.strptime(tt[0], "%Y-%m-%d") >= lowest_tide_dt) &
                    (datetime.strptime(tt[0], "%Y-%m-%d") <= highest_tide_dt))]
    return list_low, list_high, ebb_flow


def get_hydrological_months(filter_product):
    """ This function is used to return a list of hydrological date range for dry wet geomedian
        as per month list passed from config or by default from July to Nov
        :param filter_product: input year from polygon and months from config
        :return: a list of dates corresponding to predefined month range or from config
    """
    all_dates = list()
    for k, v in filter_product['year'].items():
        year = int(v)
        months = filter_product['args'].get('months')
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


def filter_sub_class(sub_class, list_low, list_high, ebb_flow):
    """ return a list of low high dates as per tide phase request
    :param sub_class: e/f/ph/pl as defined in config
    :param list_low: a list of low tide dates
    :param list_high: a list of high flow dates
    :param ebb_flow: a list of low or high ebb flow details
    :return:
    """
    if list_low is not None:
        key = set(e[0] for e in ebb_flow if e[1] == sub_class)
        list_low = [ff for ff in list_low if ff[0] in key]
    key = set(e[0] for e in ebb_flow if e[1] == sub_class)
    list_high = [ff for ff in list_high if ff[0] in key]
    return list_low, list_high


def get_ebb_flow(filter_product, list_low, list_high, ebb_flow):
    """

    :param filter_product: capture the ebb flow list
    :param list_low: collect low tide list of ebb flow
    :param list_high: collect high tide list of ebb flow
    :param ebb_flow:
    :return:
    """
    list_low, list_high = filter_sub_class(filter_product['args']['sub_class'],
                                           list_low, list_high, ebb_flow)
    _LOG.info("SUB class dates extracted %s for list low %s  and for list high %s",
              filter_product['args']['sub_class'], list_low, list_high)
    filter_time = list_low if filter_product['args']['type'] == 'low' else list_high
    key = set(e[0] for e in filter_time)
    ebb_flow_details = [ff for ff in ebb_flow if ff[0] in key]
    # dynamically capture ebb flow information for metadata purpose
    filter_product['args']['ebb_flow'] = {'ebb_flow': ebb_flow_details}
    _LOG.info('\nCreated EBB FLOW for feature length %d,  \t%s',
              len(ebb_flow_details), str(ebb_flow_details))
    return filter_time


def get_filter_product(filter_product, feature, all_dates, date_ranges):
    """
    Finding the sub product on the basis of methodology and returns a list of filter time and
    dynamically built poly index tuple to be used later in naming output file
    :param filter_product: Input filter_product object. Like tide_range/tide_percent/type/sub_class
    :param feature: Get all geometry info like lon/lat/ID
    :param all_dates: all source dates
    :param date_ranges: global date range
    :return: poly_index and filter time
    """

    if filter_product.get('method') == 'by_hydrological_months':
        # get the year id and geometry for dry or wet type
        filter_product['year'] = {k: v for k, v in feature.items() if "DY" in k.upper()} \
            if filter_product.get('type') == 'dry' else \
            {k: v for k, v in feature.items() if "WY" in k.upper()}
        poly_y = "_".join(x for x in [v for k, v in filter_product['year'].items()])
        poly_index = (str(feature['ID']), poly_y)
        filter_time = get_hydrological_months(filter_product)
    elif filter_product.get('method') == 'by_tide_height':
        poly_x = str(feature['ID']) + '_' + str(feature['lon'])
        poly_y = str(feature['lat']) + '_PER_' + str(filter_product['args']['tide_percent'])
        poly_index = (poly_x, poly_y)
        # get all relevant date time lists
        if filter_product['args'].get('tide_range'):
            # It is ITEM product
            filter_time = range_tidal_data(all_dates, feature['ID'], filter_product['args']['tide_range'],
                                           filter_product['args']['tide_percent'], feature['lon'],
                                           feature['lat'])
        else:
            # This is for low/high composite
            list_low, list_high, ebb_flow = \
                extract_otps_computed_data(all_dates, date_ranges,
                                           filter_product['args']['tide_percent'], feature['lon'],
                                           feature['lat'])
            filter_time = list_low if filter_product['args']['type'] == 'low' else list_high
            # filter out dates as per sub classification of ebb flow
            if filter_product['args'].get('sub_class'):
                filter_time = get_ebb_flow(filter_product, list_low, list_high, ebb_flow)
                poly_x = filter_product['args']['sub_class'] + "_" + str(feature['lon'])
                poly_index = (poly_x, feature['lat'])
            _LOG.info('\n DATE LIST for feature %d length %d, time period: %s \t%s',
                      feature['ID'], len(filter_time), date_ranges, str(filter_time))
    else:
        poly_index = ()
        filter_time = []
    return poly_index, filter_time
