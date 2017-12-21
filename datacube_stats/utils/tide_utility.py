import logging
import statistics
from datetime import timedelta, datetime
from operator import itemgetter

from datacube.utils.geometry import CRS, Geometry

from datacube_stats.statistics import StatsConfigurationError
from datacube_stats.utils.dates import get_hydrological_years

DERIVED_PRODS = ['dry', 'wet', 'item', 'low', 'high']
FILTER_METHOD = {
    'by_tidal_height': ['item', 'low', 'high'],
    'by_hydrological_months': ['dry', 'wet'],
}
PROD_SUB_LIST = ['e', 'f', 'ph', 'pl']

_LOG = logging.getLogger('tide_utility')


def geom_from_file(filename, feature_id):
    """
    The geometry of a feature
    :param filename: name of shape file
    :param feature_id: the id of the wanted feature
    """
    import fiona

    with fiona.open(filename) as input_region:
        for feature in input_region:
            if feature['properties']['ID'] in feature_id:
                geom = feature['geometry']
                crs = CRS(input_region.crs_wkt)
                return feature['properties'], geom, input_region.crs_wkt, Geometry(geom, crs)

    _LOG.info("No geometry found")


def load_tide_model(all_dates, lon, lat):
    """
    Load otps module and pass a list of tide information

    :param all_dates: Input a list of dates
    :param lon: model longitude
    :param lat: model latitude
    :return: a list of tides
    """
    try:
        from otps.predict_wrapper import predict_tide
        from otps import TimePoint
    except ImportError:
        raise StatsConfigurationError("otps module not found. Please load otps module separately ...")

    return predict_tide([TimePoint(lon, lat, dt) for dt in all_dates])


def format_date(tt):
    return datetime.strptime(tt.timepoint.timestamp.isoformat()[0:19], "%Y-%m-%dT%H:%M:%S")


def range_tidal_data(all_dates, feature_id, tide_range, per, lon, lat):
    """
    This routine is used for ITEM product and it returns a list of dates corresponding to the range interval.

    :param all_dates:  gets all the source dates
    :param feature_id: It is used to have a log information
    :param tide_range: It supports 10 percentage. Can be changed through config file
    :param per: tide percentage to use
    :param lon: model centroid longitude value from polygon feature
    :param lat: model centroid latitude value from polygon feature
    :return:  a list of filtered time
    """

    tides = load_tide_model(all_dates, lon, lat)
    if len(tides) == 0:
        raise ValueError("No tide height observed from OTPS model within lat/lon range")
    tide_dict = {format_date(tt): tt.tide_m for tt in tides}
    # sorting as per tide heights lowest to highest
    tide_list = sorted(tide_dict.items(), key=itemgetter(1))
    # This is hard coded to have 9 intervals for tide_range 10 to support ITEM V1
    # if tide_range changes then it needs reworking here to merge top or bottom intervals
    per_range = int(100 / tide_range) - 1
    # Extract list of dates that falls within the input range otherwise return empty list
    return input_range_data(per_range, tide_list, feature_id, per)


def input_range_data(per_range, tide_list, feature_id, per):
    """
    Returns inter tidal range median values

    :param per_range: range of the inter tidal
    :param tide_list: sorted tide list as per height
    :param feature_id: Used for important debug info
    :param per: tide percentage
    :return:
    """
    inc = tide_list[0][1]
    min_ht = tide_list[0][1]
    max_ht = tide_list[-1][1]
    _LOG.info("id, per, min, max, observation, LOT, HOT, median")
    for i in range(per_range):
        inc_cnt = inc
        perc = 20 if i == per_range - 1 else 10
        inc = float("%.3f" % (inc_cnt + (max_ht - min_ht) * perc * 0.01))
        inc = max_ht if i == per_range - 1 else inc
        range_value = [[x[0].strftime('%Y-%m-%dT%H:%M:%S'), x[1]] for x in tide_list
                       if x[1] >= inc_cnt and x[1] <= inc]
        median = float("%.3f" % (statistics.median([x[1] for x in range_value])))
        if per == (i + 1) * 10:
            _LOG.info("MEDIAN INFO %s", ",".join(str(x) for x in [feature_id, per, inc_cnt, inc, len(range_value),
                                                                  range_value[0][1], range_value[-1][1], median]))
            # return the date part only
            return [rv[0] for rv in range_value]
    return []


def extract_otps_computed_data(dates, date_ranges, per, lon, lat):
    """
    This function is used for composite products and also for sub class extraction
    like ebb/flow/peak high/low on the basis of 15 minutes before and after

    :param dates: a list of source dates for valid pq datasets
    :param date_ranges: The date range passed
    :param per: tide percentage
    :param lon: longitude
    :param lat: latitude
    :return:
    """
    # add 15 min before and after to decide the type of tide for each dates
    new_date_list = [x for dt in dates for x in [dt - timedelta(minutes=15), dt, dt + timedelta(minutes=15)]]
    tides = load_tide_model(new_date_list, lon, lat)
    if len(tides) == 0:
        raise ValueError("No tide height observed from OTPS model within lat/lon range")
    # collect in ebb/flow list
    tide_dict = {format_date(tt): tt.time_m for tt in tides}
    # Here sort data now as predict_tide can return out of order tide list and
    # send now a list of sorted tidal height, sorted tidal date data, percentage and input date range
    return list_time_otps_data(sorted(sorted(tide_dict.items(), key=itemgetter(0))[1::3],
                                      key=itemgetter(1)),
                               sorted(tide_dict.items(), key=itemgetter(0)),
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
    :param per: tide percentage
    :param date_ranges: global date range
    :return:
    """
    max_tide_ht = tide_data[-1][1]
    low_tide_ht = tide_data[0][1]

    # Creates a list of lists of date and ebb flow sub class like [['2013-09-18', 'f'], ['2013-09-25', 'e']]
    ebb_flow_data = list()
    for i in range(0, len(tide_dt), 3):
        first = tide_dt[i][1]
        second = tide_dt[i + 1][1]
        third = tide_dt[i + 2][1]
        tide_date = tide_dt[i + 1][0].strftime("%Y-%m-%dT%H:%M:%S")
        if first < second and third < second:
            ebb_flow_data.append([tide_date, 'ph'])
        elif first > second and third > second:
            ebb_flow_data.append([tide_date, 'pl'])
        elif first < third:
            ebb_flow_data.append([tide_date, 'f'])
        else:
            ebb_flow_data.append([tide_date, 'e'])

    perc_adj = 25 if per == 50 else per
    # find the low and high tide range from low_tide_ht and max_tide_ht
    lmr = low_tide_ht + (max_tide_ht - low_tide_ht) * perc_adj * 0.01  # low tide max range
    hlr = max_tide_ht - (max_tide_ht - low_tide_ht) * perc_adj * 0.01  # high tide range
    return low_high_ebb_flow(lmr, hlr, perc_adj, tide_data, date_ranges, ebb_flow_data)


def low_high_ebb_flow(lmr, hlr, perc_adj, tide_data, date_ranges, ebb_flow_data):
    """ Get all ebb flow and low high dates
    :param lmr: low max range
    :param hlr: high tide range
    :param perc_adj: adjusted percentage
    :param tide_data: tide data
    :param date_ranges: date ranges for epoch
    :param ebb_flow_data: input ebb flow data
    :return: individual list of low, high and ebb flow data
    """
    list_low = []
    lowest_tide_dt, highest_tide_dt = date_ranges[0]

    date_format = '%Y-%m-%dT%H:%M:%S'

    # doing for middle percentage to extract list of dates or for any percentage as per input.
    if perc_adj == 50:
        list_high = sorted([[x[0].strftime(date_format), x[1]]
                            for x in tide_data if (x[1] >= lmr) and (x[1] <= hlr) and
                            (x[0] >= lowest_tide_dt) and (x[0] <= highest_tide_dt)])
    else:
        list_low = sorted([[x[0].strftime(date_format), x[1]] for x in tide_data if (x[1] <= lmr) and
                           (x[0] >= lowest_tide_dt) and (x[0] <= highest_tide_dt)])
        list_high = sorted([[x[0].strftime(date_format), x[1]] for x in tide_data if (x[1] >= hlr) and
                            (x[0] >= lowest_tide_dt) and (x[0] <= highest_tide_dt)])
    # Extract list of dates and type of tide phase within the date ranges for composite products
    ebb_flow = [tt for tt in ebb_flow_data
                if ((datetime.strptime(tt[0], date_format) >= lowest_tide_dt) and
                    (datetime.strptime(tt[0], date_format) <= highest_tide_dt))]
    return list_low, list_high, ebb_flow


def filter_sub_class(sub_class, list_low, list_high, ebb_flow):
    """ return a list of low high dates as per tide phase request
    :param sub_class: e/f/ph/pl as defined in config
    :param list_low: a list of low tide dates
    :param list_high: a list of high flow dates
    :param ebb_flow: a list of low or high ebb flow details
    :return:
    """
    key = set(e[0] for e in ebb_flow if e[1] == sub_class)
    list_low = [f for f in list_low if f[0] in key]
    list_high = [f for f in list_high if f[0] in key]
    return list_low, list_high


def get_ebb_flow(filter_product, list_low, list_high, ebb_flow):
    """
    :param filter_product: capture the ebb flow list
    :param list_low: collect low tide list of ebb flow
    :param list_high: collect high tide list of ebb flow
    :param ebb_flow: ebb flow details
    :return:
    """
    list_low, list_high = filter_sub_class(filter_product['args']['sub_class'],
                                           list_low, list_high, ebb_flow)
    _LOG.info("SUB class dates extracted %s for list low %s  and for list high %s",
              filter_product['args']['sub_class'], list_low, list_high)
    filtered_times = list_low if filter_product['args']['type'] == 'low' else list_high
    key = set(e[0] for e in filtered_times)
    ebb_flow_details = [f for f in ebb_flow if f[0] in key]
    # dynamically capture ebb flow information for metadata purpose
    filter_product['args']['ebb_flow'] = {'ebb_flow': ebb_flow_details}
    _LOG.info('Created EBB FLOW for feature length %d, %s',
              len(ebb_flow_details), str(ebb_flow_details))
    return filtered_times


def get_poly_file_name(feature_id, tide_percent=None, lon=None, lat=None, years=None, sub_class=None):
    """
    Returns poly index to be used in StatsTask
    :param feature_id: Feature id
    :param tide_percent: tide percentage
    :param lon: model longitude for item/hltc
    :param lat: model latitude for item/hltc
    :param years: for dry/wet products
    :param sub_class: for ebb/flow type
    :return: poly file name
    """
    if sub_class:
        return sub_class.upper() + "_" + str(lon), str(lat)
    if years:
        return str(feature_id), "_".join(x for x in [v for k, v in years.items()])
    else:
        return str(feature_id) + '_' + str(lon), str(lat) + '_PER_' + str(tide_percent)


def get_filter_product(filter_product, feature, all_dates, date_ranges):
    """
    Finding the sub product on the basis of methodology and returns a list of filter time and
    dynamically built poly index tuple to be used later in naming output file

    :param filter_product: Input filter_product object. Like tide_range/tide_percent/type/sub_class
    :param feature: Get all geometry info like lon/lat/ID
    :param all_dates: all source dates
    :param date_ranges: global date range
    :return: poly file name and filtered dates/times
    """
    method = filter_product.get('method')
    feature_id = feature['ID']

    def by_hydrological_months(args):
        prod_type = args['type']
        months = args.get('months')

        sub_type = 'DY' if prod_type == 'dry' else 'WY'
        years = {k: v for k, v in feature.items() if sub_type in k.upper()}

        poly_fl_name = get_poly_file_name(feature_id=feature_id, years=years)
        filtered_times = get_hydrological_years(years, months)

        return poly_fl_name, sorted(filtered_times)

    def by_tide_height(args):
        # get all relevant date time lists
        lon, lat = feature['lon'], feature['lat']
        tide_percent = args['tide_percent']

        if 'tide_range' in filter_product['args']:
            # ITEM
            tide_range = args['tide_range']

            filtered_times = range_tidal_data(all_dates, feature_id, tide_range, tide_percent, lon, lat)
            poly_fl_name = get_poly_file_name(feature_id=feature_id, tide_percent=tide_percent, lon=lon, lat=lat)
        else:
            # low/high composite
            prod_type = args['type']
            sub_class = args.get('sub_class')

            list_low, list_high, ebb_flow = \
                extract_otps_computed_data(all_dates, date_ranges, tide_percent, lon, lat)
            filtered_times = list_low if prod_type == 'low' else list_high

            # filter out dates as per sub classification of ebb flow
            if sub_class is not None:
                filtered_times = get_ebb_flow(filter_product, list_low, list_high, ebb_flow)
                poly_fl_name = get_poly_file_name(feature_id=feature_id, tide_percent=tide_percent,
                                                  lon=lon, lat=lat, sub_class=sub_class)
            else:
                poly_fl_name = get_poly_file_name(feature_id=feature_id, tide_percent=tide_percent,
                                                  lon=lon, lat=lat, sub_class=None)

            _LOG.info('DATE LIST for feature %d length %d, time period: %s %s',
                      feature_id, len(filtered_times), date_ranges, str(filtered_times))

            filtered_times = [ft[0] for ft in filtered_times]
        return poly_fl_name, sorted(filtered_times)

    lookup = {'by_hydrological_months': by_hydrological_months,
              'by_tide_height': by_tide_height}

    if method in lookup:
        return lookup[method](filter_product['args'])
    else:
        _LOG.info("No filter values found ")
        raise ValueError
