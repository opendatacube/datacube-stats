import logging
import statistics
from datetime import timedelta, datetime
from operator import itemgetter

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
    Return the geom for a feature id
    :param filename: passed from input_region from_file parameter
    :param feature_id: Currently supports a single element feature id from a list
    :return:  boundary polygon or none
    """
    import fiona
    from datacube.utils.geometry import CRS, Geometry

    with fiona.open(filename) as input_region:
        for feature in input_region:
            if feature['properties']['ID'] in feature_id:
                geom = feature['geometry']
                crs = CRS(input_region.crs_wkt)
                return feature['properties'], geom, input_region.crs_wkt, Geometry(geom, crs)
    _LOG.info("No geometry found")
    try:
        raise ValueError
    except ValueError as exp:
        _LOG.info("No geometry found ")


class FilterDerivedProducts(object):
    """
    Setting Feature id and method
    Filtering all derived products depending on methods
    """

    def __init__(self, feature_id, method):
        self.feature_id = feature_id
        self.method = method


class Item(object):
    """Expecting item attributes"""

    def __init__(self, tide_percent, lon, lat, tide_range):
        # tide percentage
        self.tide_percent = tide_percent
        self.lon = lon
        self.lat = lat
        # tide range can be used from 1 to 10
        self.tide_range = tide_range


class Hltc(object):
    """HLTC product """

    def __init__(self, tide_percent, lon, lat, prod_type, sub_class):
        self.tide_percent = tide_percent
        self.lon = lon
        self.lat = lat
        # type of product low/high
        self.type = prod_type
        # sub_class is optional
        self.sub_class = sub_class


class DryWet(object):
    """Ground water dry wet products"""

    def __init__(self, prod_type, months):
        # type of product dry/wet
        self.type = prod_type
        # optional a list of months
        self.months = months

    def get_years(self, feature):
        # return a list of years
        sub_type = 'DY' if self.type == 'dry' else 'WY'
        return {k: v for k, v in feature.items() if sub_type in k.upper()}


def load_tide_model(all_dates, ln, la):
    """
    Load otps module and pass a list of tide information

    :param all_dates: Input a list of dates
    :param ln: model longitude
    :param la: model latitude
    :return: a list of tides
    """
    try:
        from otps.predict_wrapper import predict_tide
        from otps import TimePoint
    except ImportError:
        raise StatsConfigurationError("otps module not found. Please load otps module separately ...")

    tp = list()
    for dt in all_dates:
        tp.append(TimePoint(ln, la, dt))
    # Calling this routine to get the tide object for each timepoint
    tides = predict_tide(tp)
    return tides


def range_tidal_data(all_dates, feature_id, tide_range, per, ln, la):
    """
    This routine is used for ITEM product and it returns a list of dates corresponding to the range interval.

    :param all_dates:  gets all the source dates
    :param feature_id: It is used to have a log information
    :param tide_range: It supports 10 percentage. Can be changed through config file
    :param per: tide percentage to use
    :param ln: model centroid longitude value from polygon feature
    :param la: model centroid latitude value from polygon feature
    :return:  a list of filtered time
    """

    tides = load_tide_model(all_dates, ln, la)
    if len(tides) == 0:
        raise ValueError("No tide height observed from OTPS model within lat/lon range")
    tide_dict = dict()
    for tt in tides:
        tide_dict[datetime.strptime(tt.timepoint.timestamp.isoformat()[0:19], "%Y-%m-%dT%H:%M:%S")] = tt.tide_m
    # sorting as per tide heights lowest to highest
    tide_list = sorted(tide_dict.items(), key=lambda x: x[1])
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
            _LOG.info("MEDIAN INFO " + str(feature_id) + "," + str(per) + "," + str(inc_cnt) + "," +
                      str(inc) + "," + str(len(range_value)) + "," +
                      str(range_value[0][1]) + "," + str(range_value[-1][1]) + "," + str(median))
            # return the date part only
            range_value = [rv[0] for rv in range_value]
            return range_value
    return []


def extract_otps_computed_data(dates, date_ranges, per, ln, la):
    """
    This function is used for composite products and also for sub class extraction
    like ebb/flow/peak high/low on the basis of 15 minutes before and after

    :param dates: a list of source dates for valid pq datasets
    :param date_ranges: The date range passed
    :param per: tide percentage
    :param ln: longitude
    :param la: latitude
    :return:
    """
    tide_dict = dict()
    new_date_list = list()
    # add 15 min before and after to decide the type of tide for each dates
    for dt in dates:
        new_date_list.append(dt - timedelta(minutes=15))
        new_date_list.append(dt)
        new_date_list.append(dt + timedelta(minutes=15))
    tides = load_tide_model(new_date_list, ln, la)
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
    list_low = list()
    lowest_tide_dt = date_ranges[0][0]
    highest_tide_dt = date_ranges[0][1]

    # doing for middle percentage to extract list of dates or for any percentage as per input.
    if perc_adj == 50:
        list_high = sorted([[x[0].strftime('%Y-%m-%dT%H:%M:%S'), x[1]]
                            for x in tide_data if (x[1] >= lmr) and (x[1] <= hlr) and
                            (x[0] >= lowest_tide_dt) and (x[0] <= highest_tide_dt)])
    else:
        list_low = sorted([[x[0].strftime('%Y-%m-%dT%H:%M:%S'), x[1]] for x in tide_data if (x[1] <= lmr) and
                           (x[0] >= lowest_tide_dt) and (x[0] <= highest_tide_dt)])
        list_high = sorted([[x[0].strftime('%Y-%m-%dT%H:%M:%S'), x[1]] for x in tide_data if (x[1] >= hlr) and
                            (x[0] >= lowest_tide_dt) and (x[0] <= highest_tide_dt)])
    # Extract list of dates and type of tide phase within the date ranges for composite products
    ebb_flow = [tt for tt in ebb_flow_data
                if ((datetime.strptime(tt[0], "%Y-%m-%dT%H:%M:%S") >= lowest_tide_dt) and
                    (datetime.strptime(tt[0], "%Y-%m-%dT%H:%M:%S") <= highest_tide_dt))]
    return list_low, list_high, ebb_flow


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
    :param ebb_flow: ebb flow details
    :return:
    """
    list_low, list_high = filter_sub_class(filter_product['args']['sub_class'],
                                           list_low, list_high, ebb_flow)
    _LOG.info("SUB class dates extracted %s for list low %s  and for list high %s",
              filter_product['args']['sub_class'], list_low, list_high)
    filtered_times = list_low if filter_product['args']['type'] == 'low' else list_high
    key = set(e[0] for e in filtered_times)
    ebb_flow_details = [ff for ff in ebb_flow if ff[0] in key]
    # dynamically capture ebb flow information for metadata purpose
    filter_product['args']['ebb_flow'] = {'ebb_flow': ebb_flow_details}
    _LOG.info('\nCreated EBB FLOW for feature length %d,  \t%s',
              len(ebb_flow_details), str(ebb_flow_details))
    return filtered_times


def get_poly_fl_name(feature_id, tide_percent=None, lon=None, lat=None, years=None, sub_class=None):
    """
      Returns poly index to be used in Statstask
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
    der_prod = FilterDerivedProducts(feature['ID'], filter_product.get('method'))
    if der_prod.method == 'by_hydrological_months':
        # Initialising DryWet class
        prod = DryWet(filter_product['args']['type'], filter_product['args'].get('months'))
        years = prod.get_years(feature)
        poly_fl_name = get_poly_fl_name(feature_id=der_prod.feature_id, years=years)
        # filtered_times = get_hydrological_years(filter_product['year'], filter_product['args'].get('months'))
        filtered_times = get_hydrological_years(years, prod.months)
    elif der_prod.method == 'by_tide_height':
        # get all relevant date time lists
        if filter_product['args'].get('tide_range'):
            # It is ITEM product, Initialise Item class first
            prod = Item(tide_percent=filter_product['args']['tide_percent'], lon=feature['lon'], lat=feature['lat'],
                        tide_range=filter_product['args']['tide_range'])

            filtered_times = range_tidal_data(all_dates, der_prod.feature_id, prod.tide_range,
                                              prod.tide_percent, prod.lon, prod.lat)
            poly_fl_name = get_poly_fl_name(feature_id=der_prod.feature_id, tide_percent=prod.tide_percent,
                                            lon=prod.lon, lat=prod.lat)
        else:
            # This is for low/high composite. Initialising Hltc class
            prod = Hltc(tide_percent=filter_product['args']['tide_percent'], lon=feature['lon'], lat=feature['lat'],
                        prod_type=filter_product['args']['type'], sub_class=filter_product['args']['sub_class']
                        if filter_product['args'].get('sub_class') else None)
            list_low, list_high, ebb_flow = \
                extract_otps_computed_data(all_dates, date_ranges, prod.tide_percent, prod.lon, prod.lat)
            filtered_times = list_low if prod.type == 'low' else list_high
            # filter out dates as per sub classification of ebb flow
            if prod.sub_class:
                filtered_times = get_ebb_flow(filter_product, list_low, list_high, ebb_flow)
                poly_fl_name = get_poly_fl_name(feature_id=der_prod.feature_id, tide_percent=prod.tide_percent,
                                                lon=prod.lon, lat=prod.lat, sub_class=prod.sub_class)
            else:
                poly_fl_name = get_poly_fl_name(feature_id=der_prod.feature_id, tide_percent=prod.tide_percent,
                                                lon=prod.lon, lat=prod.lat, sub_class=None)

            _LOG.info('\n DATE LIST for feature %d length %d, time period: %s \t%s',
                      der_prod.feature_id, len(filtered_times), date_ranges, str(filtered_times))
            filtered_times = [ft[0] for ft in filtered_times]
    else:
        _LOG.info("No filter values found ")
        raise ValueError

    return poly_fl_name, sorted(filtered_times)
