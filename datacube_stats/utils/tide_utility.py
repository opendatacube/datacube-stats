import statistics
import sys
import logging
from datetime import timedelta, datetime
from datacube.api.query import query_group_by

from otps.predict_wrapper import predict_tide
from otps import TimePoint
from operator import itemgetter
import pandas as pd

DERIVED_PRODS = ['dry', 'wet', 'item', 'low', 'high']
FILTER_METHOD = {
    'by_tidal_height': ['item', 'low', 'high'],
    'by_hydrological_months': ['dry', 'wet'],
}
PROD_SUB_LIST = ['e', 'f', 'ph', 'pl']
HYDRO_START_CAL = '01/07/'
HYDRO_END_CAL = '30/11/'

_LOG = logging.getLogger('tide_utility')


# Return the geom for a feature id
def geom_from_file(filename, feature_id):
    import fiona
    from datacube.utils.geometry import CRS, Geometry

    with fiona.open(filename) as input_region:
        for feature in input_region:
            ID = feature['properties']['ID']
            if ID in feature_id:
                geom = feature['geometry']
                crs = CRS(input_region.crs_wkt)
                geom = Geometry(geom, crs)
    return geom


# Return all sensor dates related to the feature id
def list_poly_dates(dc, boundary_polygon, sources_spec, date_ranges):

    all_times = list()
    for source_spec in sources_spec:
        for mask in source_spec.get('masks', []):
            group_by_name = source_spec.get('group_by', 'solar_day')
            gl_range = (date_ranges[0][0], date_ranges[0][1])
            if source_spec.get('time'):
                # No need to do if start date is more than the allowed date
                if gl_range[0] > datetime.strptime(source_spec['time'][1], "%Y-%m-%d"):
                    continue
                # reset the end date in case it is more than filtered allowed date
                if gl_range[1] > datetime.strptime(source_spec['time'][1], "%Y-%m-%d"):
                    gl_range = (date_ranges[0][0], datetime.strptime(source_spec['time'][1], "%Y-%m-%d"))
            ds = dc.find_datasets(product=mask['product'], time=gl_range,
                                  geopolygon=boundary_polygon, group_by=group_by_name)
            group_by = query_group_by(group_by=group_by_name)
            sources = dc.group_datasets(ds, group_by)
            # Here is a data error specific to this date so before adding exclude it
            if len(ds) > 0:
                all_times = all_times + [dd for dd in sources.time.data.astype('M8[s]').astype('O').tolist()]
    return sorted(all_times)


# This routine is used for ITEM product and it returns a list of dates corresponding to the range interval.
def range_tidal_data(all_dates, ID, filter_product, ln, la):
    perc = filter_product['args']['tide_range']
    per = filter_product['args']['tide_percent']
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
    max_height = tide_list[-1][1]
    min_height = tide_list[0][1]
    dr = max_height-min_height
    inc = min_height
    # This is hard coded to have 9 intervals for tide_range 10 to support ITEM V1
    # if tide_range changes then it needs reworking here to merge top or bottom intervals
    per_range = int(100/perc) - 1
    _LOG.info("id, per, min, max, observation, LOT, HOT, median")
    # Extract list of dates that falls within the input range otherwise return empty list
    for i in range(per_range):
        incmn = inc
        perc = 20 if i == per_range-1 else 10
        inc = float("%.3f" % (incmn + dr*perc*0.01))
        inc = max_height if i == per_range-1 else inc
        range_value = [[x[0].strftime('%Y-%m-%d'), x[1]] for x in tide_list
                       if (x[1] >= incmn and x[1] <= inc)]
        median = float("%.3f" % (statistics.median([x[1] for x in range_value])))
        if per == (i+1)*10:
            _LOG.info("MEDIAN INFO " + str(ID) + "," + str(per) + "," + str(incmn) + "," +
                      str(inc) + "," + str(len(range_value)) + "," +
                      str(range_value[0][1]) + "," + str(range_value[-1][1]) + "," + str(median))
            return range_value
    return []


# This function is used for composite products and also for sub class extraction
# like ebb/flow/peak high/low on the basis of 15 minutes before and after
def extract_otps_computed_data(dates, date_ranges, per, ln, la):
    tp = list()
    tide_dict = dict()
    new_date_list = list()
    mnt = timedelta(minutes=15)
    # add 15 min before and after to decide the type of tide for each dates
    for dt in dates:
        new_date_list.append(dt-mnt)
        new_date_list.append(dt)
        new_date_list.append(dt+mnt)
    for dt in new_date_list:
        tp.append(TimePoint(ln, la, dt))
    tides = predict_tide(tp)
    if len(tides) == 0:
        raise ValueError("No tide height observed from OTPS model within lat/lon range")
    # collect in ebb/flow list
    for tt in tides:
        tide_dict[datetime.strptime(tt.timepoint.timestamp.isoformat()[0:19], "%Y-%m-%dT%H:%M:%S")] = tt.tide_m
    tmp_lt = sorted(tide_dict.items(), key=lambda x: x[0])
    dt_list = tmp_lt[1::3]
    my_data = sorted(dt_list, key=itemgetter(1))
    # Following logic is based on tide observed 15 min before and 15 min after = ebb tide
    # If 15min before < 15min after = flow tide
    # If 15min before & 15min after < Observed tide = Peak high
    # If 15min before & 15min after > Observed tide = Peak low
    # This is tested and worked
    tmp_lt = [[tmp_lt[i+1][0].strftime("%Y-%m-%d"), 'ph']
              if tmp_lt[i][1] < tmp_lt[i+1][1] and tmp_lt[i+2][1] < tmp_lt[i+1][1] else
              [tmp_lt[i+1][0].strftime("%Y-%m-%d"), 'pl'] if tmp_lt[i][1] > tmp_lt[i+1][1] and
              tmp_lt[i+2][1] > tmp_lt[i+1][1] else [tmp_lt[i+1][0].strftime("%Y-%m-%d"), 'f']
              if tmp_lt[i][1] < tmp_lt[i+2][1] else [tmp_lt[i+1][0].strftime("%Y-%m-%d"), 'e']
              for i in range(0, len(tmp_lt), 3)]
    perc = 25 if per == 50 else per
    max_height = my_data[-1][1]
    min_height = my_data[0][1]
    dr = max_height - min_height
    lmr = min_height + dr*perc*0.01   # low tide max range
    hlr = max_height - dr*perc*0.01   # high tide range
    list_low = list()
    # doing for middle percentage to extract list of dates or for any percentage as per input.
    if perc == 50:
        list_high = sorted([[x[0].strftime('%Y-%m-%d'), x[1]]
                            for x in my_data if (x[1] >= lmr) & (x[1] <= hlr) &
                            (x[0] >= date_ranges[0][0]) & (x[0] <= date_ranges[0][1])])
    else:
        list_low = sorted([[x[0].strftime('%Y-%m-%d'), x[1]] for x in my_data if (x[1] <= lmr) &
                          (x[0] >= date_ranges[0][0]) & (x[0] <= date_ranges[0][1])])
        list_high = sorted([[x[0].strftime('%Y-%m-%d'), x[1]] for x in my_data if (x[1] >= hlr) &
                           (x[0] >= date_ranges[0][0]) & (x[0] <= date_ranges[0][1])])
    # Extract list of dates and type of tide phase within the date ranges for composite products
    ebb_flow = [tt for tt in tmp_lt
                if ((datetime.strptime(tt[0], "%Y-%m-%d") >= date_ranges[0][0]) &
                    (datetime.strptime(tt[0], "%Y-%m-%d") <= date_ranges[0][1]))]
    return dt_list, list_low, list_high, ebb_flow


# This function is used to return a list of hydrological date range for dry wet geomedian
# as per month list passed from config or by default from July to Nov.
def get_hydrological_months(filter_product):
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


# return a list of low high dates as per tide phase request
def filter_sub_class(filter_product, list_low, list_high, ebb_flow):
    sub_class = filter_product['args']['sub_class']
    if filter_product['args']['sub_class'] == sub_class:
        if list_low is not None:
            key = set(e[0] for e in ebb_flow if e[1] == sub_class)
            list_low = [ff for ff in list_low if ff[0] in key]
        key = set(e[0] for e in ebb_flow if e[1] == sub_class)
        list_high = [ff for ff in list_high if ff[0] in key]
    return list_low, list_high, ebb_flow


def get_ebb_flow(filter_product, list_low, list_high, ebb_flow):
    list_low, list_high, ebb_flow = filter_sub_class(filter_product, list_low, list_high, ebb_flow)
    _LOG.info("SUB class dates extracted %s for list low %s  and for list high %s",
              filter_product['sub_class'], list_low, list_high)
    filter_time = list_low if filter_product['name'] == 'low' else list_high
    key = set(e[0] for e in filter_time)
    ebb_flow_class = [ff for ff in eval(ebb_flow) if ff[0] in key]
    # dynamically capture ebb flow information for metadata purpose
    filter_product['args']['ebb_flow'] = {'ebb_flow': ebb_flow_class}
    _LOG.info('\nCreated EBB FLOW for feature length %d,  \t%s',
              len(ebb_flow_class), str(ebb_flow_class))
    return filter_time
