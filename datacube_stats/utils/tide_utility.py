import statistics
from math import ceil
from datetime import timedelta, datetime
from otps.predict_wrapper import predict_tide
from otps import TimePoint
from operator import itemgetter


def find_median(data):
    data = sorted(data)
    n = len(data)
    if n == 0:
        print ("no median for empty data")
        return 0
    if n%2 == 1:
        return data[n//2]
    else:
        i = n//2
        return (data[i - 1] + data[i])/2
   
# This is hard coded to have 9 intervals for range_percent 10 to support ITEM V1
# if range_percent changes then it needs reworking here to merge top or bottom intervals 
def range_tidal_data(all_dates, date_ranges, tide_class, ln, la):
    perc = tide_class['range_percent']
    per = tide_class['percent']
    fid = tide_class['feature_id'][0]
    tp = list()
    tide_dict = dict()
    for dt in all_dates:
        tp.append(TimePoint(ln, la, dt))
    tides = predict_tide(tp)
    if len(tides) == 0:
        print ("No tide height observed from OTPS model within lat/lon range")
        sys.exit()
    for tt in tides:
        tide_dict[datetime.strptime(tt.timepoint.timestamp.isoformat()[0:19], "%Y-%m-%dT%H:%M:%S")] = tt.tide_m
    tide_list = sorted(tide_dict.items(), key=lambda x: x[1])
    max_height=tide_list[-1][1]
    min_height=tide_list[0][1]
    dr = max_height-min_height
    inc = min_height
    per_range = int(100/perc) - 1
    #print ("Following range is created from min max ("
    #       + str(min_height) + " " + str(max_height) + ")")
    print ("id, per, min, max, observation, LOT, HOT, median")
    for i in range(per_range):
        incmn = inc
        perc = 20 if i == per_range-1 else 10
        inc = float("%.3f" %(incmn + dr*perc*0.01))   
        inc = max_height if i == per_range-1 else inc
        key = 'range_' + str(i+1)
        range_value = [[x[0].strftime('%Y-%m-%d'), x[1]] for x in tide_list
               if (x[1] >= incmn and x[1] <= inc)]
        #median = float("%.3f" %(find_median([x[1] for x in range_value])))
        median = float("%.3f" %(statistics.median([x[1] for x in range_value])))
        if per == (i+1)*10:
            print ("MEDIAN INFO " + str(fid) + "," + str(per) + "," + str(incmn) + "," + str(inc) + "," + str(len(range_value)) + 
                   "," + str(range_value[0][1]) + "," + str(range_value[-1][1]) + "," + str(median))
            return range_value
    return [] 


def extract_otps_computed_data(dates, date_ranges, per, ln, la):
        
    tp = list()
    tide_dict = dict()
    ndate_list=list()
    mnt=timedelta(minutes=15)
    for dt in dates:
        ndate_list.append(dt-mnt)
        ndate_list.append(dt)
        ndate_list.append(dt+mnt)
    for dt in ndate_list:
        tp.append(TimePoint(ln, la, dt))
    tides = predict_tide(tp)
    if len(tides) == 0:
        print ("No tide height observed from OTPS model within lat/lon range")
        sys.exit()
    print ("received from predict tides 15 minutes before and after " + str(datetime.now()))
    # collect in ebb/flow list
    for tt in tides:
        tide_dict[datetime.strptime(tt.timepoint.timestamp.isoformat()[0:19], "%Y-%m-%dT%H:%M:%S")] = tt.tide_m
    tmp_lt = sorted(tide_dict.items(), key=lambda x: x[0])
    dtlist = tmp_lt[1::3]
    my_data = sorted(dtlist, key=itemgetter(1))
    # print ([[dt[0].strftime("%Y-%m-%d %H:%M:%S"), dt[1]] for dt in tmp_lt])
    tmp_lt = [[tmp_lt[i+1][0].strftime("%Y-%m-%d"), 'ph'] \
             if tmp_lt[i][1] < tmp_lt[i+1][1] and tmp_lt[i+2][1] <  tmp_lt[i+1][1]  else \
             [tmp_lt[i+1][0].strftime("%Y-%m-%d"), 'pl'] if tmp_lt[i][1] > tmp_lt[i+1][1] and \
             tmp_lt[i+2][1] >  tmp_lt[i+1][1]  else [tmp_lt[i+1][0].strftime("%Y-%m-%d"),'f'] \
             if tmp_lt[i][1] < tmp_lt[i+2][1] else [tmp_lt[i+1][0].strftime("%Y-%m-%d"),'e'] \
             for i in range(0, len(tmp_lt), 3)]
    #_LOG.info('EBB FLOW tide details for entire archive of LS data %s', str(tmp_lt))
    PERC = 25  if per == 50 else per
    max_height=my_data[-1][1]
    min_height=my_data[0][1]
    dr = max_height - min_height
    lmr = min_height + dr*PERC*0.01   # low tide max range
    hlr = max_height - dr*PERC*0.01   # high tide range
    list_low = list()
    list_high = list()
    if PERC == 50:
        list_high = sorted([[x[0].strftime('%Y-%m-%d'), x[1]] for x  in my_data if (x[1] >= lmr) & (x[1] <= hlr) &
                           (x[0] >= date_ranges[0][0]) & (x[0] <= date_ranges[0][1])])
        print (" 50 PERCENTAGE sorted date tide list " + str(len(high)))
            #_LOG.info('Created middle dates and tide heights for time period: %s %s', date_ranges, str(list_high))
    else:
        list_low = sorted([[x[0].strftime('%Y-%m-%d'), x[1]] for x in my_data if (x[1] <= lmr) & 
                          (x[0] >= date_ranges[0][0]) & (x[0] <= date_ranges[0][1])])
        list_high = sorted([[x[0].strftime('%Y-%m-%d'), x[1]] for x in my_data if (x[1] >= hlr) &
                           (x[0] >= date_ranges[0][0]) & (x[0] <= date_ranges[0][1])])
        #_LOG.info('Created low percentage dates and tide heights for time period: %s %s', date_ranges, str(list_low))
        #_LOG.info('\nCreated high percentage dates and tide heights for time period: %s %s', date_ranges, str(list_high))
    ebb_flow = str([tt for tt in tmp_lt if (datetime.strptime(tt[0], "%Y-%m-%d") >= date_ranges[0][0]) & 
                   (datetime.strptime(tt[0], "%Y-%m-%d") <= date_ranges[0][1])])
    #_LOG.info('\nCreated EBB FLOW for time period: %s %s', date_ranges, ebb_flow)
    #print (" EBB FLOW for this epoch " + str([tt for tt in tmp_lt if (datetime.strptime(tt[0], "%Y-%m-%d") >= date_ranges[0][0]) & 
    #         (datetime.strptime(tt[0], "%Y-%m-%d") <= date_ranges[0][1])]))
    return dtlist, list_low, list_high, ebb_flow

def get_hydrologic_months(tide_class):
    all_dates = list()
    for k,v in tide_class['year'].items():
        year = int(v)
        months = tide_class.get('months')
        if months is not None:
            st_dt =  str(year+1)+str(months[0])+'01'
            en_dt =  str(year+1)+str(months[1])+'30'
        else:
            st_dt = '01/07/' + str(year+1)
            en_dt = '30/11/' + str(year+1)
        date_list = pd.date_range(st_dt, en_dt)
        date_list = date_list.to_datetime().astype(str).tolist()
        all_dates = all_dates + date_list
    return all_dates 

def filter_sub_class(tide_class, list_low, list_high, ebb_flow):
    if tide_class['sub_class'] == 'e':
        if list_low is not None:
            key = set(e[0] for e in eval(ebb_flow) if e[1] == 'e')
            list_low = [ff for ff in list_low if ff[0] in key]
        key = set(e[0] for e in eval(ebb_flow) if e[1] == 'e')
        list_high = [ff for ff in list_high if ff[0] in key]
    if tide_class['sub_class'] == 'f':
        if list_low is not None:
            key = set(e[0] for e in eval(ebb_flow) if e[1] == 'f')
            list_low = [ff for ff in list_low if ff[0] in key]
        key = set(e[0] for e in eval(ebb_flow) if e[1] == 'f')
        list_high = [ff for ff in list_high if ff[0] in key]
    if tide_class['sub_class'] == 'ph':
        if list_low is not None:
            key = set(e[0] for e in eval(ebb_flow) if e[1] == 'ph')
            list_low = [ff for ff in list_low if ff[0] in key]
        key = set(e[0] for e in eval(ebb_flow) if e[1] == 'ph')
        list_high = [ff for ff in list_high if ff[0] in key]
    if tide_class['sub_class'] == 'pl':
        if list_low is not None:
            key = set(e[0] for e in eval(ebb_flow) if e[1] == 'pl')
            list_low = [ff for ff in list_low if ff[0] in key]
        key = set(e[0] for e in eval(ebb_flow) if e[1] == 'pl')
        list_high = [ff for ff in list_high if ff[0] in key]
    return list_low, list_high, ebb_flow


