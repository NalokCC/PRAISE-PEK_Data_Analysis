import time, logging, cartopy, matplotlib, warnings, os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button, TextBox
from math import acos, asin, cos, sin, dist, pi, degrees, sqrt, radians, e
from export_PEK_data import export_PEK_data

# disable annoying warnings
warnings.simplefilter('ignore')

# default variables
logging_level = logging.INFO # set the debug level
toggled_PEKs = ['35'] # set dafault PEKs to view on startup
toggled_elements = ['AR'] # set default elements to view on startup
xticks = 30 # set the default number of ticks to appear on the data subplot
default_start_date, default_end_date = '2024-07-09 00:00:00', '2024-07-11 23:59:59' # set the default time chunk to analyze
limit_map_datapoints = False # set whether or not the map subplot will be limited by the start and end dates
store_old_log = False # set whether or not to save the previous log file that was generated

# Default regression coefficients 
RegC_NO2 = 0.0004462559
RegC_SO2 = 0.0001393235
RegC_O3 = 0.0005116328
RegC_PM10 = 0.0002821751
RegC_PM2_5 = 0.0002180567

# Default infiltration rates # NOTE WILL IMPROVE ACCURACY LATER BY ADDING THE ME OF WINDOW, AC, ETC
IR_baseline_NO2 = 0.857510013618537
IR_baseline_O3 = 0.568442103625238
IR_baseline_PM10 = 0.727655350442508
IR_baseline_PM2_5 = 0.831963620181844
IR_baseline_SO2 = 1

# setup logging
os.remove('logs/recentlog.log') if not store_old_log else os.rename(f'logs/recentlog.log', f'logs/{int(time.time())}log.log')
logger = logging.getLogger('PRAISE-PEK')
logger.setLevel(logging_level) 

formatter = logging.Formatter('[%(levelname)s][%(asctime)s] %(name)s: %(message)s') # format log prefixes
file_handler = logging.FileHandler('logs/recentlog.log')
file_handler.setLevel(logging_level)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

############################
##### WIDGET FUNCTIONS #####
############################

def export_PEK_data_pressed(event):
    ST_export_PEK_data = time.time()
    logger.info(f'PEKs {all_PEKs} are being exported...')
    export_PEK_data(all_PEKs)
    logger.info(f'PEKs {all_PEKs} have been exported in {time.time()-ST_export_PEK_data} seconds.')

def xticks_submitted(inputted_xticks):
    global xticks
    xticks = int(inputted_xticks)
    logger.debug(f'Xticks textbox submitted')
    update_data_subplot()

def start_date_limit_submitted(date):
    global start_date_limit_second_time
    try:
        start_date_limit_second_time = DT_to_seconds(date)
        logger.debug(f'Start date limit textbox submitted')
        update_all()
    except:
        logger.info(f'Invalid date provided in "start date limit submitted textbox"')
    
def end_date_limit_submitted(date):
    global end_date_limit_second_time
    try:
        end_date_limit_second_time = DT_to_seconds(date)
        logger.debug(f'End date limit textbox submitted')
        update_all()
    except:
        logger.info(f'Invalid date provided in "end date limit submitted textbox"')

def paste_x_start_limit_pressed(event):
    global start_date_limit_second_time
    closest_second_time = min(total_PEK_second_times, key=lambda x:int(abs(x-clicked_x_coord))) # find the closest value that the clicked x coord is near
    DT = total_PEK_date_times[total_PEK_second_times.index(closest_second_time)] # find the date time associated with that second time

    start_date_limit_textbox.textbox.set_val(DT) # set the value of the textbox to the found date time
    start_date_limit_second_time = closest_second_time
    logger.debug(f'Paste x start limit button pressed and pasted "{DT}"')
    update_all()
    
def paste_x_end_limit_pressed(event):
    global end_date_limit_second_time
    closest_second_time = min(total_PEK_second_times, key=lambda x:int(abs(x-clicked_x_coord))) # find the closest value that the clicked x coord is near
    DT = total_PEK_date_times[total_PEK_second_times.index(closest_second_time)] # find the date time associated with that second time

    end_date_limit_textbox.textbox.set_val(DT) # set the value of the textbox to the found date time
    end_date_limit_second_time = closest_second_time
    logger.debug(f'Paste x end limit button pressed and pasted "{DT}"')
    update_all()

def update_all_pressed(event):
    logger.debug('Update all button pressed')
    update_all()

def toggle_limit_map_datapoints_pressed(event):
    global limit_map_datapoints
    limit_map_datapoints = not limit_map_datapoints
    toggle_limit_map_datapoints_button.button.color = 'green' if limit_map_datapoints else 'red'
    logger.info(f'Limit map datapoints toggled to {limit_map_datapoints}')
    update_map_subplot()

def save_fig_pressed(event):
    file_name = 'data/exported/'
    for PEK in toggled_PEKs:
        file_name += f'{PEK}+'
    for element in toggled_elements:
        file_name += f'{element}_'
    file_name += f'{int(time.time())}figure.png'
    plt.savefig(file_name)
    logger.info(f'Save figure button pressed and saved figure to "{file_name}"')

#####################
##### FUNCTIONS #####
#####################

##### GENERAL #####
def exists(var_as_string):
    try:
        x = globals()[f"{var_as_string}"]
        return True
    except:
        return False

def DT_to_seconds(DT, format='YYYY-MM-DD hh:mm:ss'): 
    DT = str(DT)
    Y = format.index('Y') if 'Y' in format else None
    M = format.index('M') if 'M' in format else None
    D = format.index('D') if 'D' in format else None
    h = format.index('h') if 'h' in format else None
    m = format.index('m') if 'm' in format else None
    s = format.index('s') if 's' in format else None

    years = int(DT[Y:Y+format.count('Y')])-2024 if Y != None else 0
    months = int(DT[M:M+2]) if M != None else 0
    days = int(DT[D:D+2]) if D != None else 0
    hours = int(DT[h:h+2]) if h != None else 0
    minutes = int(DT[m:m+2]) if m != None else 0
    seconds = int(DT[s:s+2]) if s != None else 0
    for month in range(1,int(months)):
        days += 31 if month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12 else 0
        days += 30 if month == 4 or month == 6 or month == 9 or month == 11 else 0
        days += 29 if month == 2 and years % 4 == 0 else 29 if month == 2 else 0
    return seconds + minutes*60 + hours*60*60 + days*24*60*60

def coords_to_km_distance(lat1, lon1, lat2, lon2):
    return 2*6371*asin(sqrt(sin(radians((lat1-lat2)/2))**2 + cos(radians(lat2)) * cos(radians(lat1)) * sin(radians((lon1-lon2)/2))**2))

def calculate_variance(inputs):
    return sum([(x-sum(inputs)/len(inputs))**2 for x in inputs])/(len(inputs)-1) # single line version of function
    
def chunk_list(input_list, chunk_size, skip_mod=1):
    return [input_list[i:i+chunk_size] for i, _ in enumerate(input_list[:-chunk_size]) if i % skip_mod == 0] # single line version of function

def limit_data_in_lists(lists, indices=None, range=None):
    if indices != None:
        return [[val for i, val in enumerate(list) if i not in indices] for list in lists]
    elif range != None:
        return [l[range[0]:range[1]] for l in lists]
    else:
        logger.warning(f'When removing indicies from lists {lists}, {indices=} and {range=}')

##### UNPACKING DATA FUNCTIONS #####
def unpack_PEK_data():
    global all_PEKs, all_elements
    ST_unpack_PEK_data = time.time()
    all_PEKs, all_elements = [], []
    for sheet_name in pd.ExcelFile('data/export.xlsx').sheet_names:
        ST_read_sheet = time.time()
        PEK_number = sheet_name[-2:]
        all_PEKs.append(PEK_number) if PEK_number not in all_PEKs else None
        PEK_sheet = pd.read_excel('data/export.xlsx', sheet_name=sheet_name)
        PEK_data = list(zip(*PEK_sheet.values.tolist()))
        PEK_columns = list(PEK_sheet.columns)

        PEK_data.append(list(map(DT_to_seconds, PEK_data[list(PEK_columns).index('DateTime')])))
        PEK_columns.append('DT_seconds')
        for ii, column in enumerate(PEK_columns):
            element_name = column.split(' ')[0].replace('.','_')
            all_elements.append(element_name) if element_name not in all_elements and 'str' not in str(type(PEK_data[ii][0])) else None
            # using the information from https://www.breeze-technologies.de/blog/air-pollution-how-to-convert-between-mgm3-µgm3-ppm-ppb/ 
            mod = 1.15 if element_name == 'CO' else 1.88 if element_name == 'NO2' else 1.96 if element_name == 'O3' else 1
            element_data = [int(val)*mod for val in PEK_data[ii]] if 'str' not in str(type(PEK_data[ii][0])) else PEK_data[ii]
            globals()[f"PEK_{PEK_number}_{element_name}"] = element_data
        logger.debug(f'Read sheet "{sheet_name}" in {time.time()-ST_read_sheet} seconds')
    logger.info(f'Unpacked PEK data in {time.time()-ST_unpack_PEK_data} seconds')
    
def unpack_PRAISE_data():
    ST_unpack_PRAISE_data = time.time()
    global PRAISE_start_dates, PRAISE_end_dates, PRAISE_lats, PRAISE_lons, PRAISE_start_seconds, PRAISE_end_seconds
    PRAISE_csv = pd.read_csv('data/2024_July.csv')
    unpacked_data = list(zip(*PRAISE_csv.values.tolist()[::-1]))
    unpacked_data = [list(element[::-1]) for element in unpacked_data]
    PRAISE_start_dates, PRAISE_end_dates, PRAISE_lats, PRAISE_lons, _, _, _, _, _, _, _, _ = unpacked_data
    PRAISE_start_seconds = [DT_to_seconds(DT) for DT in PRAISE_start_dates]
    PRAISE_end_seconds = [DT_to_seconds(DT) for DT in PRAISE_end_dates]

    logger.info(f'Unpacked PRAISE data in {time.time()-ST_unpack_PRAISE_data} seconds')

def unpack_ADMS_data():
    ST_unpack_ADMS_data = time.time()
    global ADMS_data_array, ADMS_data_name_array
    ADMS_data_array = []
    ADMS_data_name_array = []
    ADMS_csv = pd.read_csv('data/adms_output.csv')
    unpacked_data = list(zip(*ADMS_csv.values.tolist()))

    unpacked_data += [DT_to_seconds(time, 'YYYYMMDDhh') for time in unpacked_data[0]]
    for i, column_name in enumerate(list(ADMS_csv.columns)):
        column_name = 'ADMS_' + column_name.replace('(','_').replace(')','').replace('%','_pct').replace('.','_').replace('/','_').lower()
        globals()[f"{column_name}"] = unpacked_data[i]
        if 'ug_m3' in column_name or column_name == 'ADMS_ar_pct':
            ADMS_data_name_array.append(column_name)
            ADMS_data_array.append(globals()[f"{column_name}"])
    globals()['ADMS_seconds_times'] = [DT_to_seconds(time, 'YYYYMMDDhh') for time in unpacked_data[0]]
    logger.info(f'Unpacked ADMS data in {time.time()-ST_unpack_ADMS_data} seconds. {ADMS_data_name_array=}')

##### DATA ANALYSIS #####
def clean_PRAISE_data(angle_threshold=20):
    ST_clean_PRAISE_data = time.time()
    total_indicies_removed = 0

    global PRAISE_lats, PRAISE_lons, PRAISE_start_dates, PRAISE_end_dates, PRAISE_start_seconds, PRAISE_end_seconds
    total_indices = len(PRAISE_lats)

    div_0_count, reached_angle_threshold_count, reached_velocity_threshold_count, dupe_count = 1, 1, 1, 1
    while div_0_count + reached_angle_threshold_count + reached_velocity_threshold_count > 0: 
        PRAISE_outliers = []
        div_0_count, reached_angle_threshold_count, reached_velocity_threshold_count, dupe_count = 0, 0, 0, 0
        for i, (x, y) in enumerate([*zip(PRAISE_lats,PRAISE_lons)]):
            if i not in PRAISE_outliers and i != 0 and i != len(PRAISE_lats)-1:
                a = coords_to_km_distance(PRAISE_lats[i-1], PRAISE_lons[i-1], x, y)
                b = coords_to_km_distance(x, y, PRAISE_lats[i+1], PRAISE_lons[i+1])
                c = coords_to_km_distance(PRAISE_lats[i-1], PRAISE_lons[i-1], PRAISE_lats[i+1], PRAISE_lons[i+1])
                if a * b == 0:
                    div_0_count += 1
                    PRAISE_outliers.append(i)
                else:
                    eq = (a**2 + b**2 - c**2) / (2 * a * b)
                    eq = -1 if eq < -1 else 1 if eq > 1 else eq
                    if acos(eq)*180/pi <= angle_threshold:
                        reached_angle_threshold_count += 1
                        PRAISE_outliers.append(i)
                if PRAISE_start_seconds[i] == PRAISE_start_seconds[i-1]:
                    PRAISE_outliers.append(i-1)
                    dupe_count += 1
                elif i not in PRAISE_outliers and a/(PRAISE_start_seconds[i] - PRAISE_start_seconds[i-1]) > .03:
                    reached_velocity_threshold_count += 1
                    PRAISE_outliers.append(i)
        total_indicies_removed += len(PRAISE_outliers)
        PRAISE_start_dates, PRAISE_end_dates, PRAISE_start_seconds, PRAISE_end_seconds, PRAISE_lats, PRAISE_lons = limit_data_in_lists([PRAISE_start_dates, PRAISE_end_dates, PRAISE_start_seconds, PRAISE_end_seconds, PRAISE_lats, PRAISE_lons], PRAISE_outliers)
    logger.info(f'Cleaned PRAISE data in {time.time()-ST_clean_PRAISE_data} seconds and removed {total_indicies_removed}/{total_indices} indices')

def simplify_PRAISE_data(chunk_size=6, skip_mod=1, variance_threshold=.01, distance_threshold=0.06):
    ST_simplify_PRAISE_data = time.time()
    global low_variance_coords, low_variance_lats, low_variance_lons
    loc_lats, loc_lons = PRAISE_lats, PRAISE_lons
    simplified_points, low_variance_coords = 1, []

    while simplified_points > 0:
        simplified_points, low_variance_indices = 0, [] 
        for i, [lats, lons] in enumerate([*zip(chunk_list(loc_lats, chunk_size, skip_mod), chunk_list(loc_lons, chunk_size, skip_mod))]):
            avg_lat = sum(lats)/len(lats)
            avg_lon = sum(lons)/len(lons)
            variance = calculate_variance([coords_to_km_distance(avg_lat, avg_lon, lat, lon) for lat, lon in zip(lats,lons)])

            if variance < variance_threshold:
                loc_lats[i*skip_mod], loc_lons[i*skip_mod] = avg_lat, avg_lon
                low_variance_coords.append([avg_lat,avg_lon])
                low_variance_indices.extend([*range(i*skip_mod+1,i*skip_mod+chunk_size)])
                simplified_points += 1
        loc_lats = [lat for i, lat in enumerate(loc_lats) if i not in low_variance_indices]
        loc_lons = [lon for i, lon in enumerate(loc_lons) if i not in low_variance_indices]

    simplified_points = 1
    while simplified_points > 0:
        simplified_points = 0
        for i, [lat1, lon1] in enumerate(low_variance_coords[:-1]):
            for ii, [lat2, lon2] in enumerate(low_variance_coords[i+1:]):
                ii += i+1
                if ii >= len(low_variance_coords):
                    break
                if coords_to_km_distance(lat1, lon1, lat2, lon2) < distance_threshold:
                    low_variance_coords[i] = [(lat1+lat2)/2, (lon1+lon2)/2]
                    del low_variance_coords[ii]
                    simplified_points += 1
    
    low_variance_lats, low_variance_lons = [lat for lat, _ in low_variance_coords], [lon for _, lon in low_variance_coords]
    logger.info(f'Simplified PRAISE data in {time.time()-ST_simplify_PRAISE_data} seconds. {len(low_variance_coords)} low variance coords were found')

def average_PEK_data(time_chunk_s=1200):
    for PEK in all_PEKs:
        for element in all_elements:
            try:
                data = globals()[f"PEK_{PEK}_{element}"]
                data_seconds = globals()[f"PEK_{PEK}_DT_seconds"]
            except:
                continue
            starting_second = data_seconds[0]
            ii = 1
            prev_ind = 0
            new_element_data, new_seconds_data = [], []
            for i, second in enumerate(data_seconds):
                if second >= starting_second+time_chunk_s*ii:
                    new_element_data.append(sum(data[prev_ind:i])/(i-prev_ind))
                    new_seconds_data.append(sum(data_seconds[prev_ind:i])/(i-prev_ind))
                    prev_ind = i
                    ii += 1
            globals()[f'PEK_{PEK}_{element}'] = new_element_data
            globals()[f'PEK_{PEK}_DT_seconds'] = new_seconds_data

def create_spreadsheet_praise_specifics(*args):
    ST_create_spreadsheet_PRAISE_parsing = time.time()
    columns = ['times', 'lons', 'lats']
    columns_data = [[x.replace('-','').replace(':','').replace(' ','') for x in PRAISE_start_dates], PRAISE_lons, PRAISE_lats]
    df = pd.DataFrame({k:v for k,v in zip(columns, [column_data for column_data in columns_data])})
    df.to_excel('data/exported/data.xlsx', sheet_name='Main Sheet')

    logger.info(f'Successfully created spreadsheet for PRAISE parsing in {time.time()-ST_create_spreadsheet_PRAISE_parsing} seconds')
    if 'quit' in args:
        logger.info(f'Quit arg given after spreadsheet was created')
        quit()

def match_ADMS_data():
    global ADMS_latitude, ADMS_longitude, ADMS_data_array, ADMS_seconds_times
    ST_match_ADMS_data = time.time()
    for ADMS_second_time in set(ADMS_seconds_times):
        if ADMS_seconds_times.count(ADMS_second_time) > 1:
            indices = [ii for ii, t in enumerate(ADMS_seconds_times) if t == ADMS_second_time]
            PRAISE_ind = PRAISE_start_seconds.index(min(PRAISE_start_seconds, key=lambda x: abs(x-ADMS_second_time)))
            PRAISE_lat, PRAISE_lon = PRAISE_lats[PRAISE_ind], PRAISE_lons[PRAISE_ind]
            distances = [coords_to_km_distance(PRAISE_lat, PRAISE_lon, ADMS_lat, ADMS_lon) for ADMS_lat, ADMS_lon in zip([ADMS_latitude[ind] for ind in indices], [ADMS_longitude[ind] for ind in indices])]
            indices.remove(indices[distances.index(min(distances))])
            ADMS_data_array = limit_data_in_lists(ADMS_data_array, indices=indices)
            [ADMS_seconds_times] = limit_data_in_lists([ADMS_seconds_times], indices=indices)
        
    logger.info(f'Matched ADMS data in {time.time()-ST_match_ADMS_data} seconds')

def calculate_PEK_ARExposure():
    ST_calculate_PEK_ARExposures = time.time()
    global test_coords_lime, test_coords_light_blue
    test_coords_lime, test_coords_light_blue = [], []
    for PEK in all_PEKs:
        NO2_data = globals()[f"PEK_{PEK}_NO2"]
        O3_data = globals()[f"PEK_{PEK}_O3"]
        PM10_data = globals()[f"PEK_{PEK}_PM10"]
        PM2_5_data = globals()[f"PEK_{PEK}_PM2_5"]
        seconds_time_data = globals()[f"PEK_{PEK}_DT_seconds"]
        date_time_data = globals()[f"PEK_{PEK}_DateTime"]

        starting_second = seconds_time_data[0]
        RA_conc_NO2, RA_conc_O3, RA_conc_PM10, RA_conc_PM2_5, RA_conc_SO2 = [], [], [], [], []
        RA_AR_NO2, RA_AR_O3, RA_AR_PM10, RA_AR_PM2_5, RA_AR_SO2 = [], [], [], [], []
        RA_AR, RA_AR_seconds_times, RA_AR_date_times = [], [], []
        for hour in range(0,int((seconds_time_data[-1]-seconds_time_data[0])/3600)):
            curr_hour_seconds_time = starting_second+(hour*3600)
            RA_seconds = 10800 # how far apart the rolling average is. default is 10800 (10800 seconds, or 3 hours)
            curr_hour_index = seconds_time_data.index(min(seconds_time_data, key=lambda x:abs(x-curr_hour_seconds_time)))
            prev_hour_index = seconds_time_data.index(min(seconds_time_data, key=lambda x:abs(x-curr_hour_seconds_time+RA_seconds/2)))
            next_hour_index = seconds_time_data.index(min(seconds_time_data, key=lambda x:abs(x-curr_hour_seconds_time-RA_seconds/2)))

            PRAISE_index = PRAISE_start_seconds.index(min(PRAISE_start_seconds, key=lambda x:abs(x-curr_hour_seconds_time)))
            lat1, lon1 = PRAISE_lats[PRAISE_index], PRAISE_lons[PRAISE_index]
            distances = [coords_to_km_distance(lat1, lon1, lat2, lon2) for lat2, lon2 in low_variance_coords]
            indoors = True if min(distances) < 0.05 else False

            index_diff = next_hour_index-prev_hour_index
            THRA_avg = lambda data: sum(data[prev_hour_index:next_hour_index])/index_diff
            NO2_conc = THRA_avg(NO2_data)
            O3_conc = THRA_avg(O3_data)
            PM10_conc = THRA_avg(PM10_data)
            PM2_5_conc = THRA_avg(PM2_5_data)
            SO2_conc = 2 # assuming 2 microg/m^3

            RA_conc_NO2.append(NO2_conc)
            RA_conc_O3.append(O3_conc)
            RA_conc_PM10.append(PM10_conc)
            RA_conc_PM2_5.append(PM2_5_conc)
            RA_conc_SO2.append(SO2_conc)
            
            calc_AR = lambda regc, conc, IR_baseline: (e**(regc * conc) - 1) / [1 if not indoors else IR_baseline][0]
            NO2_AR = calc_AR(RegC_NO2, NO2_conc, IR_baseline_NO2)
            O3_AR = calc_AR(RegC_O3, O3_conc, IR_baseline_O3)
            PM10_AR = calc_AR(RegC_PM10, PM10_conc, IR_baseline_PM10)
            PM2_5_AR = calc_AR(RegC_PM2_5, PM2_5_conc, IR_baseline_PM2_5)
            SO2_AR = calc_AR(RegC_SO2, SO2_conc, IR_baseline_SO2)

            RA_AR_NO2.append(NO2_AR)
            RA_AR_O3.append(O3_AR)
            RA_AR_PM10.append(PM10_AR)
            RA_AR_PM2_5.append(PM2_5_AR)
            RA_AR_SO2.append(SO2_AR)

            RA_AR.append((NO2_AR + O3_AR + SO2_AR + max([PM10_AR, PM2_5_AR])) * 100)
            RA_AR_seconds_times.append(seconds_time_data[curr_hour_index])
            RA_AR_date_times.append(date_time_data[curr_hour_index])
        globals()[f"PEK_{PEK}_AR"] = RA_AR
        globals()[f"PEK_{PEK}_AR_seconds_times"] = RA_AR_seconds_times
        globals()[f"PEK_{PEK}_AR_date_times"] = RA_AR_date_times
    logger.info(f'Calculated AR% in {time.time()-ST_calculate_PEK_ARExposures} seconds.')

##### UPDATING SUBLOPTS #####
def update_data_subplot(drawADMS=True):
    ST_update_data_subplot = time.time()
    data_subplot.cla() # clear previous subplot data
    global total_PEK_second_times, total_PEK_date_times, element_data, seconds_times, date_times, clicked_line, ADMS_seconds_times

    # setup arrays for finding xticks later 
    total_PEK_second_times, total_PEK_date_times = [], []
    for i, PEK in enumerate(toggled_PEKs):
        for ii, element in enumerate(toggled_elements): # iterate through each element for each PEK toggled
            try:
                # pull data PEK data
                element_data = globals()[f"PEK_{PEK}_{element}"]
                seconds_times = list(globals()[f"PEK_{PEK}_DT_seconds"]) if element != 'AR' else list(globals()[f"PEK_{PEK}_AR_seconds_times"])
                date_times = globals()[f"PEK_{PEK}_DateTime"] if element != 'AR' else globals()[f"PEK_{PEK}_AR_date_times"]
                
                sd_index_PEK = seconds_times.index(min(seconds_times, key=lambda x:abs(x-start_date_limit_second_time))) if start_date_limit_second_time != 0 else 0
                ed_index_PEK = seconds_times.index(min(seconds_times, key=lambda x:abs(x-end_date_limit_second_time))) if end_date_limit_second_time != 0 else len(seconds_times)-1
                element_data, seconds_times, date_times = element_data[sd_index_PEK:ed_index_PEK], seconds_times[sd_index_PEK:ed_index_PEK], date_times[sd_index_PEK:ed_index_PEK]

                # plot PEK data
                data_subplot.plot(seconds_times, element_data, label=f'{element} | PEK {PEK}')

                # plot ADMS data if enabled
                if drawADMS:
                    sd_index_ADMS = ADMS_seconds_times.index(min(ADMS_seconds_times, key=lambda x:abs(x-start_date_limit_second_time))) if start_date_limit_second_time != 0 else 0
                    ed_index_ADMS = ADMS_seconds_times.index(min(ADMS_seconds_times, key=lambda x:abs(x-end_date_limit_second_time))) if end_date_limit_second_time != 0 else len(ADMS_seconds_times)-1

                    for i, name in enumerate(ADMS_data_name_array):
                        if element.lower() in name:
                            ADMS_data = ADMS_data_array[i]
                            data_subplot.plot(ADMS_seconds_times[sd_index_ADMS:ed_index_ADMS], ADMS_data[sd_index_ADMS:ed_index_ADMS], label=f'{element} | ADMS Data') if ADMS_data != None else None
                            break

                # append values if they are not already present, then sort them relative to one another
                total_PEK_second_times += [item for item in seconds_times if item not in total_PEK_second_times]
                total_PEK_date_times += [item for item in date_times if item not in total_PEK_date_times]
                total_PEK_second_times, total_PEK_date_times = [list(x) for x in zip(*sorted(zip(total_PEK_second_times, total_PEK_date_times)))]
            except Exception as error:
                logger.warning(f'PEK_{PEK}_{element} does not exist or other error occurred when toggling elements.\n[ERROR MESSAGE] {error}')
        drawADMS = False

    # set label
    xticks_textbox.label_obj.set_text(f'xTicks (max {len(total_PEK_second_times)})')

    # setup xticks
    plt.sca(data_subplot)
    tick_locations = total_PEK_second_times[::int(len(total_PEK_second_times)/xticks)] if len(total_PEK_second_times) > xticks else total_PEK_second_times # find xticks locations
    labels = total_PEK_date_times[::int(len(total_PEK_date_times)/xticks)] if len(total_PEK_date_times) > xticks else total_PEK_date_times  # find xticks labels
    plt.xticks(tick_locations, labels, rotation=45)

    # set general settings for subplot
    data_subplot.grid(True)
    data_subplot.legend()

    logger.debug(f'Data subplot updated in {time.time()-ST_update_data_subplot} seconds')
    plt.draw()

def update_map_subplot():
    ST_update_map_subplot = time.time()
    map_subplot.cla() # clear previous subplot data

    map_subplot.imshow(mpimg.imread('data/satellite_view.png'), extent=[114.060070, 114.350106, 22.167814, 22.358208]) # overlay satellite image
    map_subplot.gridlines(draw_labels=True)

    ss_index = PRAISE_start_seconds.index(min(PRAISE_start_seconds, key=lambda x:abs(x-start_date_limit_second_time))) if limit_map_datapoints else 0
    es_index = PRAISE_end_seconds.index(min(PRAISE_end_seconds, key=lambda x:abs(x-end_date_limit_second_time))) if end_date_limit_second_time != 0 and limit_map_datapoints else len(PRAISE_end_seconds)-1

    lons, lats = limit_data_in_lists([PRAISE_lons, PRAISE_lats], range=[ss_index,es_index])

    map_subplot.plot(lons, lats, marker='o', markersize=1, transform=proj, color = 'red') 
    [map_subplot.plot(lon, lat, marker='o', markersize=8, transform=proj, color='orange', alpha=.6) for lat, lon in low_variance_coords]
    [map_subplot.plot(lon, lat, marker='o', markersize=6, transform=proj, color='lime') for lat, lon in test_coords_lime]
    [map_subplot.plot(lon, lat, marker='o', markersize=6, transform=proj, color='lightblue') for lat, lon in test_coords_light_blue]

    plt.draw()
    logger.debug(f'Map subplot updated in {time.time()-ST_update_map_subplot} seconds')

def update_all(event=None):
    update_map_subplot()
    update_data_subplot()

##### SETUP #####
def setup_plots(**kwargs):
    ST_loading_plot = time.time() # record start time
    plot_loaded = False
    global fig, proj, data_subplot, map_subplot, clicked_x_coord, limit_map_datapoints

    logger.debug(f'Current backend is {plt.get_backend()}') # log backend

    # setup figure
    fig = plt.figure()

    # pop kwargs
    font_size = kwargs.pop('font_size', 6)
    line_width = kwargs.pop('line_width', .8)
    tight_layout_enabled = kwargs.pop('tight_layout_enabled', True)
    auto_fullscreen = kwargs.pop('auto_fullscreen', False)

    # set matplotlib params
    matplotlib.rcParams['font.size'] = font_size
    matplotlib.rcParams['lines.linewidth'] = line_width
    plt.get_current_fig_manager().window.state('zoomed') if auto_fullscreen == True else 0

    # setup subplots
    data_subplot = plt.subplot(2,1,1) # sets the data subplot to the one on the top
    proj = cartopy.crs.PlateCarree() # set type of projecetion
    map_subplot = plt.subplot(2,1,2, projection=proj) # sets the map subplot to the one on the bottom
    fig.tight_layout() if tight_layout_enabled == True else 1

    # create classes to handle making buttons and textboxes easier
    class create_button:
        def __init__(self, label, location, clicked_function, color='lightgrey'):
            self.ax = fig.add_axes(location)
            self.button = Button(self.ax, label)
            self.button.on_clicked(clicked_function)
            self.pos = self.button.ax.get_position()
            self.button.color = color

            logger.debug(f'Button "{label}" successfully created with {location=}, {clicked_function=}')

        def set_color(self, color):
            self.button.color = color

    class create_textbox:
        def __init__(self, label, location, submit_function, label_pos='top', default_val=''):
            self.ax = fig.add_axes(location)
            self.textbox = TextBox(self.ax, label)
            self.textbox.on_submit(submit_function)
            self.label_obj = self.textbox.ax.get_children()[0]
            self.pos = self.textbox.ax.get_position()
            self.textbox.set_val(default_val)

            if label_pos == 'top':
                self.label_obj.set_position([0.5,1.4])
                self.label_obj.set_verticalalignment('top')
                self.label_obj.set_horizontalalignment('center')
            
            logger.debug(f'TextBox "{label}" successfully created with {location=}, {submit_function=}, {label_pos=}')

    # setup buttons and textboxes
    global xticks_textbox, start_date_limit_textbox, end_date_limit_textbox
    xticks_textbox = create_textbox(f'xTicks (max {plt.xlim()[1]})', [0.08, 0.02, .05, .03], xticks_submitted)
    start_date_limit_textbox = create_textbox(f'start (): yyyy-mm-dd hh:mm:ss', [0.08,0.4,0.1,0.03], start_date_limit_submitted, default_val=default_start_date)
    end_date_limit_textbox = create_textbox(f'end (): yyyy-mm-dd hh:mm:ss', [0.2,0.4,0.1,0.03], end_date_limit_submitted, default_val=default_end_date)
    
    global force_update_plots_button, export_PEK_data_button, save_fig_button, paste_x_start_limit_button, paste_x_end_limit_button, toggle_limit_map_datapoints_button
    # generate_map_procedurally_button = create_button('Generate Map Procedurally', [0.7,0.1,button_xdim,button_ydim], dummy)
    force_update_plots_button = create_button('Update All', [0.98-button_xdim,0.4,button_xdim,button_ydim], update_all)
    export_PEK_data_button = create_button('Export PEK Data', [0.98-button_xdim,0.45,button_xdim,button_ydim], export_PEK_data_pressed)
    save_fig_button = create_button('Save Current Fig', [0.98-button_xdim,0.35,button_xdim,button_ydim], save_fig_pressed)
    paste_x_start_limit_button = create_button('Paste Clicked X', [start_date_limit_textbox.pos.x0+.02,start_date_limit_textbox.pos.y0-.06,
                                               start_date_limit_textbox.pos.x1-start_date_limit_textbox.pos.x0-.04,
                                               start_date_limit_textbox.pos.y1-start_date_limit_textbox.pos.y0], paste_x_start_limit_pressed)
    paste_x_end_limit_button = create_button('Paste Clicked X', [end_date_limit_textbox.pos.x0+.02,end_date_limit_textbox.pos.y0-.06,
                                               end_date_limit_textbox.pos.x1-end_date_limit_textbox.pos.x0-.04,
                                               end_date_limit_textbox.pos.y1-end_date_limit_textbox.pos.y0], paste_x_end_limit_pressed)
    toggle_limit_map_datapoints_button = create_button('Limit with PEK data?', [0.7,0.2,0.05,0.02], toggle_limit_map_datapoints_pressed, 'red')

    # create classes to procedurally make the pek and element buttons easier
    class create_PEK_button:
        def __init__(self, PEK_number, location, **kwargs):
            self.button = create_button(f'PEK {PEK_number}', location, self.clicked, color='red') # create button
            self.PEK_number = PEK_number
            self.toggled = False # set default to disabled
            # set label value to the side if desired
            self.value_label = fig.text(self.button.pos.x0-0.01, (self.button.pos.y0+self.button.pos.y1)/2, self.toggled) if kwargs.pop('label_visible', False) else None
            self.clicked() if PEK_number in toggled_PEKs else None # if PEK in list already on the creation of this object (set by default), then toggle immediately

        def clicked(self, event=None):
            global toggled_PEKs
            self.toggled = not self.toggled # toggle
            toggled_PEKs.append(self.PEK_number) if self.toggled == True and self.PEK_number not in toggled_PEKs else None # append PEK to list if enabled
            toggled_PEKs.remove(self.PEK_number) if self.toggled == False and self.PEK_number in toggled_PEKs else None # remove PEK from list if disabled
            self.value_label.set_text(self.toggled) if self.value_label != None else None # set label vaule if enabled
            self.button.set_color('lime') if self.toggled else self.button.set_color('red') # set color of botton to match toggle value
            logger.info(f'Button "PEK {PEK_number}" was toggled to {self.toggled}')
            update_all() if plot_loaded == True else None

    class create_element_button:
        def __init__(self, element_name, location, **kwargs):
            self.button = create_button(f'{element_name}', location, self.clicked, color='red') # create button
            self.element_name = element_name
            self.toggled = False # set default to disabled
            # set label value to the side if desired
            self.value_label = fig.text(self.button.pos.x1+0.01, (self.button.pos.y0+self.button.pos.y1)/2, self.toggled) if kwargs.pop('label_visible', False) else None
            self.clicked() if element_name in toggled_elements else None # if element in list already on the creation of this object (set by default), then toggle immediately

        def clicked(self, event=None):
            global toggled_elements
            self.toggled = not self.toggled # toggle
            toggled_elements.append(self.element_name) if self.toggled == True and self.element_name not in toggled_elements else None # append element to list if enabled
            toggled_elements.remove(self.element_name) if self.toggled == False and self.element_name in toggled_elements else None # remove element from list if disabled
            self.value_label.set_text(self.toggled) if self.value_label != None else None # set label value if enabled
            self.button.set_color('lime') if self.toggled else self.button.set_color('red') # set color of botton to match toggle value
            logger.info(f'Button "{self.element_name}" was toggled to {self.toggled}')
            update_all() if plot_loaded == True else None
    
    # setup PEK buttons
    for i, sheet_name in enumerate(pd.ExcelFile('data/export.xlsx').sheet_names):
        PEK_number = sheet_name[-2:]
        globals()[f"PEK_{PEK_number}_button"] = create_PEK_button(PEK_number, [.98-button_xdim, 0.02+(button_ydim+0.01)*i, button_xdim, button_ydim])

    # setup element buttons
    for i, element in enumerate(['PM2_5', 'PM10', 'PM1', 'CO', 'NO2', 'O3', 'VOC', 'Humidity', 'Temperature', 'BatVol', 'CO2', 'NO2_Raw', 'AR']):
        globals()[f"element_{element}_button"] = create_element_button(element, [.02, 0.02+(button_ydim+0.01)*i, button_xdim, button_ydim])

    # set mouse click event
    def mouse_click_event(event):
        global clicked_x_coord, clicked_y_coord, clicked_line, clicked_map_point
        if event.inaxes == data_subplot:
            clicked_x_coord = event.xdata
            clicked_y_coord = event.ydata
            logger.debug(f'Data subplot clicked at x={clicked_x_coord}, y={clicked_y_coord}')

            # draw line where last clicked and remove the previous one
            clicked_line.remove() if exists('clicked_line') else None
            clicked_line = data_subplot.axvline(min(total_PEK_second_times, key=lambda x:abs(x-clicked_x_coord)), linewidth=1) if len(total_PEK_date_times) != 0 else None
            plt.draw()

            # draw point where lat and lon is similar 
            clicked_map_point[-1].remove() if exists('clicked_map_point') else None
            clicked_map_point = map_subplot.plot(PRAISE_lons[PRAISE_start_seconds.index(min(PRAISE_start_seconds, key=lambda x:abs(x-clicked_x_coord)))], PRAISE_lats[PRAISE_start_seconds.index(min(PRAISE_start_seconds, key=lambda x:abs(x-clicked_x_coord)))], marker='o', markersize=8, alpha=0.8, transform=proj, color='lightblue')
            plt.draw() 
            
    clicked_coords = fig.canvas.mpl_connect('button_press_event', mouse_click_event)
    
    plot_loaded = True
    logger.info(f'Setup plot in {time.time()-ST_loading_plot} seconds') # log loading time in seconds

# main code run
if __name__ == '__main__':
    logger.info('--------------START--------------') # mark code init
    ST_main = time.time()

    # set default code variables
    start_date_limit_second_time = DT_to_seconds(default_start_date) if default_start_date != None else 0
    end_date_limit_second_time = DT_to_seconds(default_end_date) if default_end_date != None else 0
    button_xdim, button_ydim = 0.04, 0.025
    clicked_x_coord = 0

    # data management
    unpack_PRAISE_data()
    clean_PRAISE_data()
    simplify_PRAISE_data()

    unpack_ADMS_data()
    match_ADMS_data()

    unpack_PEK_data()
    # average_PEK_data(60)
    calculate_PEK_ARExposure()
    # create_spreadsheet_praise_specifics('quit')
    
    # visual setup
    setup_plots()
    update_all()
    logger.info(f'Successfully initiated program in {time.time()-ST_main} seconds')
    plt.show()

    logger.info('---------------END---------------') # mark code deinit