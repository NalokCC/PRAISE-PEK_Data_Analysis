import time
from pandas import read_excel, ExcelFile, read_csv
from warnings import simplefilter as warningSfilter
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox
import numpy as np
import cartopy
import geopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from export_PEK_data import export_PEK_data
# from export_PRAISE_data import export_PRAISE_data
from math import acos, asin, cos, sin, dist, pi, degrees, sqrt, radians

ST_loading = time.time()
# supress warnings
warningSfilter('ignore')

# set matplotlib parameters
# matplotlib.rcParams.update({'font.size': 6}) # font size
matplotlib.rcParams['lines.linewidth'] = .4 # line width
matplotlib.rcParams['font.size'] = 6 # line width

# setup subplots
proj = ccrs.PlateCarree()
fig = plt.figure()
data_subplot = plt.subplot(2,1,1)# sets the data subplot to the one on the left
map_subplot = plt.subplot(2,1,2, projection=proj) # sets the map subplot to the one on the right
plt.sca(map_subplot)

# global variables
button_xdim, button_ydim = 0.045, 0.03
PEKs = []
elements = []
element_data = []
xticks = 20
start_date_limit = ''
end_date_limit = ''
start_index_limit = 0
end_index_limit = 0
yeamod_start_times = []
clicked_x_coord = 0
clicked_y_coord = 0
DateTime = []
generate_map_procedurally = False
lons = []
lats = []
PRAISE_invalid_indices = []
lat_to_km = 110.574
lon_to_km = 111.320

# classes
class c_animation:
    def __init__(self, figure, function, interval):
        self.figure = figure
        self.function = function
        self.interval = interval
        
    def start(self):
        self.anim = FuncAnimation(fig=self.figure, func=self.function,interval=self.interval, repeat=False)
    
    def stop(self):
        try:
            self.anim.pause()
        except:
            pass

# functions
def YeaMoD_HoMiS_24_to_seconds(YeaMoD_HoMiS, split_chars=['-',':']):
    try:
        YeaMoD, HoMiS = YeaMoD_HoMiS.split(' ')
        years, months, days = map(int, YeaMoD.split(split_chars[0]))
        hours, minutes, seconds = map(int, HoMiS.split(split_chars[1]))
        years_since_2024 = years-2024
        for month in range(1,int(months)):
            days += 31 if month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12 else 0
            days += 30 if month == 4 or month == 6 or month == 9 or month == 11 else 0
            days += 29 if month == 2 and years_since_2024 % 4 == 0 else 29 if month == 2 else 0
        return seconds + minutes*60 + hours*60*60 + days*24*60*60
    except:
        # print(f'Invalid input for Yeamod Homis Function: {YeaMoD_HoMiS}')
        print(f'Invalid input for Yeamod Homis Function.')

def unpack_data(tuple_input):
    list_input = list(tuple_input)
    list_input.reverse()
    return list_input

def limit_PEK_data(list_input):
    global left_PEK_lim, right_PEK_lim
    return list_input[left_PEK_lim:right_PEK_lim]

def limit_PRAISE_data():
    global left_PRAISE_lim, right_PRAISE_lim, start_times, end_times, DateTime, yeamod_start_times, yeamod_end_times, lats, lons
    yeamod_start_times = [YeaMoD_HoMiS_24_to_seconds(time) for time in start_times]
    yeamod_end_times = [YeaMoD_HoMiS_24_to_seconds(time) for time in end_times]
    start_second = YeaMoD_HoMiS_24_to_seconds(DateTime[0]) if len(DateTime) != 0 else YeaMoD_HoMiS_24_to_seconds(start_times[0])
    end_second = YeaMoD_HoMiS_24_to_seconds(DateTime[-1]) if len(DateTime) != 0 else YeaMoD_HoMiS_24_to_seconds(start_times[-1])
    left_PRAISE_lim = 0
    right_PRAISE_lim = len(start_times)-1
    for i, time in enumerate(yeamod_start_times):
        if time >= start_second and left_PRAISE_lim == 0:
            left_PRAISE_lim = i if abs(yeamod_start_times[i]-start_second) < abs(yeamod_start_times[i-1]-start_second) else i-1
        if time > end_second and right_PRAISE_lim == len(start_times)-1:
            right_PRAISE_lim = i if abs(yeamod_start_times[i]-end_second) < abs(yeamod_start_times[i-1]-end_second) else i-1
    start_times, end_times, lats, lons = map(apply_limits_PRAISE_data, [start_times, end_times, lats, lons])

def apply_limits_PRAISE_data(list_input):
    global left_PRAISE_lim, right_PRAISE_lim
    return list_input[left_PRAISE_lim:right_PRAISE_lim]

def coords_to_km(lat1, lon1, lat2, lon2):
    return 2*6371*asin(sqrt(sin(radians((lat1-lat2)/2))**2 + cos(radians(lat2)) * cos(radians(lat1)) * sin(radians((lon1-lon2)/2))**2))

def filter_outliers_PRAISE(angle_threshold=20):
    global PRAISE_outliers, lats, lons, start_times, end_times, yeamod_start_times, yeamod_end_times
    div_0_count = 1
    reached_angle_threshold_count = 1
    reached_velocity_threshold_count = 1
    while div_0_count > 0 or reached_angle_threshold_count > 0 or reached_velocity_threshold_count > 0: 
        PRAISE_outliers = []
        div_0_count = 0
        reached_angle_threshold_count = 0
        reached_velocity_threshold_count = 0
        for i, (x, y) in enumerate([*zip(lats,lons)]):
            if i not in PRAISE_outliers and i != 0 and i != len(lats)-1:
                a = coords_to_km(lats[i-1], lons[i-1], x, y)
                b = coords_to_km(x, y, lats[i+1], lons[i+1])
                c = coords_to_km(lats[i-1], lons[i-1], lats[i+1], lons[i+1])
                if a * b == 0:
                    div_0_count += 1
                    PRAISE_outliers.append(i)
                else:
                    eq = (a**2 + b**2 - c**2) / (2 * a * b)
                    eq = -1 if eq < -1 else 1 if eq > 1 else eq
                    if acos(eq)*180/pi <= angle_threshold:
                        # map_subplot.plot([lons[i-1],lons[i],lons[i+1]],[lats[i-1],lats[i],lats[i+1]], marker='o',markersize=6, alpha=.5,transform=ccrs.PlateCarree())
                        reached_angle_threshold_count += 1
                        PRAISE_outliers.append(i)
                if YeaMoD_HoMiS_24_to_seconds(start_times[i]) == YeaMoD_HoMiS_24_to_seconds(start_times[i-1]):
                    PRAISE_outliers.append(i-1)
                elif i not in PRAISE_outliers and a/(YeaMoD_HoMiS_24_to_seconds(start_times[i]) - YeaMoD_HoMiS_24_to_seconds(start_times[i-1])) > .03:
                    # print(a/(YeaMoD_HoMiS_24_to_seconds(end_times[i-1]) - YeaMoD_HoMiS_24_to_seconds(start_times[i-1])))
                    # print(f'{a=}        {i=}')
                    # print(f'{YeaMoD_HoMiS_24_to_seconds(end_times[i-1])=}       {YeaMoD_HoMiS_24_to_seconds(start_times[i-1])=}')
                    # print(f'{lats[i]=}      {lons[i]=}      {lats[i-1]=}    {lons[i-1]=}')
                    reached_velocity_threshold_count += 1
                    # map_subplot.plot([lons[i-1],lons[i]],[lats[i-1],lats[i]], marker='o',markersize=8,transform=ccrs.PlateCarree())
                    PRAISE_outliers.append(i)
        # print(f'{div_0_count=}      {reached_angle_threshold_count=}        {reached_velocity_threshold_count=}')
        start_times, end_times, lats, lons = map(remove_PRAISE_outliers, [start_times, end_times, lats, lons])

def remove_PRAISE_outliers(list_input):
    global PRAISE_outliers
    return [item for i, item in enumerate(list_input) if i not in PRAISE_outliers]

def simplify_data(chunk_size=6, skip_mod=1, variance_threshold=.008):
    global lats, lons
    simplified_points = 1
    low_variance_coords = []
    while simplified_points > 0:
        low_variance_indices = []
        simplified_points = 0
        for i, [loclats, loclons] in enumerate([*zip(chunk_list(lats, chunk_size, skip_mod), chunk_list(lons, chunk_size, skip_mod))]):
            variance, avg_lat, avg_lon = calculate_location_variance(loclats,loclons)
            if variance < variance_threshold:
                lats[i*skip_mod], lons[i*skip_mod] = avg_lat, avg_lon
                low_variance_coords.append([avg_lat,avg_lon])
                low_variance_indices.extend([*range(i*skip_mod+1,i*skip_mod+chunk_size)])
                simplified_points += 1
        lats = [lat for i, lat in enumerate(lats) if i not in low_variance_indices]
        lons = [lon for i, lon in enumerate(lons) if i not in low_variance_indices]
    
    # print(len(low_variance_coords))
    # for lat, lon in low_variance_coords:
    #     map_subplot.plot(lon, lat, marker='o', markersize=8, alpha=.6, transform=ccrs.PlateCarree(), color='blue')  
      
    distance_threshold = 0.1
    stationary_coords = []
    simplified_points = 1
    while simplified_points > 0:
        simplified_points = 0
        for i, [lat1, lon1] in enumerate(low_variance_coords[:-1]):
            # i -= simplified_points
            for ii, [lat2, lon2] in enumerate(low_variance_coords[i+1:]):
                ii += i+1
                # print(f'{i=}     {ii=}      {len(low_variance_coords)=}     {simplified_points=}')
                if ii >= len(low_variance_coords):
                    break
                if coords_to_km(lat1, lon1, lat2, lon2) < distance_threshold:
                    low_variance_coords[i] = [(lat1+lat2)/2, (lon1+lon2)/2]
                    del low_variance_coords[ii]
                    simplified_points += 1
        # print(f'{simplified_points=}')
    # print(len(stationary_coords))
    for lat, lon in low_variance_coords:
        map_subplot.plot(lon, lat, marker='o', markersize=8, alpha=.6, transform=ccrs.PlateCarree(), color='black')    
    return low_variance_coords

def avg(list_input):
    return sum(list_input)/len(list_input)

def calculate_location_variance(loclats, loclons):
    avg_lat = avg(loclats)
    avg_lon = avg(loclons)
    distances = []
    for lat, lon in [*zip(loclats,loclons)]:
        distances.append(coords_to_km(avg_lat, avg_lon, lat, lon))
    avg_distance = avg(distances)
    variance = sum([(d-avg_distance)**2 for d in distances])/(len(distances)-1)
    return variance, avg_lat, avg_lon

def chunk_list(list_input, chunk_size, skip_mod=1):
    chunks = []
    for i, _ in enumerate(list_input[:-chunk_size]):
        if i % skip_mod == 0:
            chunks.append(list_input[i:i+chunk_size])
    return chunks

def textBox_xticks_submitted(inputted_xticks):
    global xticks
    xticks = int(inputted_xticks)
    update_data_subplot()

def textBox_date_start_limit_submitted(date):
    global start_date_limit
    start_date_limit = str(date)
    update_all_subplots()

def textBox_date_end_limit_submitted(date):
    global end_date_limit
    end_date_limit = str(date)
    update_all_subplots()

def button_paste_x_date_end_limit_clicked(event):
    global DateTime, clicked_x_coord
    textBox_date_end_limit.set_val(DateTime[int(clicked_x_coord)])
    update_all_subplots()
    
def button_paste_x_date_start_limit_clicked(event):
    global DateTime, clicked_x_coord
    textBox_date_start_limit.set_val(DateTime[int(clicked_x_coord)])
    update_all_subplots()
    clicked_x_coord = 0

def update_label_texts(event=0):
    global textBox_date_start_limit_label, textBox_date_start_limit_label, textBox_xticks_label
    textBox_date_start_limit_label.set_text(f'start ({start_index_limit}): yyyy-mm-dd hh:mm:ss')
    textBox_date_end_limit_label.set_text(f'end ({end_index_limit}): yyyy-mm-dd hh:mm:ss')
    textBox_xticks_label.set_text(f'xTicks (max {plt.xlim()[1]})')

def button_generate_map_procedurally_clicked(event=0):
    global generate_map_procedurally
    generate_map_procedurally = True
    update_map_subplot()

def animate_map(i):
    global lons, lats
    # print(f'{i}/{len(lons)}')
    # print([lons[i], lats[i]])
    plt.sca(data_subplot)
    # if i % 3 == 0:
    #     data_subplot.axvline(plt.xlim()[1]/len(lats)*i)
    map_subplot.plot(lons[i],lats[i],marker='o',markersize=2,transform=ccrs.PlateCarree(), color='red')

def update_data_subplot(event=0):
    global left_PEK_lim, right_PEK_lim, start_date_limit, end_date_limit, element_data, DateTime, clicked_x_coord, xticks
    data_subplot.cla()
    for PEK in PEKs:
        device_name = f'dev9507E2103000{PEK}'
        pek_data_sheet = read_excel('export.xlsx',sheet_name=f'{device_name}') # read excel sheet
        formatted_pek_data_sheet = list(zip(*pek_data_sheet.values.tolist()[::-1])) # set sheet to list, then rotate it clockwise
        if len(formatted_pek_data_sheet) == 14:
            DateTime, Status, PM2_5, PM10, PM1, CO, NO2, O3, VOC, Humidity, Temperature, BatVol, CO2, NO2_Raw = map(unpack_data, formatted_pek_data_sheet) # unpack data into lists instead of tuples
            left_PEK_lim = DateTime.index(start_date_limit) if start_date_limit in DateTime else 0
            right_PEK_lim = DateTime.index(end_date_limit) if end_date_limit in DateTime else len(DateTime)-1
            DateTime, Status, PM2_5, PM10, PM1, CO, NO2, O3, VOC, Humidity, Temperature, BatVol, CO2, NO2_Raw = map(limit_PEK_data, [DateTime, Status, PM2_5, PM10, PM1, CO, NO2, O3, VOC, Humidity, Temperature, BatVol, CO2, NO2_Raw])
        elif len(formatted_pek_data_sheet) == 12: 
            DateTime, Status, PM2_5, PM10, PM1, CO, NO2, O3, Humidity, Temperature, CO2, NO2_Raw = map(unpack_data, formatted_pek_data_sheet) # unpack data into lists instead of tuples
            left_PEK_lim = DateTime.index(start_date_limit) if start_date_limit in DateTime else 0
            right_PEK_lim = DateTime.index(end_date_limit) if end_date_limit in DateTime else len(DateTime)-1
            DateTime, Status, PM2_5, PM10, PM1, CO, NO2, O3, Humidity, Temperature, CO2, NO2_Raw = map(limit_PEK_data, [DateTime, Status, PM2_5, PM10, PM1, CO, NO2, O3, Humidity, Temperature, CO2, NO2_Raw])
        else:
            print(f'{len(formatted_pek_data_sheet)=}')
        # Note that incoming data is formatted as follows:
        # PM2.5, PM10, and PM1 are in micrograms/m^3            CO NO2, O3, and VOC are in parts per billion
        # Huumidity is in %         Temperature is in C         BatVol is in V
        # CO2 is in parts per million                           NO2_Raw is in unknown format

        for element in elements: # display all elements
            try:
                exec(f'globals()["element_data"] = {element}')
                data_subplot.plot(DateTime, element_data, label=f'{element} | PEK {PEK}')
            except:
                print('exception in element display')
    update_label_texts()
    
    data_subplot.axvline(clicked_x_coord,linewidth=1)
    plt.sca(data_subplot)
    try:
        plt.xticks(np.arange(0,plt.xlim()[1], int(plt.xlim()[1]/xticks)))
    except:
        print(f'Invalid amount of xticks with the x limit being {plt.xlim()[1]} and the variable being {xticks}')
    plt.xticks(rotation=45) # set tick labels to be rotated so that they're visible
    data_subplot.grid(True) # show grid
    data_subplot.legend()# display legend
    plt.draw()

# def start_thread_update_data_subplot(event=0):
#     thread_update_data_subplot = Thread(target=update_data_subplot)
#     thread_update_data_subplot.start()

map_anim = c_animation(fig, animate_map, 200)
def update_map_subplot(event=0):
    global left_PRAISE_lim, right_PRAISE_lim, lats, lons, generate_map_procedurally, map_anim, PRAISE_invalid_indices, start_times, end_times
    map_subplot.cla()
    # setup map_subplot
    # map_subplot.set_extent([114.060070, 114.350106, 22.167814, 22.358208], crs=ccrs.PlateCarree()) # sets bottom left and upper right corner of map (lon1,lon2,lat1,lat2)
    # map_subplot.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=.5) 
    # map_subplot.add_feature(cfeature.LAKES,  alpha=0.5)
    # map_subplot.add_feature(cfeature.LAND)
    # map_subplot.add_feature(cfeature.COASTLINE, edgecolor='yellow', linewidth=.5)
    # map_subplot.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.5)
    # states_provinces = cfeature.NaturalEarthFeature(
    #             category='cultural',  name='admin_1_states_provinces',
    #             scale='10m', facecolor='none')
    # map_subplot.add_feature(states_provinces, edgecolor='black', zorder=10, linestyle = '-', linewidth=0.5)
    map_subplot.imshow(mpimg.imread('satellite_view.png'), extent=[114.060070, 114.350106, 22.167814, 22.358208])

    map_subplot.gridlines(draw_labels=True) # enable labels
    start_times, end_times, lats, lons, os, phone_model, remarks, location_name, location_zh_name, exposure, environment, micro_environment = map(unpack_data,formatted_sheet)
    
    limit_PRAISE_data()
    filter_outliers_PRAISE()
    simplify_data()
    
    lats, lons = map(np.array, [lats,lons])
    plt.sca(map_subplot)
    if generate_map_procedurally == True:
        map_anim.start()
        generate_map_procedurally = False
    else:
        map_anim.stop()
        map_subplot.plot(lons, lats, marker='o', markersize=1, transform=ccrs.PlateCarree(), color = 'red')
    map_subplot.plot(114.18156524344481, 22.250541509340582, marker='o', markersize=12, alpha=.3, transform=ccrs.PlateCarree(), color='blue') #house
    map_subplot.plot(114.2635637688688, 22.338068845044553, marker='o',markersize=12,alpha=.3,transform=ccrs.PlateCarree(), color='blue') #HKUST

def update_all_subplots(event=0):
    update_data_subplot()
    update_map_subplot()

def on_mouse_click(event):
    global clicked_x_coord, clicked_y_coord
    if event.inaxes == data_subplot:
        clicked_x_coord = event.xdata
        clicked_y_coord = event.ydata
        update_data_subplot()

all_PEKs = [x[-2:] for x in ExcelFile('export.xlsx').sheet_names] # take sheet names and take the last two numbers from each to get available PEKs
unuseable_PEKs = []
for PEK in all_PEKs:
    device_name = f'dev9507E2103000{PEK}'
    sheet = read_excel('export.xlsx',sheet_name=f'{device_name}') # read excel sheet
    number_of_elements = len(sheet.values[0])
    if number_of_elements != 14:
        # unuseable_PEKs.append(PEK)
        print(f'PEK {PEK} is unuseable as it has {number_of_elements} elements instead of 14')
available_PEKs = sorted([PEK for PEK in all_PEKs if PEK not in unuseable_PEKs])

for i, PEK in enumerate(available_PEKs):
    exec(f'toggle_value_{PEK} = 0\n'
         f'def toggle_{PEK}(event):\n'
         f'    global toggle_value_{PEK}\n'
         f'    toggle_value_{PEK} = 1 if toggle_value_{PEK} == 0 else 0\n'
         f'    if toggle_value_{PEK}:\n'
         f'        PEKs.append(\'{PEK}\')\n'
         f'    else:\n'
         f'        PEKs.remove(\'{PEK}\')\n'
         f'    display_toggle_{PEK}.set_text(toggle_value_{PEK})\n'
         f'    update_all_subplots()\n'
         f'plot_{PEK} = fig.add_axes([.98-button_xdim, 0.02+(button_ydim+0.01)*i, button_xdim, button_ydim])\n'
         f'button_{PEK} = Button(plot_{PEK}, \'PEK {PEK}\')\n'
         f'button_{PEK}.on_clicked(toggle_{PEK})\n'
         f'display_toggle_{PEK} = fig.text(.98-button_xdim-button_ydim/2, 0.02+(button_ydim+0.01)*i+button_ydim/2, \'0\')\n')

for i, element in enumerate(['PM2_5', 'PM10', 'PM1', 'CO', 'NO2', 'O3', 'VOC', 'Humidity', 'Temperature', 'BatVol', 'CO2', 'NO2_Raw']):
    exec(f'toggle_{element}_value = 0\n'
         f'def toggle_{element}(event):\n'
         f'    global toggle_{element}_value\n'
         f'    toggle_{element}_value = 1 if toggle_{element}_value == 0 else 0\n'
         f'    if toggle_{element}_value:\n'
         f'        elements.append(\'{element}\')\n'
         f'    else:\n'
         f'        elements.remove(\'{element}\')\n'
         f'    display_toggle_{element}.set_text(toggle_{element}_value)\n'
         f'    update_data_subplot()\n'
         f'plot_{element} = fig.add_axes([.02, 0.02+(button_ydim+0.01)*i, button_xdim, button_ydim])\n'
         f'button_{element} = Button(plot_{element}, \'{element}\')\n'
         f'button_{element}.on_clicked(toggle_{element})\n'
         f'display_toggle_{element} = fig.text(.02-button_ydim/2, 0.02+(button_ydim+0.01)*i+button_ydim/2, \'0\')\n')

textBox_labels = []

plot_xticks = fig.add_axes([0.08, 0.02, .05, .03])
textBox_xticks = TextBox(plot_xticks,f'xTicks (max {plt.xlim()[1]})')
textBox_xticks.on_submit(textBox_xticks_submitted)
textBox_xticks_label = textBox_xticks.ax.get_children()[0]
textBox_labels.append(textBox_xticks_label)

plot_date_start_limit = fig.add_axes([0.08,0.4,0.1,0.03])
textBox_date_start_limit = TextBox(plot_date_start_limit,f'start ({start_index_limit}): yyyy-mm-dd hh:mm:ss')
textBox_date_start_limit.on_submit(textBox_date_start_limit_submitted)
textBox_date_start_limit_position = textBox_date_start_limit.ax.get_position()
textBox_date_start_limit_label = textBox_date_start_limit.ax.get_children()[0]
textBox_labels.append(textBox_date_start_limit_label)

plot_date_end_limit = fig.add_axes([0.2,0.4,0.1,0.03])
textBox_date_end_limit = TextBox(plot_date_end_limit,f'end ({end_index_limit}): yyyy-mm-dd hh:mm:ss')
textBox_date_end_limit.on_submit(textBox_date_end_limit_submitted)
textBox_date_end_limit_position = textBox_date_end_limit.ax.get_position()
textBox_date_end_limit_label = textBox_date_end_limit.ax.get_children()[0]
textBox_labels.append(textBox_date_end_limit_label)

plot_paste_x_date_end_limit = fig.add_axes([textBox_date_end_limit_position.x0+.02,textBox_date_end_limit_position.y0-.06,
                                               textBox_date_end_limit_position.x1-textBox_date_end_limit_position.x0-.04,
                                               textBox_date_end_limit_position.y1-textBox_date_end_limit_position.y0])
button_paste_x_date_end_limit = Button(plot_paste_x_date_end_limit, 'Paste Clicked X')
button_paste_x_date_end_limit.on_clicked(button_paste_x_date_end_limit_clicked)

plot_paste_x_date_start_limit = fig.add_axes([textBox_date_start_limit_position.x0+.02,textBox_date_start_limit_position.y0-.06,
                                               textBox_date_start_limit_position.x1-textBox_date_start_limit_position.x0-.04,
                                               textBox_date_start_limit_position.y1-textBox_date_start_limit_position.y0])
button_paste_x_date_start_limit = Button(plot_paste_x_date_start_limit, 'Paste Clicked X')
button_paste_x_date_start_limit.on_clicked(button_paste_x_date_start_limit_clicked)

plot_export_data = fig.add_axes([0.98-button_xdim,0.45,button_xdim,button_ydim])
button_export_data = Button(plot_export_data, 'Export PEK\nData (takes time)')
button_export_data.on_clicked(export_PEK_data)

plot_force_update = fig.add_axes([0.98-button_xdim,0.4,button_xdim,button_ydim])
button_force_update = Button(plot_force_update, 'Force Update\nGraphs')
button_force_update.on_clicked(update_all_subplots)

plot_generate_map_procedurally = fig.add_axes([0.7,0.1,button_xdim,button_ydim])
button_generate_map_procedurally = Button(plot_generate_map_procedurally, 'Generate Map')
button_generate_map_procedurally.on_clicked(button_generate_map_procedurally_clicked)

clicked_coords = fig.canvas.mpl_connect('button_press_event', on_mouse_click)

for label in textBox_labels:
    label.set_position([0.5,1.4])
    label.set_verticalalignment('top')
    label.set_horizontalalignment('center')

sheet = read_csv('2024_July.csv')
formatted_sheet = list(zip(*sheet.values.tolist()[::-1])) # set sheet to list, then rotate it clockwise

update_map_subplot()

fig.tight_layout() # makes graphs larger and tighter together
print(plt.get_backend())
# plt.get_current_fig_manager().window.state('zoomed') # set to fullscreen automatically
print(f'Loaded in {time.time()-ST_loading} seconds')
plt.show()