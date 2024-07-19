import time, logging, cartopy, matplotlib, warnings, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button, TextBox
from math import acos, asin, cos, sin, dist, pi, degrees, sqrt, radians

# disable annoying warnings
warnings.simplefilter('ignore')

# default variables
logging_level = logging.DEBUG
button_xdim, button_ydim = 0.045, 0.03
toggled_PEKs = ['35', '42']
toggled_elements = ['CO2']
xticks = 30
clicked_x_coord = 0
start_date_limit_second_time, end_date_limit_second_time = 0, 0

# setup logging
os.remove('logs/recentlog.log')
logger = logging.getLogger('PRAISE-PEK')
logger.setLevel(logging_level) 

formatter = logging.Formatter('[%(levelname)s][%(asctime)s] %(name)s: %(message)s') # format log prefixes
file_handler = logging.FileHandler('logs/recentlog.log')
file_handler.setLevel(logging_level)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

##### WIDGET FUNCTIONS #####
def xticks_submitted(inputted_xticks):
    global xticks
    xticks = int(inputted_xticks)
    logger.debug(f'Xticks textbox submitted')
    update_data_subplot()

def start_date_limit_submitted(date):
    global start_date_limit_second_time
    start_date_limit_second_time = DT_to_seconds(date)
    logger.debug(f'Start date limit textbox submitted')
    update_all()
    
def end_date_limit_submitted(date):
    global end_date_limit_second_time
    end_date_limit_second_time = DT_to_seconds(date)
    logger.debug(f'End date limit textbox submitted')
    update_all()

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

##### FUNCTIONS #####
def DT_to_seconds(DT, split_chars=['-',':']): # what I call the YeaMoD HoMiS function
    YeaMoD, HoMiS = DT.split(' ')
    years, months, days = map(int, YeaMoD.split(split_chars[0]))
    hours, minutes, seconds = map(int, HoMiS.split(split_chars[1]))
    years_since_2024 = years-2024
    for month in range(1,int(months)):
        days += 31 if month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12 else 0
        days += 30 if month == 4 or month == 6 or month == 9 or month == 11 else 0
        days += 29 if month == 2 and years_since_2024 % 4 == 0 else 29 if month == 2 else 0
    return seconds + minutes*60 + hours*60*60 + days*24*60*60

def coords_to_km(lat1, lon1, lat2, lon2):
    return 2*6371*asin(sqrt(sin(radians((lat1-lat2)/2))**2 + cos(radians(lat2)) * cos(radians(lat1)) * sin(radians((lon1-lon2)/2))**2))

def unpack_PEK_data():
    ST_unpack_PEK_data = time.time()
    for sheet_name in pd.ExcelFile('data/export.xlsx').sheet_names:
        ST_read_sheet = time.time()
        PEK_number = sheet_name[-2:]
        PEK_sheet = pd.read_excel('data/export.xlsx', sheet_name=sheet_name)
        PEK_data = list(zip(*PEK_sheet.values.tolist()))
        PEK_columns = list(PEK_sheet.columns)

        PEK_data.append(list(map(DT_to_seconds, PEK_data[list(PEK_columns).index('DateTime')])))
        PEK_columns.append('DT_seconds')
        for ii, column in enumerate(PEK_columns):
            element_name = column.split(' ')[0].replace('.','_')
            exec(f'globals()["PEK_{PEK_number}_{element_name}"] = PEK_data[ii]')
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
    
def remove_indicies_from_lists(lists, indices):
    return [[val for i, val in enumerate(list) if i not in indices] for list in lists]

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
                a = coords_to_km(PRAISE_lats[i-1], PRAISE_lons[i-1], x, y)
                b = coords_to_km(x, y, PRAISE_lats[i+1], PRAISE_lons[i+1])
                c = coords_to_km(PRAISE_lats[i-1], PRAISE_lons[i-1], PRAISE_lats[i+1], PRAISE_lons[i+1])
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
        PRAISE_start_dates, PRAISE_end_dates, PRAISE_start_seconds, PRAISE_end_seconds, PRAISE_lats, PRAISE_lons = remove_indicies_from_lists([PRAISE_start_dates, PRAISE_end_dates, PRAISE_start_seconds, PRAISE_end_seconds, PRAISE_lats, PRAISE_lons], PRAISE_outliers)
    logger.info(f'Cleaned PRAISE data in {time.time()-ST_clean_PRAISE_data} seconds and removed {total_indicies_removed}/{total_indices} indices')

def update_data_subplot():
    ST_update_data_subplot = time.time()
    data_subplot.cla() # clear previous subplot data
    global total_PEK_second_times, total_PEK_date_times, element_data, seconds_times, date_times

    # setup arrays for finding xticks later 
    total_PEK_second_times, total_PEK_date_times = [], []
    for i, PEK in enumerate(toggled_PEKs):
        for ii, element in enumerate(toggled_elements): # iterate through each element for each PEK toggled
            try:
                # pull data
                exec(f'globals()["element_data"] = PEK_{PEK}_{element}')
                exec(f'globals()["seconds_times"] = list(PEK_{PEK}_DT_seconds)')
                exec(f'globals()["date_times"] = list(PEK_{PEK}_DateTime)')
                sd_index = seconds_times.index(start_date_limit_second_time) if start_date_limit_second_time != 0 else 0
                ed_index = seconds_times.index(end_date_limit_second_time) if end_date_limit_second_time != 0 else len(seconds_times)-1
                element_data, seconds_times, date_times = element_data[sd_index:ed_index], seconds_times[sd_index:ed_index], date_times[sd_index:ed_index]
                
                # plot data
                data_subplot.plot(seconds_times, element_data, label=f'{element} | PEK {PEK}') 

                # append values if they are not already present, then sort them relative to one another
                total_PEK_second_times += [item for item in seconds_times if item not in total_PEK_second_times]
                total_PEK_date_times += [item for item in date_times if item not in total_PEK_date_times]
                total_PEK_second_times, total_PEK_date_times = [list(x) for x in zip(*sorted(zip(total_PEK_second_times, total_PEK_date_times)))]
            except:
                logger.warning(f'PEK_{PEK}_{element} does not exist or other error occurred when toggling elements')

    # set label
    xticks_textbox.label_obj.set_text(f'xTicks (max {len(total_PEK_second_times)})')

    # draw line where last clicked
    data_subplot.axvline(min(total_PEK_second_times, key=lambda x:abs(x-clicked_x_coord)), linewidth=1)

    # setup xticks
    plt.sca(data_subplot)
    tick_locations = total_PEK_second_times[::int(len(total_PEK_second_times)/xticks)] # find xticks locations
    labels = total_PEK_date_times[::int(len(total_PEK_second_times)/xticks)] # find xticks labels
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

    ss_index, es_index = PRAISE_start_seconds.index(min(PRAISE_start_seconds, key=lambda x:abs(x-start_date_limit_second_time))), PRAISE_end_seconds.index(min(PRAISE_end_seconds, key=lambda x:abs(x-end_date_limit_second_time)))
    print(f'{ss_index=}     {es_index=}')
    lons = PRAISE_lons[ss_index:es_index]
    lats = PRAISE_lats[ss_index:es_index]
    map_subplot.plot(lons, lats, marker='o', markersize=1, transform=proj, color = 'red') 
    logger.debug(f'Map subplot updated in {time.time()-ST_update_map_subplot} seconds')

def update_all(event=None):
    update_map_subplot()
    update_data_subplot()

# setup figure, plots, parameters, buttons, textboxes, etc
def setup_plots(**kwargs):
    ST_loading_plot = time.time() # record start time
    plot_loaded = False
    global fig, proj, data_subplot, map_subplot, clicked_x_coord, start_date_limit_textbox, end_date_limit_textbox, xticks_textbox, force_update_plots_button, export_PEK_data_button, paste_x_start_limit, paste_x_end_limit

    logger.debug(f'Current backend is {plt.get_backend()}') # log backend

    # setup figure
    fig = plt.figure()

    # pop kwargs
    font_size = kwargs.pop('font_size', 6)
    line_width = kwargs.pop('line_width', .4)
    tight_layout_enabled = kwargs.pop('tight_layout_enabled', True)
    auto_fullscreen = kwargs.pop('auto_fullscreen', False)

    # set matplotlib params
    matplotlib.rcParams['font.size'] = font_size
    matplotlib.rcParams['lines.linewidth'] = line_width
    fig.tight_layout() if tight_layout_enabled == True else 1
    plt.get_current_fig_manager().window.state('zoomed') if auto_fullscreen == True else 0

    # setup subplots
    data_subplot = plt.subplot(2,1,1) # sets the data subplot to the one on the top
    proj = cartopy.crs.PlateCarree() # set type of projecetion
    map_subplot = plt.subplot(2,1,2, projection=proj) # sets the map subplot to the one on the bottom

    # create classes to handle making buttons and textboxes easier
    class create_button:
        def __init__(self, label, location, clicked_function, **kwargs):
            self.ax = fig.add_axes(location)
            self.button = Button(self.ax, label)
            self.button.on_clicked(clicked_function)
            self.pos = self.button.ax.get_position()
            self.button.color = kwargs.pop('color', 'lightgrey')

            logger.debug(f'Button "{label}" successfully created with {location=}, {clicked_function=}')

        def set_color(self, color):
            self.button.color = color

    class create_textbox:
        def __init__(self, label, location, submit_function, label_pos='top'):
            self.ax = fig.add_axes(location)
            self.textbox = TextBox(self.ax, label)
            self.textbox.on_submit(submit_function)
            self.label_obj = self.textbox.ax.get_children()[0]
            self.pos = self.textbox.ax.get_position()

            if label_pos == 'top':
                self.label_obj.set_position([0.5,1.4])
                self.label_obj.set_verticalalignment('top')
                self.label_obj.set_horizontalalignment('center')
            
            logger.debug(f'TextBox "{label}" successfully created with {location=}, {submit_function=}, {label_pos=}')

    # setup buttons and textboxes
    xticks_textbox = create_textbox(f'xTicks (max {plt.xlim()[1]})', [0.08, 0.02, .05, .03], xticks_submitted)
    start_date_limit_textbox = create_textbox(f'start (): yyyy-mm-dd hh:mm:ss', [0.08,0.4,0.1,0.03], start_date_limit_submitted)
    end_date_limit_textbox = create_textbox(f'end (): yyyy-mm-dd hh:mm:ss', [0.2,0.4,0.1,0.03], end_date_limit_submitted)
    
    # generate_map_procedurally_button = create_button('Generate Map Procedurally', [0.7,0.1,button_xdim,button_ydim], dummy)
    force_update_plots_button = create_button('Update All', [0.98-button_xdim,0.4,button_xdim,button_ydim], update_all)
    export_PEK_data_button = create_button('Export PEK Data', [0.98-button_xdim,0.45,button_xdim,button_ydim], update_all)
    paste_x_start_limit = create_button('Paste Clicked X', [start_date_limit_textbox.pos.x0+.02,start_date_limit_textbox.pos.y0-.06,
                                               start_date_limit_textbox.pos.x1-start_date_limit_textbox.pos.x0-.04,
                                               start_date_limit_textbox.pos.y1-start_date_limit_textbox.pos.y0], paste_x_start_limit_pressed)
    paste_x_end_limit = create_button('Paste Clicked X', [end_date_limit_textbox.pos.x0+.02,end_date_limit_textbox.pos.y0-.06,
                                               end_date_limit_textbox.pos.x1-end_date_limit_textbox.pos.x0-.04,
                                               end_date_limit_textbox.pos.y1-end_date_limit_textbox.pos.y0], paste_x_end_limit_pressed)

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
            update_map_subplot() if plot_loaded == True else None

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
            update_map_subplot() if plot_loaded == True else None
    
    # setup PEK buttons
    for i, sheet_name in enumerate(pd.ExcelFile('data/export.xlsx').sheet_names):
        PEK_number = sheet_name[-2:]
        exec(f'globals()["PEK_{PEK_number}_button"] = create_PEK_button(PEK_number, [.98-button_xdim, 0.02+(button_ydim+0.01)*i, button_xdim, button_ydim])')

    # setup element buttons
    for i, element in enumerate(['PM2_5', 'PM10', 'PM1', 'CO', 'NO2', 'O3', 'VOC', 'Humidity', 'Temperature', 'BatVol', 'CO2', 'NO2_Raw']):
        exec(f'globals()["element_{element}_button"] = create_element_button(element, [.02, 0.02+(button_ydim+0.01)*i, button_xdim, button_ydim])')

    # set mouse click event
    def mouse_click_event(event):
        global clicked_x_coord, clicked_y_coord
        if event.inaxes == data_subplot:
            clicked_x_coord = event.xdata
            clicked_y_coord = event.ydata
            logger.debug(f'Data subplot clicked at x={clicked_x_coord}, y={clicked_y_coord}')
            update_data_subplot()
            
    clicked_coords = fig.canvas.mpl_connect('button_press_event', mouse_click_event)
    
    plot_loaded = True
    logger.info(f'Setup plot in {time.time()-ST_loading_plot} seconds') # log loading time in seconds

# main code run
if __name__ == '__main__':
    logger.info('--------------START--------------') # mark code init
    ST_main = time.time()

    # data management
    unpack_PEK_data()
    unpack_PRAISE_data()
    clean_PRAISE_data()
    # create_spreadsheet_praise_specifics('quit')
    
    # visual setup
    setup_plots()
    update_all()
    logger.info(f'Successfully initiated program in {time.time()-ST_main} seconds')
    plt.show()

    logger.info('---------------END---------------') # mark code deinit