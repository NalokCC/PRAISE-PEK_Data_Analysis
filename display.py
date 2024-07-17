import time, logging, cartopy, matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button, TextBox

# default variables
logging_level = logging.DEBUG
button_xdim, button_ydim = 0.045, 0.03

# setup logging
logger = logging.getLogger('PRAISE-PEK')
logger.setLevel(logging_level) 

formatter = logging.Formatter('[%(levelname)s][%(asctime)s] %(name)s: %(message)s') # format log prefixes
file_handler = logging.FileHandler('logs/recentlog.log')
file_handler.setLevel(logging_level)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# setup figure, pyplots, parameters, etc
def setup_plots(**kwargs):
    global fig, proj, data_subplot, map_subplot

    ST_loading_plot = time.time() # record start time
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

    # setup buttons and textboxes
    xticks_textbox = create_textbox(f'xTicks (max {plt.xlim()[1]})', [0.08, 0.02, .05, .03], dummy)
    start_date_limit_textbox = create_textbox(f'start (): yyyy-mm-dd hh:mm:ss', [0.08,0.4,0.1,0.03], dummy)
    end_date_limit_textbox = create_textbox(f'end (): yyyy-mm-dd hh:mm:ss', [0.2,0.4,0.1,0.03], dummy)
    
    generate_map_procedurally_button = create_button('Generate Map Procedurally', [0.7,0.1,button_xdim,button_ydim], dummy)
    force_update_plot_button = create_button('Force Update\nGraphs', [0.98-button_xdim,0.4,button_xdim,button_ydim], dummy)
    export_PEK_data_button = create_button('Export PEK Data', [0.98-button_xdim,0.45,button_xdim,button_ydim], dummy)
    paste_x_start_limit = create_button('Paste Clicked X', start_date_limit_textbox.pos, dummy)
    # paste_x_end_limit = create_button('Paste Clicked X', [end_date_limit_textbox.x0+.02,end_date_limit_textbox.y0-.06,
    #                                            end_date_limit_textbox.x1-end_date_limit_textbox.x0-.04,
    #                                            end_date_limit_textbox.y1-end_date_limit_textbox.y0], dummy)
    
    logger.info(f'Setup plot in {time.time()-ST_loading_plot} seconds') # log loading time in seconds

# handles creating a new axis and setting its location, label, and click function
class create_button:
    def __init__(self, label, location, clicked_function):
        self.ax = fig.add_axes(location)
        self.button = Button(self.ax, label)
        self.button.on_clicked(clicked_function)
        logger.debug(f'Button "{label}" successfully created with {location=}, {clicked_function=}')

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

def unpack_PEK_data():
    global PEK_sheet

def unpack_PRAISE_data():
    global PRAISE_start_times, PRAISE_end_times, PRAISE_lats, PRAISE_lons
    sheet = pd.read_csv('2024_July.csv')
    unpacked_data = list(zip(*sheet.values.tolist()[::-1]))
    unpacked_data = [list(element[::-1]) for element in unpacked_data]
    PRAISE_start_times, PRAISE_end_times, PRAISE_lats, PRAISE_lons, _, _, _, _, _, _, _, _ = unpacked_data

def update_map_subplot():
    map_subplot.cla() # clear previous map 

    map_subplot.imshow(mpimg.imread('satellite_view.png', extent=[114.060070, 114.350106, 22.167814, 22.358208])) # overlay satellite image
    map_subplot.gridlines(draw_labels=True)

    map_subplot.plot(PRAISE_lons, PRAISE_lats, marker='o', markersize=1, transform=proj, color = 'red')    

def dummy(event=1):
    print('pass')

if __name__ == '__main__':
    logger.info('--------------START--------------') # mark code init

    setup_plots()
    unpack_PRAISE_data()
    plt.show()

    logger.info('---------------END---------------') # mark code deinit