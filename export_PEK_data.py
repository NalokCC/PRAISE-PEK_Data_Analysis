from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import os
import time
import pathlib

######################################################################
# INCLUDE THE USERNAME AND PASSWORD HERE IN ORDER TO RUN THE PROGRAM #
######################################################################
SEIN_password, SEIN_username = '2021', 'PRAISE-PEK'

default_PEKs = '24,34,35,42'
PEK_list = default_PEKs.split(',')
hide_browser = ''

def export_PEK_data(PEKs=PEK_list):
    # for PEK in export_PEKs:
    #     PEK = '0' + PEK if len(PEK) == 1 else PEK
    print(f'PEKs being used are {PEK_list}')
    
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option('prefs', {'download.default_directory' : str(pathlib.Path().resolve())}) #set default download directory

    if list(globals()["hide_browser"]).count('n') == 0:
        chrome_options.add_argument('--headless') #no window will appear if user preses enter

    driver = webdriver.Chrome(options=chrome_options)
    driver.implicitly_wait(5) #set waittime to make sure page loads
    print('Setup default directory, driver, and implicit wait')

    driver.get('https://sapiens-iot.com/SEIN/')
    print('Website Loaded')

    driver.find_element(By.XPATH, '//*[@id="userName"]').send_keys(SEIN_username) #username
    driver.find_element(By.XPATH, '//*[@id="password"]').send_keys(SEIN_password) #password
    driver.find_element(By.XPATH, '/html/body/div[2]/div/div[2]/div/div[2]/div[4]/button').click() #press login button
    print('Logged in')

    time.sleep(10)
    deviceData_path = '/html/body/div/div[1]/div/div[2]/ul/li[3]/a' #set device data path
    print('Home Page Loaded')

    ActionChains(driver).move_to_element(driver.find_element(By.XPATH, deviceData_path)).perform() #perform hover 
    print('Hovered')
    driver.find_element(By.XPATH, '/html/body/div/div[1]/div/div[2]/ul/li[3]/ul/li[3]/a').click() #open data export screen
    print('Clicked Data Export Tab')

    warningClose_path = '//*[@id="warningClose"]' #set device data path

    time.sleep(1)
    try: #closes popup if it appears
        driver.find_element(By.XPATH, warningClose_path).click()
        print('Popup closed')
    except:
        print('No popup found')
    finally:
        print('Data Export Page Loaded')

    time.sleep(1)
    for PEK_number in PEK_list: #select PEKs
        print(f'Selecting PEK {PEK_number} with name dev9507E2103000{PEK_number}')
        try:
            driver.find_element(By.XPATH, f'//*[text()="dev9507E2103000{PEK_number}"]').click()
        except:
            print(f'[EXCEPTION]: No element with text \'dev9507E2103000{PEK_number}\' found')

    driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div/div[2]/div[2]/button').click() #Submit selected PEKs to be exported
    print('Selected PEKs to be exported')

    try: #move export file in current directory into archive if possible 
        os.replace('C:\\Users\\nolan\\Documents\\GitHub\\PRAISEHK\\export.xlsx', f'C:\\Users\\nolan\\Documents\\GitHub\\PRAISEHK\\archived_exports\\{time.time()}export.xlsx')
        print('Export file already present in directory has been moved to archived exports')
    except:
        print('No export file already current directory')

    t = 0
    while t < 40:
        try:
            driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div/div[2]/div[4]/div[1]/div[2]/div[1]').click()
            break
        except:
            t += 1
            time.sleep(0.5)
            pass
    print('Downloading spreadsheet')

    while True:
        if os.path.isfile('export.xlsx'):
            print('Speadsheet downloaded')
            break
        print('Waiting for export')
        time.sleep(2)

    time.sleep(2)

if __name__ == '__main__':
    PEK_list = input(f'Which PEKs would you like to update? [Enter] to use defaults ({default_PEKs}). Separate inputs with commas. \n')
    PEK_list = ''.join([s for s in list(PEK_list) if s in list('1234567890,')]) #remove letters and spaces
    PEK_list = default_PEKs if PEK_list == '' else PEK_list #set PEKs to default if no input is given
    PEK_list = PEK_list.split(',') #split into 
    
    hide_browser = input('Do you want to hide the browser? [Enter] to default yes, otherwise type \'n\'. \n')

    export_PEK_data(PEK_list)