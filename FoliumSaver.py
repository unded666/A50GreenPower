import folium
import constants
from selenium import webdriver
from pathlib import Path
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pdfkit

import time

def folium_saver(map: folium.Map, temp_file_path=constants.TEMP_MAP_HTML, save_file_path = constants.TEMP_MAP_FILE):
    """
    Save a folium map to a file
    :param folium.Map: folium map object
    :param file_path: file path to save the folium map
    :return: None
    """
    map.save(temp_file_path)

    # Convert the relative path to an absolute path, then to a URL
    file_url = Path(temp_file_path).absolute().as_uri()

    driver = webdriver.Chrome()
    driver.get(file_url)

    time.sleep(1)

    driver.save_screenshot(save_file_path)

    driver.quit()

if __name__ == '__main__':
    m = folium.Map(location=constants.RSA_LOCATION, zoom_start=6)
    folium_saver(m)
    print('Done')


