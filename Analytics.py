import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from DataReader import DataCentreWrangler, DataReader, convert_to_decimal_degrees
from ImageManipulation import ImageManipulation
import rasterio
from constants import (SOLAR_FILE,
                       DATA_CENTRE_FILE,
                       SOLAR_EFFICIENCY,
                       YSUM_GHI,
                       YSUM_DNI,
                       YSUM_DIF,
                       DEGREEE_TO_METER,
                       PROVINCE_LOCATION_FILE,
                       POP_FIL,
                       PROP_FILE)

# SOLAR_DIR = './Data/Global Solar Atlas/'
# SOLAR_FILE = SOLAR_DIR + 'PVOUT_Yearly_sum.tif'
# DATA_CENTRE_FILE = './Data/Data centres - preliminary information.xlsx'
# DEGREEE_TO_METER = 111139
# SOLAR_EFFICIENCY = 0.2
# YSUM_DIR = './Data/Global Solar Atlas/YearlySum/'
# YSUM_GHI = YSUM_DIR + 'GHI.tif'
# YSUM_DNI = YSUM_DIR + 'DNI.tif'
# YSUM_DIF = YSUM_DIR + 'DIF.tif'


def collate_energy_data(solar_files: list) -> tuple[np.ndarray, np.ndarray, rasterio.io.DatasetReader]:
    """
    collates all of the energy data from the passed solar energy files. The energy file images
    are read in using the DataReader class, and then added together after being realigned to the
    same grid coordinates. The final image is then returned, along with the transform and source

    :param solar_files:
    :return:
    """

    # Read in the first image
    reader_solar_data = DataReader(solar_files[0])
    solar_image, solar_transform, solar_src = reader_solar_data.read_tiff()

    # Add the remaining images to the first image
    for file in solar_files[1:]:
        reader_solar_data = DataReader(file)
        solar_image_temp, solar_transform_temp, solar_src_temp = reader_solar_data.read_tiff()
        solar_image += solar_image_temp

    return solar_image, solar_transform, solar_src


def calculate_required_land(centre_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Expected Land Required by multiplying the PV output by the static conversion factor,
    which then gives an output of energy produced per square meter. This is used to determine how many
    square meters of land are required to meet the Annual Power Consumption of the data centre.

    :param centre_frame: data centre dataframe, containing a 'PV' column
    that contains the expected annual PV values, as well as 'Total Annual Power Consumption'.
    that gives how much power is required.

    :return: a dataframe with the 'Expected Total Land Required' column added
    """

    # Create a copy of the dataframe
    centre_frame_copy = centre_frame.copy()

    # Calculate the energy per square meter and the expected land required
    centre_frame_copy['EnergyPerSqM'] = centre_frame_copy['PV'] * SOLAR_EFFICIENCY
    centre_frame_copy['Expected Total Land Required'] = (centre_frame_copy['Total Annual Power Consumption'] /
                                                   centre_frame_copy['EnergyPerSqM'])
    # Replace NaNs in total available land with 0
    centre_frame_copy['Total Available Land'] = centre_frame_copy['Total Available Land'].fillna(0)
    centre_frame_copy['Excess Land Required'] = (centre_frame_copy['Expected Total Land Required'] -
                                                centre_frame_copy['Total Available Land'])

    # Drop the 'EnergyPerSqM' column
    centre_frame_copy.drop('EnergyPerSqM', axis=1, inplace=True)

    # Remove any negative values
    centre_frame_copy['Excess Land Required'] = centre_frame_copy['Excess Land Required'].apply(lambda x: 0 if x < 0 else x)

    return centre_frame_copy

def generate_land_price_image(province_file: str = PROVINCE_LOCATION_FILE,
                              pop_file: str = POP_FIL,
                              solar_reference = YSUM_GHI,
                              property_file: str = PROP_FILE) -> np.ndarray:
    """
    Generates a land price image using the province file and the population file. The land price is
    calculated by dividing the population by the area of the province, and then multiplying by the price
    of the land in the province. The land price is then returned as an image.

    :param province_file: the file containing the province data
    :param pop_file: the file containing the population data

    :return: a numpy array containing the land price image
    """

    # Read in image and get src
    solar_data = DataReader(solar_reference)
    PVimage, PVtransform, PVsrc = solar_data.read_tiff()

    # instantiate Image Manipulator object
    manipulator = ImageManipulation()

    # Get the base shape of the landmass
    landmass = manipulator.get_base_landmass_shape(PVimage)
    landmass_with_prices = manipulator.generate_land_prices(PVimage,
                                                            PVsrc,
                                                            pd.read_excel(property_file),
                                                            province_file)

    return landmass_with_prices


def main():
    # Read the data from the excel file
    print ('Reading the data from the excel file')
    data_centre_wrangler = DataCentreWrangler(DATA_CENTRE_FILE)
    data_centre_wrangler.wrangle()

    # Get the solar data from the data reader
    print('Getting the solar data from the data reader')
    reader_solar_data = DataReader(SOLAR_FILE)
    solar_image, solar_transform, solar_src = reader_solar_data.read_tiff()

    # Instantiate image manipulator
    manip = ImageManipulation(save_location='./WorkingData/Maps/')
    solar_enriched_df = manip.get_PV_from_tiff(solar_src, solar_image, data_centre_wrangler.df)



if __name__ == '__main__':
    # main()
    DC = DataCentreWrangler(DATA_CENTRE_FILE)
    DC.wrangle()
    dcf = DC.df
    img, _, src = DataReader(YSUM_GHI).read_tiff()
    lp_img = generate_land_price_image()
    ZB=(0, 1600, 2200, 0)
    # ZB = None
    ImageManipulation().project_data_centres_onto_map(dcf, src, img, zoombounds=ZB, savefile='./WorkingData/Maps/test.png')