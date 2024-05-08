import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from DataReader import DataCentreWrangler, DataReader, convert_to_decimal_degrees
from ImageManipulation import ImageManipulation
import rasterio

SOLAR_DIR = './Data/Global Solar Atlas/'
SOLAR_FILE = SOLAR_DIR + 'PVOUT_Yearly_sum.tif'
DATA_CENTRE_FILE = './Data/Data centres - preliminary information.xlsx'
DEGREEE_TO_METER = 111139
SOLAR_EFFICIENCY = 0.2
YSUM_DIR = './Data/Global Solar Atlas/YearlySum/'
YSUM_GHI = YSUM_DIR + 'GHI.tif'
YSUM_DNI = YSUM_DIR + 'DNI.tif'
YSUM_DIF = YSUM_DIR + 'DIF.tif'


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
    centre_frame_copy['Excess Land Required'] = (centre_frame_copy['Expected Total Land Required'] -
                                                centre_frame_copy['Total Available Land'])

    # Drop the 'EnergyPerSqM' column
    centre_frame_copy.drop('EnergyPerSqM', axis=1, inplace=True)

    return centre_frame_copy


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
    data_centre_wrangler = DataCentreWrangler(DATA_CENTRE_FILE)
    data_centre_wrangler.wrangle()
    data_center_df = data_centre_wrangler.df.copy()
    reader_solar_data = DataReader(YSUM_GHI)
    solar_image, solar_transform, solar_src = reader_solar_data.read_tiff()
    # manip = ImageManipulation(save_location='./WorkingData/Maps/')
    # x_size, y_size, pixel_sqm = manip.get_pixel_size(reader_solar_data.file_path)
    # solar_enriched_df = manip.get_PV_from_tiff(solar_src, solar_image, data_centre_wrangler.df)
    data_center_df = ImageManipulation().get_PV_from_tiff(src=solar_src,
                                                          img=solar_image,
                                                          data_centre_frame=data_center_df)
    data_center_df = calculate_required_land(data_center_df)