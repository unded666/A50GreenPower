import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from DataReader import DataCentreWrangler, DataReader, convert_to_decimal_degrees
from ImageManipulation import ImageManipulation
import rasterio
import constants

# SOLAR_DIR = './Data/Global Solar Atlas/'
# SOLAR_FILE = SOLAR_DIR + 'PVOUT_Yearly_sum.tif'
# DATA_CENTRE_FILE = './Data/Data centres - preliminary information.xlsx'
# DEGREEE_TO_METER = 111139
# SOLAR_EFFICIENCY = 0.2
# YSUM_DIR = './Data/Global Solar Atlas/YearlySum/'
# YSUM_GHI = YSUM_DIR + 'GHI.tif'
# YSUM_DNI = YSUM_DIR + 'DNI.tif'
# YSUM_DIF = YSUM_DIR + 'DIF.tif'


def analyse_monthly_data(data_centre_wrangler: DataCentreWrangler,
                         monthly_file_dir: str = constants.MONTHLY_FILE_DIR) -> DataCentreWrangler:

    """

    :param monthly_file_dir:
    :param data_centre_frame:
    :return: wrangler whose dataframe has added mean and variance columns
    """

    monthly_files = DataReader(monthly_file_dir).get_all_tiffs(monthly_file_dir)
    monthly_readers = [DataReader(monthly_file) for monthly_file in monthly_files]
    long, lat, coords = data_centre_wrangler.get_coordinate_list()
    dc_df = data_centre_wrangler.df

    for month_i, reader_i in zip (constants.MONTHS, monthly_readers):
        img_i, _, src_i = reader_i.read_tiff()
        dc_df[month_i] = [ImageManipulation().get_pixel_value(lat_i, long_i, src_i, img_i) for lat_i, long_i in coords]

    data_centre_wrangler.df['mean'] = data_centre_wrangler.df[constants.MONTHS].mean(axis=1)
    data_centre_wrangler.df['std'] = data_centre_wrangler.df[constants.MONTHS].std(axis=1)

    return data_centre_wrangler

def translate_image_values_by_mapping_frame(image: np.ndarray,
                                            mapping_frame: pd.DataFrame,
                                            key_column: str,
                                            value_column: str) -> np.ndarray:

    """
    Translates the values in the image using the mapping frame. The mapping frame contains the key column
    and the value column, which are used to translate the values in the image. The key column contains the
    original values in the image, and the value column contains the values that the original values should
    be translated to. The function returns the image with the values translated.

    :param image: the image to be translated
    :param mapping_frame: dataframe containing the mapping values
    :param key_column: column name containing the original values
    :param value_column: column name containing the values to be translated to
    :return: the image with the values translated
    """

    # Create a copy of the image
    image_copy = image.copy()
    # create a mapping dictionary from the mapping frame
    mapping_dict = mapping_frame.set_index(key_column)[value_column].to_dict()
    # add a default value of zero to the dictionary to any unique values
    # in the image that are not in the mapping frame
    all_keys = list(mapping_dict.keys())
    for key in np.unique(image_copy):
        if key not in all_keys and np.isnan(key) == False:
            mapping_dict[int(key)] = 0

    # Translate the values in the image
    for key, value in mapping_dict.items():
        image_copy[image_copy == key] = value

    return image_copy

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
    centre_frame_copy['EnergyPerSqM'] = centre_frame_copy['PV'] * constants.SOLAR_EFFICIENCY
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

def generate_land_price_image(province_file: str = constants.PROVINCE_LOCATION_FILE,
                              pop_file: str = constants.POP_FIL,
                              solar_reference = constants.YSUM_GHI,
                              property_file: str = constants.PROP_FILE) -> np.ndarray:
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
    # landmass = manipulator.get_base_landmass_shape(PVimage)
    landmass_with_prices = manipulator.generate_land_prices(PVimage,
                                                            PVsrc,
                                                            pd.read_excel(property_file),
                                                            province_file)

    return landmass_with_prices


def get_best_X(dataframe: pd.DataFrame, X: int, column: str) -> pd.DataFrame:
    """
    Returns the top X rows of the dataframe, sorted by the specified column

    :param dataframe: the dataframe to be sorted
    :param X: the number of rows to return
    :param column: the column to sort by
    :return: the top X rows of the dataframe
    """

    return dataframe.sort_values(by=column, ascending=False).head(X)


def run_analysis(outfile: str = None) -> pd.DataFrame:
    """
    Runs the full set of analysis. Obtains the PV data from the solar file, and then calculates the expected
    land required for each data centre. The land price is then estimated, all returned in the output dataframe.
    If outfile is not None, then the resulting output file is saved to the specified location as an excel file.

    :param outfile: the location to save the output file to (if any)
    :return: the output dataframe
    """

    # Read the data from the excel file
    print('Reading the data from the excel file')
    data_centre_wrangler = DataCentreWrangler(constants.DATA_CENTRE_FILE)
    data_centre_wrangler.wrangle()
    data_centre_df = data_centre_wrangler.df

    # Get the solar data from the data reader
    print('Getting the solar data from the data reader')
    solar_image, solar_transform, solar_src = DataReader(constants.SOLAR_FILE).read_tiff()

    # get monthly solar data for the data centres
    monthly_wrangler = DataCentreWrangler(constants.DATA_CENTRE_FILE)
    monthly_wrangler.wrangle()
    monthly_wrangler = analyse_monthly_data(monthly_wrangler)
    monthly_df = monthly_wrangler.df
    data_centre_df['std'] = monthly_df['std']
    data_centre_df['mean'] = monthly_df['mean']

    bar_df = data_centre_df[['Site', 'mean', 'std']]
    bar_df.set_index('Site', inplace=True)
    bar_df.plot(kind='bar')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.title('Mean and Standard Deviation of Monthly Solar Energy by site')
    plt.ylabel('Energy (kWh/m^2)')
    plt.xticks(rotation=80, fontsize=6)
    plt.tight_layout()
    plt.savefig('./Data/Output_files/Maps/MonthlySolarEnergy.png', dpi=600)

    # Get the land use data from the data reader
    land_use_img, _, land_use_src = DataReader(constants.LAND_USE_FILE).read_tiff(reference_src=solar_src,
                                                                                  offset=[10, 30])
    # clean land use image by replacing zeroes with np.NaN
    land_use_img[land_use_img == 0] = np.NaN
    # get the preference data from the excel file
    preference_df = pd.read_excel(constants.LAND_PREF_FILE, sheet_name='Land Use', skiprows=2)



    PV_df = ImageManipulation().get_values_from_tiff(solar_src,
                                                   solar_image,
                                                   data_centre_df,
                                                   output_column='PV')

    # Calculate the expected land required
    print('Calculating the expected land required')
    expected_land_df = calculate_required_land(PV_df)

    # find the expected land price
    property_frame = pd.read_excel(constants.PROP_FILE)
    # landmass = ImageManipulation().get_base_landmass_shape(solar_image)
    landmass_with_prices = ImageManipulation().generate_land_prices(solar_image,
                                                                    solar_src,
                                                                    property_frame,
                                                                    constants.PROVINCE_LOCATION_FILE)

    # Add the land price to the expected land dataframe
    price_df = ImageManipulation().get_values_from_tiff(src=solar_src,
                                                        img=landmass_with_prices,
                                                        data_centre_frame=expected_land_df,
                                                        output_column='Land Price (R per Ha)')

    price_df['Total Land Price (R)'] = price_df['Expected Total Land Required'] * price_df['Land Price (R per Ha)']

    preference_img = translate_image_values_by_mapping_frame(image=land_use_img,
                                                             mapping_frame=preference_df,
                                                             key_column='#',
                                                             value_column='Implication (Weighting)')

    # plot the data centres on a backdrop of the land preferences
    ImageManipulation().project_data_centres_onto_map(data_centre_frame=price_df,
                                                      src=solar_src,
                                                      img=preference_img,
                                                      title='Preference by land use',
                                                      zoombounds=constants.ZOOM_BOUNDS,
                                                      savefile='./Data/Output_files/Maps/LandUse.png')

    # plot the data centres by land requirement on a backdrop of the land price
    ImageManipulation().project_graded_data_centres_onto_map(data_centre_frame=price_df,
                                                             src=solar_src,
                                                             img=landmass_with_prices,
                                                             intensity_column='Expected Total Land Required',
                                                             invert_preference=True,
                                                             title='Preferred Data Centre Locations by land requirement',
                                                             zoombounds=constants.ZOOM_BOUNDS,
                                                             cmap='cool',
                                                             cbar=False,
                                                             savefile='./Data/Output_files/Maps/LandRequirement.png')

    # plot the graded data centres by cost of developing the land
    ImageManipulation().project_graded_data_centres_onto_map(data_centre_frame=price_df,
                                                             src=solar_src,
                                                             img=solar_image,
                                                             intensity_column='Total Land Price (R)',
                                                             invert_preference=True,
                                                             title='Preferred Data Centre Locations by cost requirement',
                                                             zoombounds=constants.ZOOM_BOUNDS,
                                                             cmap='cool',
                                                             cbar=False,
                                                             savefile='./Data/Output_files/Maps/CostRequirement.png')


    # saving the file if required
    if outfile is not None:
        price_df.to_excel(outfile)


def main():
    # Read the data from the excel file
    print ('Reading the data from the excel file')
    data_centre_wrangler = DataCentreWrangler(constants.DATA_CENTRE_FILE)
    data_centre_wrangler.wrangle()

    # Get the solar data from the data reader
    print('Getting the solar data from the data reader')
    reader_solar_data = DataReader(constants.SOLAR_FILE)
    solar_image, solar_transform, solar_src = reader_solar_data.read_tiff()

    # Instantiate image manipulator
    manip = ImageManipulation(save_location='./WorkingData/Maps/')
    solar_enriched_df = manip.get_PV_from_tiff(solar_src, solar_image, data_centre_wrangler.df)



if __name__ == '__main__':
    # main()
    # DC = DataCentreWrangler(constants.DATA_CENTRE_FILE)
    # DC.wrangle()
    # dcf = DC.df
    # s_img, _, s_src = DataReader(constants.YSUM_GHI).read_tiff()
    # img, _, src = DataReader(constants.LAND_USE_FILE).read_tiff(reference_src=s_src, offset=[10, 30])
    # img[img == 0] = np.NaN
    # preference_df = pd.read_excel(constants.LAND_PREF_FILE, sheet_name='Land Use', skiprows=2)
    # img_2 = translate_image_values_by_mapping_frame(img,
    #                                                 preference_df,
    #                                                 key_column='#',
    #                                                 value_column='Implication (Weighting)')
    run_analysis('./Data/Output_files/Spreadsheets/outfile.xlsx')