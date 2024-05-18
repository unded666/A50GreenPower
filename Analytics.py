import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from DataReader import DataCentreWrangler, DataReader, convert_to_decimal_degrees
from ImageManipulation import ImageManipulation
import rasterio
import constants
import folium
from FoliumSaver import folium_saver
import selenium
from PIL import Image


# SOLAR_DIR = './Data/Global Solar Atlas/'
# SOLAR_FILE = SOLAR_DIR + 'PVOUT_Yearly_sum.tif'
# DATA_CENTRE_FILE = './Data/Data centres - preliminary information.xlsx'
# DEGREEE_TO_METER = 111139
# SOLAR_EFFICIENCY = 0.2
# YSUM_DIR = './Data/Global Solar Atlas/YearlySum/'
# YSUM_GHI = YSUM_DIR + 'GHI.tif'
# YSUM_DNI = YSUM_DIR + 'DNI.tif'
# YSUM_DIF = YSUM_DIR + 'DIF.tif'

def save_minimal_image(image: np.ndarray,
                       savefile: str,
                       zoombounds: tuple = constants.ZOOM_BOUNDS,
                       cmap: str = 'Greens') -> None:

    """
    Saves a minimal image of the image, with the specified zoom bounds and colour map.

    :param image: the image to save
    :param savefile: the location to save the image to
    :param zoombounds: the zoom bounds of the image
    :param cmap: the colour map of the image
    :return:
    """

    left, bottom, right, top = zoombounds
    plt.imshow(image[top:bottom, left:right], cmap=cmap)
    plt.axis('off')
    plt.savefig(savefile)

def convert_deg_minutes_string_to_decimal_degrees(deg_minutes_string: str) -> float:
    """
    Converts a string in the format 'CD MM.MMM' to decimal degrees. C is the compass direction, D is the degree,
    and MM.MMM is the minutes. South and West directions are negative.

    :param deg_minutes_string: the string to convert
    :return: the decimal degrees
    """

    # remove trailing whitespace
    deg_minutes_string = deg_minutes_string.strip()

    # Split the string into direction and degree part
    direction, degree_str = deg_minutes_string[0], deg_minutes_string[1:]

    # Split the degree part into degree and minute
    degree, minute = map(float, degree_str.split(' '))

    # Convert to decimal degrees
    decimal_degrees = degree + minute / 60

    # If the direction is South or West, make the result negative
    if direction in ['S', 'W']:
        decimal_degrees = -decimal_degrees

    return decimal_degrees




def rich_map(location,
             zoom_start=15,
             marker_locations=None,
             marker_labels=None) -> folium.Map:
    """
    Creates a rich map of the location by using the folium, returning a map complete with markers and annotations

    :param location: the location to create the map of
    :param zoom_start: the zoom level of the map
    :param marker_locations: the locations of the markers to add to the map
    :param marker_labels: the labels of the markers to add to the map
    :return: None
    """
    # Create a folium map object
    m = folium.Map(location=location, zoom_start=zoom_start)

    # add markers and labels to the map
    if marker_locations is not None and marker_labels is not None:
        for location, label in zip(marker_locations, marker_labels):
            folium.CircleMarker(location=location, popup=label, color='blue', radius=2).add_to(m)
            folium.Marker(location=location,
                          popup=label,
                          icon=folium.DivIcon(html='<div style="font-size: 8pt">%s</div>' % label)).add_to(m)

    return m

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
                                            value_column: str,
                                            default_value = 0.6) -> np.ndarray:

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
            mapping_dict[int(key)] = default_value

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
    centre_frame_copy['Adjusted Total Land Required'] = (centre_frame_copy.apply
                                                         (lambda x: x['Expected Total Land Required'] * (1 + x['vari_score']), axis=1))


    # Replace NaNs in total available land with 0
    centre_frame_copy['Total Available Land'] = centre_frame_copy['Total Available Land'].fillna(0)
    centre_frame_copy['Excess Land Required'] = (centre_frame_copy['Adjusted Total Land Required'] -
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


def get_best_X(dataframe: pd.DataFrame, X: int, column: str, invert_best = False) -> pd.DataFrame:
    """
    Returns the top X rows of the dataframe, sorted by the specified column

    :param dataframe: the dataframe to be sorted
    :param X: the number of rows to return
    :param column: the column to sort by
    :param invert_best: if True, the top X rows are the worst X rows
    :return: the top X rows of the dataframe
    """

    # Create a copy of the dataframe
    dataframe_copy = dataframe.copy()

    # Sort the dataframe by the specified column
    dataframe_copy.sort_values(by=column, ascending=not invert_best, inplace=True)

    # Return the top X rows
    return dataframe_copy.head(X)


def score_sites(dataframe: pd.DataFrame) -> pd.DataFrame:
    """

    :param dataframe: input dataframe
    :return: scored dataframe with added "score" column
    """

    # Create a copy of the dataframe
    dataframe_copy = dataframe.copy()

    # Calculate the score


    return dataframe_copy

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
    solar_image = solar_image.astype(float)
    solar_image[solar_image < 0] = np.NaN
    solar_base = solar_image.copy()

    # get monthly solar data for the data centres
    monthly_wrangler = DataCentreWrangler(constants.DATA_CENTRE_FILE)
    monthly_wrangler.wrangle()
    monthly_wrangler = analyse_monthly_data(monthly_wrangler)
    monthly_df = monthly_wrangler.df
    data_centre_df['std'] = monthly_df['std']
    data_centre_df['mean'] = monthly_df['mean']
    data_centre_df['vari_score'] = data_centre_df['std'] / data_centre_df['mean']

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

    # generate a land price image
    land_price_image = generate_land_price_image()

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

    price_df = ImageManipulation().get_values_from_tiff(src=solar_src,
                                                        img=preference_img,
                                                        data_centre_frame=price_df,
                                                        output_column='Land Use Preference')

    price_df['coordinates'] = price_df.apply(lambda x: [convert_deg_minutes_string_to_decimal_degrees(x['Longitude']),
                                                        convert_deg_minutes_string_to_decimal_degrees(x['Latitude'])], axis=1)

    marker_locations = price_df['coordinates'].tolist()
    marker_labels = price_df['Site'].tolist()

    # plot the data centres on a backdrop of solar energy
    ImageManipulation().project_data_centres_onto_map(data_centre_frame=price_df,
                                                      src=solar_src,
                                                      img_in=solar_base,
                                                      title='Solar Energy Heatmap',
                                                      zoombounds=constants.ZOOM_BOUNDS,
                                                      savefile='./Data/Output_files/Maps/SolarWithDCentres.png')

    # create minimal land price file
    save_minimal_image(image=land_price_image,
                       savefile=constants.LAND_PRICE_MIN_FILE,
                       zoombounds=constants.ZOOM_BOUNDS,
                       cmap='Greens')


    # plot data centres on a folium map of RSA, with land requirement as an intensity value for the data centre markers
    # and an overlay of land price

    O1_map = folium.Map(location=constants.RSA_LOCATION, zoom_start=6)
    O1_map = folium_overlay(map=O1_map,
                            overlay_path=constants.LAND_PRICE_MIN_FILE,
                            overlay_src=solar_src,
                            overlay_stretch=(2.6, 2.1),
                            overlay_offset=(-0.35, 0.3),
                            overlay_opacity=0.5)

    O1_map = ImageManipulation().project_data_centres_onto_folium_map(data_centre_frame=price_df,
                                                                      map=O1_map,
                                                                      title='Data Centre Locations by Land Requirement',
                                                                      pointsize=5,
                                                                      intensity_column='Adjusted Total Land Required',
                                                                      label_column='Site',
                                                                      hide_labels=True,
                                                                      colour_variation=True)
    folium_saver(O1_map, constants.TEMP_MAP_HTML, constants.LAND_REQUIREMENT_OUTPUT)

    # second big output is cost centre prices
    save_minimal_image(image=land_use_img,
                       savefile=constants.LAND_USE_MIN_FILE,
                       zoombounds=constants.ZOOM_BOUNDS,
                       cmap='Greens')

    O2_map = folium.Map(location=constants.RSA_LOCATION, zoom_start=6)
    O2_map = folium_overlay(map=O2_map,
                            overlay_path=constants.LAND_USE_MIN_FILE,
                            overlay_src=solar_src,
                            overlay_stretch=(2.6, 2.1),
                            overlay_offset=(-0.35, 0.3),
                            overlay_opacity=0.5)

    ImageManipulation().project_data_centres_onto_folium_map(data_centre_frame=price_df,
                                                             map=O2_map,
                                                             title='Data Centre Locations by Cost',
                                                             pointsize=5,
                                                             intensity_column='Total Land Price (R)',
                                                             label_column='Site',
                                                             hide_labels=True,
                                                             colour_variation=False,
                                                             invert_intensity=True)

    folium_saver(O2_map, constants.TEMP_MAP_HTML, constants.LAND_COST_OUTPUT)


    # Zoom in somewhere for glory!
    loc = price_df.coordinates[0]
    O3_map = folium.Map(location=loc, zoom_start=13)
    O3 = folium_overlay(map=O3_map,
                        overlay_path=constants.LAND_PRICE_MIN_FILE,
                        overlay_src=solar_src,
                        overlay_stretch=(2.6, 2.1),
                        overlay_offset=(-0.35, 0.3),
                        overlay_opacity=0.5)
    ImageManipulation().project_data_centres_onto_folium_map(data_centre_frame=price_df,
                                                             map=O3_map,
                                                             title='Data Centre Locations by Cost',
                                                             pointsize=8,
                                                             intensity_column='Total Land Price (R)',
                                                             static_intensity=True,
                                                             label_column='Site',
                                                             hide_labels=False,
                                                             colour_variation=False,
                                                             invert_intensity=False)
    folium_saver(O3_map, constants.TEMP_MAP_HTML, './WorkingData/Maps/ZoomedCostMap.png')



    # plot the data centres on a backdrop of solar energy, graded by the variance score
    ImageManipulation().project_graded_data_centres_onto_map(data_centre_frame=price_df,
                                                             src=solar_src,
                                                             img_in=solar_base,
                                                             intensity_column='vari_score',
                                                             invert_preference=True,
                                                             title='Preferred Data Centre Locations by solar reliability',
                                                             zoombounds=constants.ZOOM_BOUNDS,
                                                             cmap='cool',
                                                             cbar=False,
                                                             savefile='./Data/Output_files/Maps/SolarReliability.png')



    # plot the data centres on a backdrop of the land preferences
    ImageManipulation().project_data_centres_onto_map(data_centre_frame=price_df,
                                                      src=solar_src,
                                                      img_in=preference_img,
                                                      title='Preference by land use',
                                                      zoombounds=constants.ZOOM_BOUNDS,
                                                      savefile='./Data/Output_files/Maps/LandUse.png')

    # plot the data centres by land requirement on a backdrop of the land price
    ImageManipulation().project_graded_data_centres_onto_map(data_centre_frame=price_df,
                                                             src=solar_src,
                                                             img_in=landmass_with_prices,
                                                             intensity_column='Expected Total Land Required',
                                                             invert_preference=True,
                                                             title='Preferred Data Centre Locations by land requirement',
                                                             zoombounds=constants.ZOOM_BOUNDS,
                                                             cmap='PuRd',
                                                             cbar=False,
                                                             savefile='./Data/Output_files/Maps/LandRequirementSimple.png')

    # plot data centres on a map of land price
    ImageManipulation().project_data_centres_onto_map(data_centre_frame=price_df,
                                                      src=solar_src,
                                                      img_in=land_price_image,
                                                      title='Land Price in Rands/Hectare',
                                                      zoombounds=constants.ZOOM_BOUNDS,
                                                      savefile='./Data/Output_files/Maps/LandPrice.png')

    # create a high-resolution map of the land price, saved to the temporary file
    ImageManipulation().project_data_centres_onto_map(data_centre_frame=price_df,
                                                        src=solar_src,
                                                        img_in=land_price_image,
                                                        title='Land Price in Rands/Hectare',
                                                        zoombounds=constants.ZOOM_BOUNDS,
                                                        bbox=(12, 8),
                                                        pointsize=1,
                                                        savefile=constants.TEMP_FILE)


    sa_det_map = rich_map(location=constants.RSA_LOCATION,
                          zoom_start=6,
                          marker_locations=marker_locations,
                          marker_labels=marker_labels)

    folium_saver(sa_det_map, constants.TEMP_MAP_HTML, './Data/Output_files/Maps/SAmap.png')

    temp_img = plt.imread(constants.TEMP_FILE)
    zoomed_image = temp_img[250:350, 600:700, :]
    plt.imsave('./Data/Output_files/Maps/LandPriceZoomed.png', zoomed_image)

    # plot the graded data centres by cost of developing the land
    ImageManipulation().project_graded_data_centres_onto_map(data_centre_frame=price_df,
                                                             src=solar_src,
                                                             img_in=solar_base,
                                                             intensity_column='Total Land Price (R)',
                                                             invert_preference=True,
                                                             title='Preferred Data Centre Locations by cost requirement',
                                                             zoombounds=constants.ZOOM_BOUNDS,
                                                             cmap='cool',
                                                             cbar=False,
                                                             savefile='./Data/Output_files/Maps/CostRequirement.png')

    # plot the graded data centres by detailed cost of developing the land
    ImageManipulation().project_graded_detailed_data_centres_onto_map(data_centre_frame=price_df,
                                                                      src=solar_src,
                                                                      img_in=solar_base,
                                                                      intensity_column='Total Land Price (R)',
                                                                      invert_preference=True,
                                                                      title='Preferred Data Centre Locations by cost requirement',
                                                                      zoombounds=constants.ZOOM_BOUNDS,
                                                                      cmap='cool',
                                                                      cbar=False,
                                                                      savefile='./Data/Output_files/Maps/CostRequirementDetailed.png')

    price_best_land_use = price_df[price_df['Land Use Preference'] > 0]
    # show the remaining data centres after the filtering by land use preference
    ImageManipulation().project_data_centres_onto_map(data_centre_frame=price_best_land_use,
                                                      src=solar_src,
                                                      img_in=solar_base,
                                                      title='Data Centre Locations by Solar radiation \nafter removing unwanted sites by land use',
                                                      zoombounds=constants.ZOOM_BOUNDS,
                                                      savefile='./Data/Output_files/Maps/LandUseFilter.png')

    price_reliability = get_best_X(dataframe=price_df, X=10, column='vari_score')
    # show the top 10 most reliable data centres
    ImageManipulation().project_graded_data_centres_onto_map(data_centre_frame=price_reliability,
                                                             src=solar_src,
                                                             img_in=solar_base,
                                                             intensity_column='Total Land Price (R)',
                                                             invert_preference=True,
                                                             title='Most reliable data centres',
                                                             zoombounds=constants.ZOOM_BOUNDS,
                                                             cmap='cool',
                                                             cbar=False,
                                                             savefile='./Data/Output_files/Maps/Top10Reliable.png')

    reliability_limited = price_df[price_df['vari_score'] < 0.08]
    # show the data centres with the lowest variance score
    ImageManipulation().project_graded_data_centres_onto_map(data_centre_frame=reliability_limited,
                                                             src=solar_src,
                                                             img_in=solar_base,
                                                             intensity_column='Total Land Price (R)',
                                                             invert_preference=True,
                                                             title='Most cost-effective Data Centres, all with high reliability',
                                                             zoombounds=constants.ZOOM_BOUNDS,
                                                             cmap='cool',
                                                             cbar=False,
                                                             savefile='./Data/Output_files/Maps/BestCentres8percLenient.png')

    ImageManipulation().project_graded_detailed_data_centres_onto_map(data_centre_frame=reliability_limited,
                                                                      src=solar_src,
                                                                      img_in=solar_base,
                                                                      intensity_column='Total Land Price (R)',
                                                                      invert_preference=True,
                                                                      title='Most cost-effective Data Centres, all with high reliability',
                                                                      zoombounds=constants.ZOOM_BOUNDS,
                                                                      cmap='cool',
                                                                      cbar=False,
                                                                      savefile='./Data/Output_files/Maps/BestCentres8percLenientDetailed.png')




    bar_df = data_centre_df[['Site', 'mean', 'std']]
    bar_df.set_index('Site', inplace=True)
    bar_df.plot(kind='bar')
    plt.title('Mean and Standard Deviation of Monthly Solar Energy by site')
    plt.ylabel('Energy (kWh/m^2)')
    plt.xticks(rotation=80, fontsize=6)
    plt.tight_layout()
    plt.savefig('./Data/Output_files/Maps/MonthlySolarEnergy.png', dpi=600)

    # saving the file if required
    if outfile is not None:
        price_df.to_excel(outfile)

def folium_overlay(map: folium.Map,
                   overlay_path: str,
                   overlay_src: rasterio.io.DatasetReader,
                   overlay_stretch: tuple = (0, 0),
                   overlay_offset: tuple = (0, 0),
                   overlay_opacity: float = 0.3) -> folium.Map:

    """
    Overlays an image on a folium map, using the offset and stretch parameters to orient the overlay.
    :param map:  folium map object
    :param overlay_path: path to the overlay image
    :param overlay_src: rasterio source object
    :param overlay_stretch: tuple of stretch parameters. expected is (x_stretch, y_stretch
    :param overlay_offset: tuple of offset parameters. expected is (x_offset, y_offset)
    :param overlay_opacity: opacity of the overlay
    :return: folium map object
    """

    # Read in the overlay image
    # overlay_img, _, _ = DataReader(overlay_path).read_tiff()
    bound_X0, bound_Y1, bound_X1, bound_Y0 = constants.ZOOM_BOUNDS
    left, top = overlay_src.transform * (bound_X1, bound_Y1)
    right, bottom = overlay_src.transform * (bound_X0, bound_Y0)
    folium.raster_layers.ImageOverlay(image=overlay_path,
                                      bounds=[[top + overlay_offset[1] - overlay_stretch[1], left + overlay_offset[0] + overlay_stretch[0]],
                                              [bottom + overlay_offset[1] + overlay_stretch[1], right + overlay_offset[0] - overlay_stretch[0]]],
                                      opacity=overlay_opacity).add_to(map)

    return map




def run_bespoke_analysis():

    solar_image, solar_transform, solar_src = DataReader(constants.SOLAR_FILE).read_tiff()
    solar_image = solar_image.astype(float)
    solar_image[solar_image < 0] = np.NaN

    property_frame = pd.read_excel(constants.PROP_FILE)

    landmass_with_prices = ImageManipulation().generate_land_prices(solar_image,
                                                                    solar_src,
                                                                    property_frame,
                                                                    constants.PROVINCE_LOCATION_FILE)

    zoombounds = constants.ZOOM_BOUNDS
    CMAP = 'Greens'

    img = solar_image.copy()
    # Plot the image
    if zoombounds is None:
        plt.imshow(img, cmap=CMAP)
    else:
        left, bottom, right, top = zoombounds
        plt.imshow(img[top:bottom, left:right], cmap=CMAP)

    plt.axis('off')


    # plt.axis('equal')  # Set the aspect ratio of the axes to be equal

    savefile = constants.SOLAR_MIN_FILE
    if savefile is not None:
        plt.savefig(savefile)
    plt.close()

    df = pd.read_excel(constants.EXCEL_OUTPUT_FILE)

    # Create a folium map object
    map = folium.Map(location=constants.RSA_LOCATION, zoom_start=6)
    map = folium_overlay(map=map,
                         overlay_path=constants.SOLAR_MIN_FILE,
                         overlay_src=solar_src,
                         overlay_stretch=(2.6, 2.1),
                         overlay_offset=(-0.35, 0.3),
                         overlay_opacity=0.3)

    map = ImageManipulation().project_data_centres_onto_folium_map(data_centre_frame=df,
                                                                   map=map,
                                                                   title='Whatever',
                                                                   pointsize=5,
                                                                   intensity_column='Adjusted Total Land Required',
                                                                   label_column='Site',
                                                                   hide_labels=True,
                                                                   colour_variation=True)

    folium_saver(map, constants.TEMP_MAP_HTML, './Data/Output_files/Maps/SolarOverlay.png')

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
    run_analysis(constants.EXCEL_OUTPUT_FILE)
    # run_bespoke_analysis()