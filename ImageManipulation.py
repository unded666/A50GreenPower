import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
import rasterio
from rasterio.features import shapes, geometry_mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import Point
import geopandas as gpd
from DataReader import DataCentreWrangler, DataReader, convert_to_decimal_degrees
from constants import (SAVE_LOCATION,
                       DEGREEE_TO_METER,
                       PROP_PRICE_LOCATION,
                       PROP_FILE,
                       PROVINCE_LOCATION_FILE,
                       POP_FIL,
                       MAX_POP_SCALE,
                       MIN_POP_SCALE,
                       YSUM_GHI)


class ImageManipulation:
    def __init__(self, save_location=SAVE_LOCATION):
        self.save_location = save_location

    def plot_image(self, image, title, src, cmap='hot', reverse_cmap=True, coords=None, save = True):
        """
        plots the image using matplotlib, saving the image to the noted save location. reverses the colormap by default

        :param image: image to plot
        :param title: title of image
        :param src: rasterio dataset
        :param cmap: base color map
        :param reverse_cmap: whether or not to reverse the colormap
        :param coords: list of (lat, long) tuples
        :return:
        """
        plt.figure()
        if reverse_cmap:
            cmp = cmap + '_r'
        else:
            cmp = cmap

        # Get the geographic extent of the image
        left, bottom, right, top = src.bounds

        # Plot the image with the correct geographic extent
        pl.imshow(image, cmap=cmp, extent=[left, right, bottom, top])
        plt.title(title)
        plt.colorbar()

        if coords is not None:
            for lat, long in coords:
                px, py = self.translate_coordinates_to_pixels(lat, long, src)
                # plt.scatter(px, py, color='blue')
                plt.scatter(long, lat, color='blue', s=4)

        plt.axis('equal')  # Set the aspect ratio of the axes to be equal

        if save:
            plt.savefig(self.save_location + title + '.png')
        plt.close()

    def project_data_centres_onto_map(self,
                                      data_centre_frame: pd.DataFrame,
                                      src: rasterio.io.DatasetReader,
                                      img_in: np.ndarray,
                                      title: str = None,
                                      zoombounds: tuple = None,
                                      pointsize = None,
                                      bbox: int = None,
                                      savefile: str = SAVE_LOCATION):
        """
        Projects the data centres onto the map

        :param data_centre_frame: The data centre frame.
        :param src: The rasterio source object for the image.
        :param img_in: The image as a numpy array.
        :param title: The title of the plot.
        :param zoombounds: The bounds of the plot.
        :param pointsize: The size of the points.
        :param bbox: The size of the image.
        :param savefile: The location to save the plot.
        :return: The data centre frame with the PV values added.
        """

        img = img_in.copy()
        if not bbox is None:
            plt.figure(figsize = bbox)
        # Plot the image
        if zoombounds is None:
            plt.imshow(img, cmap='hot_r')
        else:
            left, bottom, right, top = zoombounds
            plt.imshow(img[top:bottom, left:right], cmap='hot_r')

        plt.colorbar()

        # Get the coordinates of the data centres
        lat = [convert_to_decimal_degrees(L) for L in data_centre_frame['Latitude'].to_list()]
        long = [convert_to_decimal_degrees(L) for L in data_centre_frame['Longitude'].to_list()]
        coords = [(longitude, latitude) for longitude, latitude in zip(long, lat)]

        # default point size if None given
        if pointsize is None:
            pointsize = 4
        # Plot the data centres on the map
        for lat, long in coords:
            px, py = self.translate_coordinates_to_pixels(lat, long, src)
            plt.scatter(px, py, color='blue', s=pointsize)

        plt.axis('equal')  # Set the aspect ratio of the axes to be equal
        if title is not None:
            plt.title(title)
        if savefile is not None:
            if not bbox is None:
                plt.savefig(savefile)
            else:
                plt.savefig(savefile, bbox_inches='tight')
        plt.close()

    def project_graded_detailed_data_centres_onto_map(self,
                                                      data_centre_frame: pd.DataFrame,
                                                      src: rasterio.io.DatasetReader,
                                                      img_in: np.ndarray,
                                                      intensity_column: str,
                                                      title: str = None,
                                                      invert_preference: bool = False,
                                                      zoombounds: tuple = None,
                                                      cmap='hot_r',
                                                      cbar=True,
                                                      savefile: str = SAVE_LOCATION):
        """
        Projects the data centres onto the map with the intensity of the scatter point indicating the relative
        value of the intensity column of interest at the data centre location. All but the 5 most intense points
        are plotted on a scale of white to green, with the 5 most intense points plotted in bright pink

        :param data_centre_frame: The data centre frame.
        :param src: The rasterio source object for the image.
        :param img_in: the image as a numpy array
        :param intensity_column: the column in the data centre frame that contains the intensity values
        :param title: the title of the plot
        :param invert_preference: whether or not to invert the preference of the intensity values
        :param zoombounds: the bounds of the plot
        :param cmap: the color map to use
        :param cbar: whether or not to include a color bar
        :param savefile: the location to save the plot
        """

        img = img_in.copy()
        # Plot the image
        if zoombounds is None:
            plt.imshow(img, cmap=cmap)
        else:
            left, bottom, right, top = zoombounds
            plt.imshow(img[top:bottom, left:right], cmap=cmap)

        if cbar:
            plt.colorbar()

        # sort the dataframe by the intensity column, sort order depending on the invert_preference flag
        data_centre_frame = data_centre_frame.sort_values(by=intensity_column, ascending=not invert_preference)

        # Get the coordinates of the data centres
        lat = [convert_to_decimal_degrees(L) for L in data_centre_frame['Latitude'].to_list()]
        long = [convert_to_decimal_degrees(L) for L in data_centre_frame['Longitude'].to_list()]
        coords = [(longitude, latitude) for longitude, latitude in zip(long, lat)]

        # Get the intensity values
        intensity_values = np.array(data_centre_frame[intensity_column].to_list()[:-5])
        normalised_intensity_values = (intensity_values - np.min(intensity_values)) / (np.max(intensity_values) - np.min(intensity_values))
        if invert_preference:
            normalised_intensity_values = 1 - normalised_intensity_values

        # Plot all but the 5 most intense points
        for (lat, long), intensity in zip(coords[:-5], normalised_intensity_values):
            px, py = self.translate_coordinates_to_pixels(lat, long, src)
            plt.scatter(px, py, color='green', alpha=intensity)

        # Plot the 5 most intense points
        for (lat, long) in coords[-5:]:
            px, py = self.translate_coordinates_to_pixels(lat, long, src)
            plt.scatter(px, py, color='magenta')

        plt.axis('equal')  # Set the aspect ratio of the axes to be equal
        if title is not None:
            plt.title(title)
        if savefile is not None:
            plt.savefig(savefile)

        plt.close()


    def project_graded_data_centres_onto_map(self,
                                             data_centre_frame: pd.DataFrame,
                                             src: rasterio.io.DatasetReader,
                                             img_in: np.ndarray,
                                             intensity_column: str,
                                             invert_preference: bool = False,
                                             title: str = None,
                                             zoombounds: tuple = None,
                                             cmap='hot_r',
                                             cbar=True,
                                             savefile: str = SAVE_LOCATION):
        """
        Projects the data centres onto the map with the intensity of the scatter point indicating the relative
        value of the intensity column of interest at the data centre location"

        :param data_centre_frame: The data centre frame.
        :param src: The rasterio source object for the image.
        :param img_in: The image as a numpy array.
        :param intensity_column: The column in the data centre frame that contains the intensity values.
        :param invert_preference: Whether or not to invert the preference of the intensity values.
        :param title: The title of the plot.
        :param zoombounds: The bounds of the plot.
        :param cmap: The color map to use.
        :param savefile: The location to save the plot.
        :param cbar: Whether or not to include a color bar.
        """

        img = img_in.copy()
        # Plot the image
        if zoombounds is None:
            plt.imshow(img, cmap=cmap)
        else:
            left, bottom, right, top = zoombounds
            plt.imshow(img[top:bottom, left:right], cmap=cmap)

        if cbar:
            plt.colorbar()

        # Get the coordinates of the data centres
        lat = [convert_to_decimal_degrees(L) for L in data_centre_frame['Latitude'].to_list()]
        long = [convert_to_decimal_degrees(L) for L in data_centre_frame['Longitude'].to_list()]
        coords = [(longitude, latitude) for longitude, latitude in zip(long, lat)]

        # Get the intensity values
        intensity_values = np.array(data_centre_frame[intensity_column].to_list())
        normalised_intensity_values = (intensity_values - np.min(intensity_values)) / (np.max(intensity_values) - np.min(intensity_values))
        if invert_preference:
            normalised_intensity_values = 1 - normalised_intensity_values

        # Plot the data centres on the map
        for (lat, long), intensity in zip(coords, normalised_intensity_values):
            px, py = self.translate_coordinates_to_pixels(lat, long, src)
            plt.scatter(px, py, color='green', alpha=intensity)

        # plt.axis('equal')  # Set the aspect ratio of the axes to be equal
        if title is not None:
            plt.title(title)
        if savefile is not None:
            plt.savefig(savefile)
        plt.close()


    def get_pixel_size(self, geotiff_path) -> tuple:
        """
        Get the size of a pixel in a GeoTIFF image. Converts from degrees into meters and square meters

        :param geotiff_path: The path to the GeoTIFF file.
        :return: The size of a pixel in the x-direction, the size of a pixel in the y-direction, and the area of a pixel.
        """
        with rasterio.open(geotiff_path) as src:
            transform = src.transform
            pixel_size_x = transform[0]*DEGREEE_TO_METER
            pixel_size_y = -transform[4]*DEGREEE_TO_METER  # pixel size in y is reported as negative, so we take the absolute value
        return pixel_size_x, pixel_size_y, pixel_size_x*pixel_size_y

    def get_pixel_value(self, lat, long, src, img):
        """
        Get the pixel value at the given latitude and longitude

        :param lat: latitude
        :param long: longitude
        :param src: rasterio dataset
        :return: pixel value
        """
        # Convert the latitude and longitude to pixel coordinates
        px, py = self.translate_coordinates_to_pixels(lat, long, src)

        # Read the pixel value at the given coordinates
        # pixel_value = next(src.sample([(px, py)]))
        pixel_value = img[int(py), int(px)]

        return pixel_value

    def translate_coordinates_to_pixels(self, lat, long, src):
        """
        Translates latitude and longitude coordinates to pixel coordinates

        :param lat: latitude
        :param long: longitude
        :param transform: affine transformation matrix
        :return: pixel coordinates
        """
        return ~src.transform * (long, lat)

    def get_values_from_tiff(self, src, img, data_centre_frame: pd.DataFrame, output_column: str) -> pd.DataFrame:
        """
        gets the PV values from the source img at locations specified in the data_centre_frame dataframe,
        taking latitude and longitude from the 'latitude' and 'longitude' columns respectively. Returns a dataframe

        :param src:
        :param img:
        :param data_centre_frame:
        :return:
        """
        # Get the coordinates of the data centres
        # lat, long, coords = data_centre_frame.get_coordinate_list()
        lat = [convert_to_decimal_degrees(L) for L in data_centre_frame['Latitude'].to_list()]
        long = [convert_to_decimal_degrees(L) for L in data_centre_frame['Longitude'].to_list()]
        coords = [(longitude, latitude) for longitude, latitude in zip(long, lat)]

        # Get the PV values at the data centre locations
        output_values = [self.get_pixel_value(lat, long, src, img) for lat, long in coords]

        # Add the PV values to the data centre frame
        data_centre_frame[output_column] = output_values

        return data_centre_frame

    def get_base_landmass_shape(self, img: np.ndarray) -> np.ndarray:
        """
        This function is used to get the base shape of a landmass from an image. The image is expected to be a numpy array
        where the landmass is represented by non-NaN values and the non-landmass areas are represented by NaN values.

        The function works by creating a copy of the input image and then setting all non-NaN values to 1. This results in
        a binary image where the landmass is represented by 1s and everything else is represented by NaNs.

        :param img: The input image as a numpy array.
        :return: A binary image representing the base shape of the landmass.
        """

        landmass = img.copy()
        landmass[~np.isnan(landmass)] = 1

        return landmass

    def assign_land_prices_to_base_shape(self,
                                         landmass_img: np.ndarray,
                                         landmass_src: rasterio.io.DatasetReader,
                                         province_prices: pd.DataFrame,
                                         province_shape_path: str) -> np.ndarray:
        """
        This function is used to assign land prices to the base shape of a landmass. The base shape is expected to be a
        binary image where the landmass is represented by 1s and everything else is represented by NaNs. The function
        works by iterating over the polygons in the province_shapes GeoDataFrame and assigning the corresponding land
        price to the landmass pixels that fall within each polygon.

        :param landmass:
        :param province_prices:
        :param province_shapes:
        :return:
        """

        # Create a copy of the landmass
        land_prices = landmass_img.copy()
        province_shape_gdf = gpd.read_file(province_shape_path)
        # create a dictionary of province names and their corresponding base prices
        province_prices_dict = province_prices.set_index(province_prices.columns[0])[province_prices.columns[1]].to_dict()
        # extract base baseline prices and add to the geopandas dataframe
        province_shape_gdf['base_price'] = [province_prices_dict[Name] for Name in province_shape_gdf.name.values]
        # Write the base prices to the landmass, using the geometry of the province shapes as a mask
        for index, row in province_shape_gdf.iterrows():
            mask = geometry_mask([row['geometry']],
                                 transform=landmass_src.transform,
                                 out_shape=landmass_src.shape,
                                 invert=True)
            land_prices[mask] = row['base_price']


        return land_prices

    def refine_land_prices_with_population(self,
                                           land_price_map: np.ndarray,
                                           land_price_src: rasterio.io.DatasetReader,
                                           population_map_src: str,
                                           province_shape_path: str) -> np.ndarray:
        """
        This function is used to refine the land price map by scaling the land prices based on the population density of the

        :param land_price_map: The land price map as a numpy array.
        :param land_price_src: The rasterio source object for the land price map.
        :param population_map_src: The rasterio source object for the population density map.
        :return: A numpy array representing the refined land price map.
        """

        # Get the geographic extent of the land price map
        left, bottom, right, top = land_price_src.bounds

        # ensure we have a numpy array of floats for the land price map
        land_price_map_float = land_price_map.astype(float)

        # Open the population map source
        with rasterio.open(population_map_src) as pop_src:
            # Convert the geographic extent to pixel coordinates
            left_px, top_px = ~pop_src.transform * (left, top)
            right_px, bottom_px = ~pop_src.transform * (right, bottom)
            # Make sure the pixel coordinates are integers
            left_px, top_px, right_px, bottom_px = map(int, [left_px, top_px, right_px, bottom_px])
            the_window = rasterio.windows.Window.from_slices((top_px, bottom_px), (left_px, right_px))

            population_map_aligned = pop_src.read(1,
                                                  window=the_window,
                                                  out_shape=land_price_map.shape,
                                                  resampling=Resampling.bilinear)
            # population_map_aligned = interpolate_population_map(land_price_src, land_price_map, population_map_src, the_window)


        # Replace all negative values in the population map with NaNs
        population_map_aligned[population_map_aligned < 0] = np.nan
        population_map_aligned[population_map_aligned == 0] = 1
        log_population = np.log(population_map_aligned)

        # read in prince shape geodataframe
        province_shape_frame = gpd.read_file(province_shape_path)

        # Cycle through the land price regions and scale the prices based on the population density
        for index, row in province_shape_frame.iterrows():
            mask = geometry_mask([row['geometry']],
                                 transform=land_price_src.transform,
                                 out_shape=land_price_src.shape,
                                 invert=True)

            # Get the relative population density (relative to the political area)
            masked_log_population = log_population[mask]
            relative_log_population = masked_log_population / np.nanmean(masked_log_population)

            # Scale the land prices based on the relative population density, ensuring that the prices are within the
            # bounds set by the MAX_POP_SCALE and MIN_POP_SCALE constants. This is done by creating a straight-line
            # function that maps the relative_log_population values to the range [MIN_POP_SCALE, MAX_POP_SCALE].
            relative_log_population = np.interp(relative_log_population,
                                                (np.nanmin(relative_log_population), np.nanmax(relative_log_population)),
                                                (MIN_POP_SCALE, MAX_POP_SCALE))

            # scale the land prices based on the relative population density
            land_price_map_float[mask] *= relative_log_population

        return land_price_map_float

    def generate_land_prices(self, img: np.ndarray, src, province_prices: pd.DataFrame, province_shapes: gpd.GeoDataFrame) -> np.ndarray:
        """
        This function is used to generate a land price map for a given landmass. The landmass is expected

        :param img: The input image as a numpy array.
        :param src: The rasterio source object for the input image.
        :param province_prices: A DataFrame containing the land prices for each province.
        :param province_shapes: A GeoDataFrame containing the shapes of the provinces.
        :return: A numpy array representing the land price map.
        """

        # Get the base shape of the landmass
        landmass = self.get_base_landmass_shape(img)
        landmass_with_base_prices = self.assign_land_prices_to_base_shape(landmass, src, province_prices, province_shapes)
        landmass_with_refined_prices = self.refine_land_prices_with_population(landmass_with_base_prices,
                                                                               src,
                                                                               POP_FIL,
                                                                               province_shapes)

        return landmass_with_refined_prices


def determine_name (latitude: float, longitude: float, location_frame: gpd.GeoDataFrame):
    """
    determines if the given latitude and longitude are within the bounds of a location in the location_frame,
    returning the corresponding 'name' value from the dataframe if the location is found, and 'None' otherwise

    :param latitude: latitude to search for
    :param longitude: longitude to search for
    :param location_frame: geopandas dataframe containing location data
    :return: name of the location
    """

    for index, row in location_frame.iterrows():
        if row['geometry'].contains(Point(longitude, latitude)):
            return row['name']
    return 'None'

def get_province_baseline(target_province: str, df_in: pd.DataFrame) -> float:
    """
    Returns the per-hectare land price baseline for a given province. The price is read from the
    'PricePerHectare' column in the input dataframe. If the target province is not found in the dataframe, the function
    returns -1.

    Args:
        target_province (str): The name of the province for which the baseline price is to be found.
        df_in (pd.DataFrame): The input dataframe containing province names and corresponding per-hectare land prices.

    Returns:
        float: The per-hectare land price baseline for the target province if found, else -1.
    """
    # Check if the target province is present in the dataframe
    if target_province in df_in['Province'].to_list():
        # If present, return the corresponding per-hectare land price
        return df_in[df_in['Province'] == target_province]['PricePerHectare'].values[0]
    else:
        # If not present, return -1
        return -1

def get_province_baseline_from_geocoordinates(longitude:  float,
                                              latitude: float,
                                              df_in: pd.DataFrame,
                                              gdf_in: gpd.GeoDataFrame) -> float:
    """
    Returns the per-hectare land price baseline for a given province. The price is read
    from the 'PricePerHectare' column in the input dataframe. If the target province is
    not found in the dataframe, the function returns -1. The target province is first read
    from the gdf_in geopandas dataframe using the given latitude and longitude.

    :param longitude: longitude of point of interest
    :param latitude: latitude of point of interest
    :param df_in: dataframe containing province price-per-hectare values
    :param gdf_inf: geodataframe conatining province boundaries
    :return:
    """

    province = determine_name(latitude, longitude, gdf_in)
    return get_province_baseline(province, df_in)


if __name__ == '__main__':

    # Read in image and get src
    data = DataReader('./Data/Global Solar Atlas/PVOUT.tif')
    data2 = DataReader(YSUM_GHI)
    PVimage, PVtransform, PVsrc = data.read_tiff()
    PV2, PVT2, PVsrc2 = data2.read_tiff()
    manipulator = ImageManipulation()

    # Get the base shape of the landmass
    landmass_with_base_prices = manipulator.generate_land_prices(PVimage,
                                                                 PVsrc,
                                                                 pd.read_excel(PROP_FILE),
                                                                 PROVINCE_LOCATION_FILE)


