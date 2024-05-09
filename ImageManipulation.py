import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
import rasterio
from rasterio.features import shapes, geometry_mask
from rasterio.plot import show
from shapely.geometry import Point
import geopandas as gpd
from DataReader import DataCentreWrangler, DataReader, convert_to_decimal_degrees
from constants import SAVE_LOCATION, DEGREEE_TO_METER, PROP_PRICE_LOCATION, PROP_FILE, PROVINCE_LOCATION_FILE, POP_FIL

# SAVE_LOCATION = './WorkingData/Maps/'
# DEGREEE_TO_METER = 111139
# PROP_PRICE_LOCATION = './Data/PropertyPrices/'
# PROP_FILE = PROP_PRICE_LOCATION + 'RSAPropertyPrices.xlsx'
# PROVINCE_LOCATION_FILE = './Data/Location/za.json'
# POP_FIL = './Data/NASA-SEDAC/gpw-v4-population-density-rev11_2020_2pt5_min_tif/gpw_v4_population_density_rev11_2020_2pt5_min.tif'

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

    def get_PV_from_tiff(self, src, img, data_centre_frame: pd.DataFrame) -> pd.DataFrame:
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
        PV_values = [self.get_pixel_value(lat, long, src, img) for lat, long in coords]

        # Add the PV values to the data centre frame
        data_centre_frame['PV'] = PV_values

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
                                           population_map_src: rasterio.io.DatasetReader,
                                           province_shape_frame: gpd.GeoDataFrame) -> np.ndarray:
        """
        This function is used to refine the land price map by scaling the land prices based on the population density of the

        :param land_price_map: The land price map as a numpy array.
        :param land_price_src: The rasterio source object for the land price map.
        :param population_map_src: The rasterio source object for the population density map.
        :return: A numpy array representing the refined land price map.
        """

        # Read the population density map
        population_map = population_map_src.read(1)

        # Get the geographic extent of the land price map
        left, bottom, right, top = land_price_src.bounds

        # Align the population map with the land price map
        population_map_aligned = population_map_src.read(1,
                                                         window=land_price_src.window(left, bottom, right, top))

        # Cycle through the land price regions and scale the prices based on the population density
        for index, row in province_shape_frame.iterrows():
            mask = geometry_mask([row['geometry']],
                                 transform=land_price_src.transform,
                                 out_shape=land_price_src.shape,
                                 invert=True)
            scaled_population = population_map_aligned[mask] / np.nanmean(population_map_aligned[mask])
            land_price_map[mask] *= scaled_population[mask]

        return land_price_map

    def generate_land_prices(self, img: np.ndarray, src, province_prices: pd.DataFrame, province_shapes: gpd.GeoDataFrame) -> np.ndarray:
        """
        This function is used to generate a land price map for a given landmass. The landmass is expected

        :param img: The input image as a numpy array.
        :param province_prices: A DataFrame containing the land prices for each province.
        :param province_shapes: A GeoDataFrame containing the shapes of the provinces.
        :return: A numpy array representing the land price map.
        """

        # Get the base shape of the landmass
        landmass = self.get_base_landmass_shape(img)
        landmass_with_base_prices = self.assign_land_prices_to_base_shape(landmass, src, province_prices, province_shapes)

        return landmass_with_base_prices


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
    # Read in data centre info
    wrangler = DataCentreWrangler('./Data/Data centres - preliminary information.xlsx')
    wrangler.wrangle()

    # Convert coordinates to decimal degrees
    # lat = convert_to_decimal_degrees(wrangler.df['Latitude'].values())
    # long = convert_to_decimal_degrees(wrangler.df['Longitude'].values())
    lat = [convert_to_decimal_degrees(L) for L in wrangler.df['Latitude'].to_list()]
    long = [convert_to_decimal_degrees(L) for L in wrangler.df['Longitude'].to_list()]
    coords = [(longitude, latitude) for longitude, latitude in zip(long, lat)]
    # coords = [(long, lat)]

    # Read in image and get src
    data = DataReader('./Data/Global Solar Atlas/PVOUT.tif')
    PVimage, PVtransform, PVsrc = data.read_tiff()
    # with rasterio.open(data.file_path) as src:
    #     PVimage = src.read(1)

    # Plot
    manipulator = ImageManipulation()
    # manipulator.plot_image(PVimage, 'Solar Power Output plus Data Centres2', PVsrc, coords=coords)
    PVexpectations = [manipulator.get_pixel_value(lat, long, PVsrc, PVimage) for lat, long in coords]
    print(PVexpectations)

    # Get the base shape of the landmass
    landmass = manipulator.get_base_landmass_shape(PVimage)

    landmass_with_base_prices = manipulator.generate_land_prices(PVimage, PVsrc, pd.read_excel(PROP_FILE), PROVINCE_LOCATION_FILE)

