import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
import rasterio
from rasterio.features import shapes
from rasterio.plot import show
from shapely.geometry import Point
import geopandas as gpd
from DataReader import DataCentreWrangler, DataReader, convert_to_decimal_degrees

SAVE_LOCATION = './WorkingData/Maps/'
DEGREEE_TO_METER = 111139

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
