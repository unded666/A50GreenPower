import geopandas as gpd
import pandas as pd
import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import Resampling
from rasterio.features import shapes
import constants

COLUMN_MAPPING = ['Country',
                  'Location',
                  'Total Land Area',
                  'Total Build Area',
                  'Total Available Land',
                  'Total Annual Power Consumption',
                  'Land Requirement for Solar']


def convert_to_decimal_degrees(coord_str):
    # Split the string into direction and degree part
    direction, degree_str = coord_str[0], coord_str[1:]

    # Split the degree part into degree and minute
    degree, minute = map(float, degree_str.split())

    # Convert to decimal degrees
    decimal_degrees = degree + minute / 60

    # If the direction is South or West, make the result negative
    if direction in ['S', 'W']:
        decimal_degrees = -decimal_degrees

    return decimal_degrees

class DataCentreWrangler:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_excel_data(self):
        self.df: pd.DataFrame = pd.read_excel(self.file_path)

        return self.df

    def rename_columns(self, column_names=COLUMN_MAPPING):
        old_columns = [col for col in self.df.columns]
        col_map = {old_col: new_col for old_col, new_col in zip(old_columns, column_names)}
        self.df.rename(columns=col_map, inplace=True)

        return self.df

    def fill_country_na(self):
        self.df['Country'] = self.df['Country'].ffill()

        return self.df

    def slice_country(self, country='South Africa'):
        self.df = self.df[self.df['Country'] == country]

        return self.df

    def reshape_Location(self):
        # Create 'group' and 'subgroup' columns
        self.df['group'] = self.df.index // 3
        self.df['subgroup'] = self.df.index % 3

        # Pivot DataFrame
        df_pivot = self.df.pivot(index='group', columns='subgroup', values='Location')

        # Reset index and rename columns
        df_pivot.reset_index(inplace=True)
        df_pivot.columns = ['group', 'Site', 'Latitude', 'Longitude']

        # Ensure 'group' column in both dataframes are of the same data type
        self.df['group'] = self.df['group'].astype(int)
        df_pivot['group'] = df_pivot['group'].astype(int)

        # Join df_pivot with original DataFrame
        self.df = self.df.merge(df_pivot, on='group')

        # Drop 'group', 'subgroup', and 'Location' columns
        self.df.drop(columns=['group', 'subgroup', 'Location'], inplace=True)

        # drop duplicates
        self.df.drop_duplicates(inplace=True)

        return self.df

    def get_coordinate_list(self):
        lat = [convert_to_decimal_degrees(L) for L in self.df['Latitude'].to_list()]
        long = [convert_to_decimal_degrees(L) for L in self.df['Longitude'].to_list()]
        coords = [(longitude, latitude) for longitude, latitude in zip(long, lat)]

        return lat, long, coords

    def wrangle(self):

        _ = self.read_excel_data()
        _ = self.rename_columns()
        _ = self.fill_country_na()
        _ = self.slice_country()
        _ = self.reshape_Location()


class DataReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_geotiff(self):
        with rasterio.open(self.file_path) as src:
            image = src.read(1)  # assuming you want to read the first band
            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v)
                in enumerate(
                shapes(image, transform=src.transform)))

        geodf = gpd.GeoDataFrame.from_features(list(results))
        return geodf

    def read_population_data(self) -> tuple[np.ndarray, rasterio.io.DatasetReader, rasterio.transform]:
        """
        Reads the population data from the file path and returns the data, transform, and source, after
        modifying the image to replace all negative values with numpy NaNs

        :return: a tuple containing the population data, transform, and source
        """

        # Read in required data
        img_in, txfm_in, src_in = self.read_tiff()
        img_in[img_in < 0] = np.nan

        return img_in, txfm_in, src_in

    def get_all_tiffs(self, path: str):
        """
        returns all tiff files, with full paths, in the specified directory

        :param path:
        :return:
        """

        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif')]

    def read_geojson(self) -> gpd.GeoDataFrame:
        """
        This method reads a GeoJSON file and returns a GeoDataFrame.

        The method uses the geopandas library's read_file function to read the GeoJSON file.
        The file path is taken from the instance variable self.file_path.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the data from the GeoJSON file.
        """
        gdf = gpd.read_file(self.file_path)
        return gdf

    def read_tiff(self, reference_src: rasterio.io.DatasetReader=None):
        with rasterio.open(self.file_path) as src:
            if reference_src is None:
                image = src.read(1)
                transform = src.transform
            else:
                overlap_window = from_bounds(*src.bounds, reference_src.transform)
                nonsense_image = src.read(1, window=overlap_window)

                sampled_image = src.read(1,
                                         out_shape=nonsense_image.shape,
                                         resampling=Resampling.bilinear)
                transform = reference_src.transform

                # create a numpy ndarray filled with NaNs, of the size of the reference image
                image = np.full(reference_src.shape, np.nan)
                # copy the sampled image into the correct location in the reference image
                image[:sampled_image.shape[0], :sampled_image.shape[1]] = sampled_image

        return image, transform, src

    def read_shapefile(self):
        gdf = gpd.read_file(self.file_path)
        return gdf

    def read_csv(self):
        df = pd.read_csv(self.file_path)
        return df

    def read_excel(self):
        df = pd.read_excel(self.file_path)
        return df

    def read_txt(self):
        df = pd.read_csv(self.file_path, sep='\t')
        return df

    def read_json(self):
        df = pd.read_json(self.file_path)
        return df

    def read_html(self):
        df = pd.read_html(self.file_path)
        return df

    def read_hdf(self):
        df = pd.read_hdf(self.file_path)
        return df

    def read_pickle(self):
        df = pd.read_pickle(self.file_path)
        return df

    def read_msgpack(self):
        df = pd.read_msgpack(self.file_path)
        return df

    def read_feather(self):
        df = pd.read_feather(self.file_path)
        return df

    def read_parquet(self):
        df = pd.read_parquet(self.file_path)
        return df

    def read_orc(self):
        df = pd.read_orc(self.file_path)
        return df

    def read_sas(self):
        df = pd.read_sas(self.file_path)
        return df

    def read_spss(self):
        df = pd.read_spss(self.file_path)
        return df

if __name__ == '__main__':

    # data = DataReader('./Data/SA_NLC_2020_ALBERS.tif')
    # geodf = data.read_geotiff()
    # print (geodf.head())
    solar_data = DataReader(constants.YSUM_GHI)
    land_data = DataReader(constants.LAND_USE_FILE)
    s_img, _, s_src = solar_data.read_tiff()
    l_img, _, l_src = land_data.read_tiff(reference_src=s_src)
