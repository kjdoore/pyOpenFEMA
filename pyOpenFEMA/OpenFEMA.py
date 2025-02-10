from __future__ import annotations
from typing import TYPE_CHECKING

import fsspec
import pandas as pd
import geopandas as gpd
import json
import yaml
import warnings
from pyOpenFEMA.api_url_generator import generate_url

if TYPE_CHECKING:
    from pandas import DataFrame


class OpenFEMA():
    """
    Class for reading OpenFEMA datasets into pandas dataframes.

    OpenFEMA API documentaion: https://www.fema.gov/about/openfema/api
    """
    
    def __init__(self,
                 openapi_metadata_endpoint: str = "https://www.fema.gov/api/open/metadata/v3.0/OpenApi",
                 file_format: str = "json"):
        """
        Initializes the class using the specified OpenAPI metadata endpoint.

        Parameters
        ----------
        openapi_metadata_endpoint : str, default "https://www.fema.gov/api/open/metadata/v3.0/OpenApi"
            The URL to the OpenFEMA OpenAPI metadata endpoint.
            This end point contains the meta data of all datasets connected to the API.
        file_format : str, default "json"
            The file format of the endpoint.
        """
        # Create a https file system to read the endpoint metadata file
        fs = fsspec.filesystem("https")
        valid_formats = ['json', 'yaml']

        if file_format not in valid_formats:
            raise ValueError(
                f"The specified format of '{file_format}' for the metadata endpoint is "
                "not available. Acceptable formats are 'json' or 'yaml'"
            )

        # Open the endpoint metadata file
        try:
            openapi_metadata_file = fs.open(f'{openapi_metadata_endpoint}.{file_format}')
        except FileNotFoundError:
            raise FileNotFoundError(
                'The OpenFEMA OpenAPI metadata endpoint of '
                f'{openapi_metadata_endpoint}.{file_format} is not valid. Double check '
                'the endpoint at https://www.fema.gov/about/openfema/api.'
            )

        # Read the metadata file in as json or yaml
        if file_format == 'json':
            self.openapi_metadata = json.loads(openapi_metadata_file.read())
        elif file_format == 'yaml':
            self.openapi_metadata = yaml.safe_load(openapi_metadata_file.read())

        # Extract the URL of the OpenAPI server
        self.url = [server_dict['url']
                    for server_dict in self.openapi_metadata['servers']
                    if server_dict['description'] == 'Production'][0]

        # Get the json key for the dataset containing the metadata of all datasets
        datasets_keys = pd.Series([dataset for dataset in self.openapi_metadata['paths'].keys()])
        metadata_dataset_key = list(datasets_keys[datasets_keys.str.contains('DataSets')])

        if len(metadata_dataset_key) != 1:
            raise NotImplementedError('Multiple possible options found for the '
                                      'metadata for the OpenFEMA API data sets. '
                                      f'Current options are {metadata_dataset_key}. This '
                                      'will require a patch to the OpenFEMA package.')
        else:
            metadata_dataset_key = metadata_dataset_key[0]

        # Read the metadata dataset file as parquet
        # (parquet is a compressed file and should read the fastest)
        try:
            self.df_metadata_dataset = pd.read_parquet(
                f'{self.url}{metadata_dataset_key}.parquet'
            )
        except ValueError:
            raise NotImplementedError('The parquet file containing the metadata for the '
                                      'OpenFEMA API data sets no longer exists. '
                                      'This will require a patch to the OpenFEMA package.')
            

    def _validate_metadata_dataset_list(self):
        """
        Validates the metadata dataset has the individual dataset names.
        """
        # Check if he metadata dataset has the column name (which contains the datasets)
        if ~(self.df_metadata_dataset.columns.str.contains('name').any()):
            raise NotImplementedError('The column indicating the names of the datasets in '
                                      'the metadata for the OpenFEMA API data sets has changed. '
                                      'This will require a patch to the OpenFEMA package.')


    def _check_if_dataset_exists(self, dataset: str):
        """
        Validates if the chosen dataset exists in the metadata dataset.

        Parameters
        ----------
        dataset : str
            The name of the dataset to validate for existence.
        """
        self._validate_metadata_dataset_list()

        # Confirm the dataset is in the list of datasets
        if pd.Series(self.list_datasets()).isin([dataset]).sum() != 1:
            raise ValueError(
                f'The specified dataset of {dataset} was not found in the dataset list. '
                'Please ensure the dataset selected is correct.'
            )


    def list_datasets(self) -> list[str]:
        """
        Returns a list of all non-metadata datasets available on OpenFEMA.

        Returns
        -------
        dataset_names : list[str]
            A list of all datasets available on OpenFEMA.
        """
        self._validate_metadata_dataset_list()

        # Get the dataset names from the metadata dataset
        dataset_names = self.df_metadata_dataset['name']

        # Drop the two metadata datasets (Datasets and DataSetFields)
        metadata_datasets = dataset_names.str.contains('DataSet')
        if metadata_datasets.sum() > 2:
            metadata_datasets_names = list(dataset_names[metadata_datasets])
            warnings.warn("Non-metadata datasets may have been dropped from the "
                          "dataset list. Currently dropped metadata datasets: "
                          f"{metadata_datasets_names}.", UserWarning)
        else:
            dataset_names = dataset_names[~metadata_datasets]

        # Return the list of datasets
        return list(dataset_names.sort_values())


    def dataset_info(self, dataset: str) -> dict:
        """
        Returns the metadata info on the specified dataset.

        Parameters
        ----------
        dataset : str
            The name of the dataset to get metadata for.

        Returns
        -------
        dataset_dict : dict
            A dictionary containing a variety of metadata info on the specified dataset.
        """
        self._validate_metadata_dataset_list()

        self._check_if_dataset_exists(dataset)
        
        # Get the dataset names from the metadata dataset
        dataset_names = self.df_metadata_dataset['name']
        
        # Get the dataset metadata row from the metadata dataset
        # and convert to a dictionary
        dataset_dict = self.df_metadata_dataset[dataset_names == dataset].to_dict(orient='records')[0]

        # This metadata dictionary doesn't contain the columns. Let's add that.
        column_dtypes = self._get_dataset_dtype(dataset)
        dataset_dict['columns'] = column_dtypes

        return dataset_dict


    def read_dataset(self,
                     dataset: str,
                     columns: list[str] | None = None,
                     filters: list[list[tuple]] | None = None,
                     sort_by: list[tuple] | None = None,
                     top: int | None = None,
                     skip: int | None = None,
                     parse_dates: bool = True
                    ) -> DataFrame:
        """
        Reads the specified dataset into a dataframe.

        Parameters
        ----------
        dataset : str
            The name of the dataset to read.
        columns : list[str], default None
            A list of the columns to read.
            Defaults to None, meaning all columns are read.
        filters : list[list[tuple]], default None
            Filter to apply to the data.
            Filter syntax: [[(column, op, val), ...],...] where column is the column name;
            op is a string operator of 'eq', 'ne', 'gt', 'ge', 'lt', 'le', 'in', 'not',
            'substringof', 'endswith', 'startswith', 'contains', or 'geo.intersects'
            (See https://www.fema.gov/about/openfema/api#filter for details on each operator);
            and val is the limiting value(s).
            The innermost tuples are transposed into a set of filters applied through an `AND` operation.
            The outer list combines these sets of filters through an `OR` operation.
        sort_by : list[tuple], default None
            The sorting to apply to the data.
            Sort syntax: [(column, ascending), ...]  where ascending is a boolen indicating if the sort should be in ascending order (True is ascending, False is descending).
            The order of each tuple expresses sorting order, with the first tuple specifying the column that is sorted first.
        top : int, default None
            The number of records returned.
            Defaults to returning all records the meet the specified criteria.
        skip : int, default None
            The number of records to skip in the dataset.
            Defaults to skipping no records.
        parse_dates : bool, default True
            If true, automatically parse any datetime columns into datetime dtype.
            Otherwise, set them as strings.
            Setting this to false is useful if there is an error in the datetime that prevents it from being parsed.
        
        Returns
        -------
        df_dataset : DataFrame
            The dataframe containing the dataset.
        """
        # Get the possible file formats from the metadata dataset
        file_formats = [json.loads(format_str)['format']
                        for format_str in self.dataset_info(dataset)['distribution']]

        # Get the API URL
        web_service_url = self.dataset_info(dataset)['webService']
        
        # Get the columns and dtypes of the dataset
        column_dtypes = self._get_dataset_dtype(dataset)

        # We will select the file format in a preferential order of parquet, then csv, then jsona
        # However, if the dataset is geospatial, select a geojson first if possible
        if 'geojson' in file_formats:
            file_format = 'geojson'
        elif 'parquet' in file_formats:
            file_format = 'parquet'
        elif 'csv' in file_formats:
            file_format = 'csv'
        elif 'jsona' in file_formats:
            file_format = 'jsona'

        # Get the URL of the dataset
        url = generate_url(web_service_url, column_dtypes, file_format, columns, filters, sort_by, top, skip)

        # We will read data in a preferential order of parquet, then csv, then jsona
        # However, if the dataset is geospatial, read as a geojson first if possible
        if 'geojson' in file_formats:
            df_dataset = gpd.read_file(url, engine='pyogrio', use_arrow=True)
        elif 'parquet' in file_formats:
            df_dataset = pd.read_parquet(url)
        elif 'csv' in file_formats:
            df_dataset = pd.read_csv(url)
        elif 'jsona' in file_formats:
            df_dataset = pd.read_json(url)

        # Enforce the dtype of each column
        # Limit to the requested columns
        if columns:
            column_dtypes = {column: dtype for column, dtype in column_dtypes.items() if column in columns}
        df_dataset = self._set_dataset_dtype(df_dataset, column_dtypes, parse_dates=parse_dates)

        return df_dataset


    def _get_dataset_dtype(self, dataset: str) -> dict:
        """
        Gets the dtype of each column in the given dataset.

        Parameters
        ----------
        dataset : str
            The name of the dataset to get its columns' dtypes.

        Returns
        -------
        column_dtypes : dict
            A dictionary containing the column names as the keys and dtype as the values.
        """
        # The column dtypes are in the OpenAPI metadata.
        # First, get the dataset name in the OpenAPI metadata
        openapi_dataset_metadata = self.openapi_metadata['components']['schemas']
        dataset_names = list(openapi_dataset_metadata.keys())
        dataset_openapi_name = [dataset_name for dataset_name in dataset_names if dataset == dataset_name.split('-')[-1]]

        # Confirm this is the only dataset with the OpenAPI metadata name
        if len(dataset_openapi_name) != 1:
            raise NotImplementedError('Multiple possible options found for the '
                                      'dataset in the OpenFEMA API metadata. '
                                      f'Current options are {dataset_openapi_name}. This '
                                      'will require a patch to the OpenFEMA package.')
        else:
            dataset_openapi_name = dataset_openapi_name[0]

        # Extract the columns dtype metadata
        column_metadata = openapi_dataset_metadata[dataset_openapi_name]['properties']
        column_dtypes = {key: value['format'] for key, value in column_metadata.items()}

        return column_dtypes


    def _set_dataset_dtype(self,
                           df_dataset: DataFrame,
                           column_dtypes: dict,
                           parse_dates: bool = True
                          ) -> DataFrame:
        """
        Gets the dtype of each column in the given dataset.

        Parameters
        ----------
        df_dataset : DataFrame
            The dataframe containing the dataset.
        column_dtypes : dict
            A dictionary containing the column names as the keys and dtype as the values.
        parse_dates : bool, default True
            If true, automatically parse any datetime columns into datetime dtype.
            Otherwise, set them as strings.
            Setting this to false is useful if there is an error in the datetime that prevents it from being parsed.

        Returns
        -------
        df_dataset : DataFrame
            The dataframe containing the dataset with its dtypes set.
        """
        # Convert dtypes of date or date-time to datetimes
        if parse_dates:
            non_datetime_columns = {}
            datetime_columns = []
            for column, dtype in column_dtypes.items():
                if 'date' in dtype:
                    datetime_columns.append(column)
                # Allow for ints to accomodate NaNs. This is allowed
                # in pandas if we use Int vs int
                elif 'int' in dtype:
                    non_datetime_columns[column] = dtype.capitalize()                   
                else:
                    non_datetime_columns[column] = dtype

            df_dataset = df_dataset.astype(non_datetime_columns)
            for datetime_column in datetime_columns:
                df_dataset[datetime_column] = pd.to_datetime(df_dataset[datetime_column])

            return df_dataset

        # If not parsing dates, set the dates as strings
        else:
            for column, dtype in column_dtypes.items():
                if 'date' in dtype:
                    column_dtypes[column] = 'string'
                elif 'int' in dtype:
                    column_dtypes[column] = dtype.capitalize()                   

            df_dataset = df_dataset.astype(column_dtypes)
            return df_dataset
 