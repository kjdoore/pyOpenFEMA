from __future__ import annotations

import pandas as pd

"""
Here is the list of possible API commands:

Command      Description                                                            Default
$allrecords  To specify that all records are to be returned when downloading data.  false (only records up to $top returned)
$count       To specify if a total entity count should be returned with the query.  false (the count is not returned)
$filename    To specify the download filename.                                      None
$filter      To filter or limit the results returned.                               All records up to $top returned
$format      To specify the format of the returned data.                            JSON
$metadata    Controls the presence of the metadata object in data set returns.      on (metadata is returned)
$orderby     Sort the returned data.                                                None
$select      To specify the fields to return.                                       All fields returned
$skip        Number of records to skip from the dataset.                            0
$top         Limit the number of records returned.                                  1,000 (maximum is 10,000)

Of these, we only need to implement $filter, $select, $skip, $top, and
$orderby in the API call as they are the only commands that subset the
data. (Technically, $orderby does not subset, but it can be used to sort
and get the top number of entries. So, we include it.) As the goal is
not to download data but to read it into python directly, we do not implement
$allrecords, $filename, or $format for user interaction as they are only for
specifying the file to read. As for $metadata, this can be determined externally
by calling the `dataset_info` method in this package instead. Finally, the $count
command can be done externally to the API call in pandas, and does not need
to be implemented in the API call.
"""


def generate_column_select_command(columns: list[str],
                                   column_dtypes: dict
                                   ) -> str:
    """
    Generate URL substring for column (select) command.

    Parameters
    ----------
    columns : list[str]
        A list of the columns to read.
    column_dtypes : dict
        A dictionary containing the dataset column names as the keys and dtype as the values.

    Returns
    -------
    select_url_substring : str
        The URL substring for the select command
    """
    if isinstance(columns, list):
        # Ensure all specified columns are in the dataset
        columns_bool = pd.Series(columns).isin(column_dtypes.keys())
        if columns_bool.all():
            command = '$select='
            field_names = ",".join(columns)
            return command + field_names
        else:
            raise ValueError('The specfied column(s) of '
                             f'{list(pd.Series(columns)[~columns_bool])} '
                             'for selecting do not exist in the dataset.')
    elif isinstance(columns, type(None)):
        return None
    else:
        raise TypeError('Columns must be a list of strings. '
                        f'Current type: {type(columns)}')


def generate_filter_command(filters: list[list[tuple]],
                            column_dtypes: dict
                            ) -> str:
    """
    Generate URL substring for filter command.

    Parameters
    ----------
    filters : list[list[tuple]]
        Filter to apply to the data.
        Filter syntax: [[(column, op, val), ...],...] where column is the column name;
        op is a string operator of 'eq', 'ne', 'gt', 'ge', 'lt', 'le', 'in', 'not',
        'substringof', 'endswith', 'startswith', 'contains', or 'geo.intersects'
        (See https://www.fema.gov/about/openfema/api#filter for details on each operator);
        and val is the limiting value(s).
        The innermost tuples are transposed into a set of filters applied through an `AND` operation.
        The outer list combines these sets of filters through an `OR` operation.
    column_dtypes : dict
        A dictionary containing the dataset column names as the keys and dtype as the values.

    Returns
    -------
    filter_url_substring : str
        The URL substring for the filter command
    """
    if isinstance(filters, list):
        # Ensure all specified columns are in the dataset
        columns = [filter_tuple[0] for filter_tuples in filters for filter_tuple in filter_tuples]
        columns_bool = pd.Series(columns).isin(column_dtypes.keys())

        if not columns_bool.all():
            raise ValueError('The specfied column(s) of '
                             f'{list(pd.Series(columns)[~columns_bool])} '
                             'for filtering do not exist in the dataset.')

        # Ensure all operators are valid
        mathmatical_operators = ['eq', 'ne', 'gt', 'ge', 'lt', 'le']
        in_operator = ['in']
        parenthetical_operators = ['substringof', 'endswith', 'startswith',
                                   'contains']
        geo_operator = ['geo.intersects']
        valid_operators = (mathmatical_operators
                           + in_operator
                           + parenthetical_operators
                           + geo_operator
                           + [f'not {parenthetical_operator}'
                              for parenthetical_operator in (parenthetical_operators + in_operator)])
        operators = [filter_tuple[1] for filter_tuples in filters for filter_tuple in filter_tuples]
        operators_bool = pd.Series(operators).isin(valid_operators)

        if not operators_bool.all():
            raise ValueError('The specfied operator(s) of '
                             f'{list(pd.Series(operators)[~operators_bool])} '
                             'for filtering are not valid.')
        command = '$filter='

        # Loop through filter tuples and format for URL
        filter_strings = []
        for filter_tuples in filters:
            filter_string = []
            for col, op, val in filter_tuples:
                if op in mathmatical_operators + in_operator + [f'not {in_operator}']:
                    if isinstance(val, str):
                        filter_string.append(f"{col} {op} '{val}'")
                    elif isinstance(val, bool):
                        filter_string.append(f"{col} {op} {str(val).lower()}")
                    elif isinstance(val, (list, tuple)):
                        joined_val = ','.join([f"'{str(value)}'" for value in val])
                        filter_string.append(f"{col} {op} ({joined_val})")
                    else:
                        filter_string.append(f'{col} {op} {val}')

                elif (op in parenthetical_operators) or (op in [
                    f'not {parenthetical_operator}' for parenthetical_operator in parenthetical_operators
                ]):
                    if isinstance(val, str):
                        filter_string.append(f"{op}({col},'{val}')")
                    elif isinstance(val, (list, tuple)):
                        joined_val = ','.join([f"'{str(value)}'" for value in val])
                        filter_string.append(f"{op}({col},({joined_val}))")
                    else:
                        filter_string.append(f"{op}({col},{val})")

                elif op in geo_operator:
                    if isinstance(val, str):
                        filter_string.append(f"{op}({col},geography'{val}')")
                    elif isinstance(val, (list, tuple)):
                        joined_val = ','.join([f"'{str(value)}'" for value in val])
                        filter_string.append(f"{op}({col},({joined_val}))")

            filter_strings.append(' and '.join(filter_string))

        filter_command = ' or '.join([f'({filter_string})' for filter_string in filter_strings])

        # Replace spaces and apostrophes with http codes
        filter_command = filter_command.replace(' ', '%20')
        filter_command = filter_command.replace("'", "%27")

        return command + filter_command
    elif isinstance(filters, type(None)):
        return None
    else:
        raise TypeError('Filters must be a list of tuples. '
                        f'Current type: {type(filters)}')


def generate_sortby_command(sort_by: list[tuple],
                            column_dtypes: dict
                            ) -> str:
    """
    Generate URL substring for sort_by (orderby) command.

    Parameters
    ----------
    sort_by : list[tuple]
        The sorting to apply to the data.
        Sort syntax: [(column, ascending), ...]  where ascending is a boolen indicating
        if the sort should be in ascending order (True is ascending, False is descending).
        The order of each tuple expresses sorting order, with the first tuple specifying the column that is sorted first.
    column_dtypes : dict
        A dictionary containing the dataset column names as the keys and dtype as the values.

    Returns
    -------
    orderby_url_substring : str
        The URL substring for the orderby command
    """
    if isinstance(sort_by, list):
        # Ensure all specified columns are in the dataset
        columns = [column_tuple[0] for column_tuple in sort_by]
        columns_bool = pd.Series(columns).isin(column_dtypes.keys())

        if columns_bool.all():
            command = '$orderby='

            sortby_order = [column_tuple[0]
                            if column_tuple[1]
                            else f'{column_tuple[0]}%20desc'
                            for column_tuple in sort_by]
            orderby = ",".join(sortby_order)
            return command + orderby
        else:
            raise ValueError('The specfied column(s) of '
                             f'{list(pd.Series(columns)[~columns_bool])} '
                             'for sorting do not exist in the dataset.')
    elif isinstance(sort_by, type(None)):
        return None
    else:
        raise TypeError('Sort_by must be a list of tuples. '
                        f'Current type: {type(sort_by)}')


def generate_top_command(top: int) -> str:
    """
    Generate URL substring for top command.

    Parameters
    ----------
    top : int
        The number of records returned.

    Returns
    -------
    top_url_substring : str
        The URL substring for the top command
    """
    if isinstance(top, int):
        if (top > 0) and (top <= 10000):
            return f'$top={top}'
        else:
            raise ValueError('Top must be a positive value less than or '
                             f'equal to 10000. Current value: {top}')
    elif isinstance(top, type(None)):
        return None
    else:
        raise TypeError('Top must be of type int. '
                        f'Current type: {type(top)}')


def generate_skip_command(skip: int) -> str:
    """
    Generate URL substring for skip command.

    Parameters
    ----------
    skip : int
        The number of records to skip in the dataset.

    Returns
    -------
    skip_url_substring : str
        The URL substring for the skip command
    """
    if isinstance(skip, int):
        if skip >= 0:
            return f'$skip={skip}'
        else:
            raise ValueError('Skip must be a non-negative value. '
                             f'Current value: {skip}')
    elif isinstance(skip, type(None)):
        return None
    else:
        raise TypeError('Skip must be of type int. '
                        f'Current type: {type(skip)}')
