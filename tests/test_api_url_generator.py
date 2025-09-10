import pytest

from pyOpenFEMA.api_url_command_generators import *


@pytest.fixture
def column_dtypes():
    column_dtypes = {
        'test1': 'int',
        'test2': 'float',
        'test3': 'string',
        'test4': 'string',
    }

    return column_dtypes


@pytest.mark.parametrize('columns', [
    ['test1'], ['test1', 'test2'], ['test1', 'test2', 'test3', 'test4']
])
def test_generate_column_select_command(columns, column_dtypes):
    select_command = generate_column_select_command(columns, column_dtypes)

    select_command_manual = '$select=' + ','.join(columns)

    assert isinstance(select_command, str)
    assert select_command == select_command_manual


@pytest.mark.parametrize('filters', [
    [[('test1', 'eq', 10)]],
    [[('test1', 'lt', 5)], [('test2', 'gt', 7)]],
    [[('test1', 'ne', 2), ('test2', 'le', 1)], [('test3', 'startswith', 'test'), ('test4', 'endswith', 'ing')]],
    [[('test3', 'not substringof', 'test')]],
])
def test_generate_filter_command(filters, column_dtypes):
    filter_command = generate_filter_command(filters, column_dtypes)

    filter_command_manual = '$filter=' + '%20or%20'.join(
        ['('+'%20and%20'.join(
            [f"{and_filter[0]}%20{and_filter[1]}%20{and_filter[2]}"
             if column_dtypes[and_filter[0]] != 'string'
             else f"{and_filter[1]}({and_filter[0]},%27{and_filter[2]}%27)"
             for and_filter in or_filters])+')'
         for or_filters in filters]
    ).replace(' ', '%20')

    assert isinstance(filter_command, str)
    assert filter_command == filter_command_manual


@pytest.mark.parametrize('sort_by', [
    [('test1', False)],
    [('test1', True), ('test2', False)],
    [('test1', True), ('test2', True), ('test3', False), ('test4', True)]
])
def test_generate_sortby_command(sort_by, column_dtypes):
    sortby_command = generate_sortby_command(sort_by, column_dtypes)

    sortby_command_manual = '$orderby=' + ','.join(
        [column_tuple[0] if column_tuple[1] else f'{column_tuple[0]}%20desc'
         for column_tuple in sort_by]
    )

    assert isinstance(sortby_command, str)
    assert sortby_command == sortby_command_manual


@pytest.mark.parametrize('top', [1, 5, 100, 1000, 10000])
def test_generate_top_command(top):
    top_command = generate_top_command(top)

    assert isinstance(top_command, str)
    assert top_command == f'$top={top}'


@pytest.mark.parametrize('skip', [0, 10, 100, 1000])
def test_generate_skip_command(skip):
    skip_command = generate_skip_command(skip)

    assert isinstance(skip_command, str)
    assert skip_command == f'$skip={skip}'
