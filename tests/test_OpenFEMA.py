import pytest
from pandas import DataFrame

from pyOpenFEMA import OpenFEMA

# Initialize an OpenFEMA object for reuse and faster testing
openfema = OpenFEMA()


def test_list_datasets():
    datasets = openfema.list_datasets()

    assert isinstance(datasets, list)
    assert len(datasets) > 0


@pytest.mark.parametrize('dataset', openfema.list_datasets())
def test_dataset_info(dataset):
    info_dict = openfema.dataset_info(dataset)

    assert isinstance(info_dict, dict)
    assert info_dict['name'] == dataset
    assert 'columns' in info_dict.keys()
    assert 'distribution' in info_dict.keys()
    assert 'webService' in info_dict.keys()


@pytest.mark.parametrize('dataset, columns, filters, sort_by, top, skip',
                         [('FemaRegions', None, None, None, None, None),
                          ('FemaRegions', ['name', 'region'], [[('region', 'lt', 5)]], [('name', True)], None, None),
                          ('DeclarationDenials', ['declarationRequestNumber', 'region', 'stateAbbreviation'],
                           [[('declarationRequestNumber', 'gt', 10)]], [('declarationRequestNumber', False)], 5, 4)])
def test_read_dataset(dataset, columns, filters, sort_by, top, skip):
    df = openfema.read_dataset(dataset, columns, filters, sort_by, top, skip)

    assert isinstance(df, DataFrame)
    if columns is not None:
        if 'geometry' in list(df.columns):
            columns += ['geometry']
        assert len(df.columns) == len(columns)
        assert list(df.columns) == columns
    if top is not None:
        assert len(df) == top
