# OpenFEMA
*A python package for easily reading OpenFEMA datasets.*

This package is intended to facilitate better access and promote more rapid analysis of OpenFEMA data.
It is a wrapper around the OpenFEMA API, which allows for direct reading of datasets into a [`pandas`](https://pandas.pydata.org/pandas-docs/stable/index.html) dataframe without a user having to manually set up the API URL to the data.
Users can either read in the full dataset or read in a subset of a dataset.

Please report any bugs, suggest enhancements, or ask questions by creating an [issue](https://github.com/kjdoore/OpenFEMA/issues).
  
## Installation
First clone the repo to your machine:
```sh
git clone https://github.com/kjdoore/OpenFEMA.git
```
Then it can be installed locally using pip
```sh
pip install ./OpenFEMA
```

## Basic Usage
At a basic level, a dataset can be read with:

```python
from OpenFEMA import OpenFEMA

# Initialize the OpenFEMA session
openfema = OpenFEMA()

# List available datasets
print(openfema.list_datasets())

# Read in a dataset
df_femaregions = openfema.read_dataset('FemaRegions')
```

For more details and how to subset a dataset, see the [tutorial notebook](https://github.com/kjdoore/OpenFEMA/blob/main/examples/tutorial.ipynb).
