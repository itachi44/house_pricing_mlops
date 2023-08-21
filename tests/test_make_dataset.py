# test_make_dataset.py

import pandas as pd
import pytest
from ..src.make_dataset import load_data

# Test the load_data function
def test_load_data():
    """
    Test the load_data function.

    Validates if the load_data function correctly loads data from the specified dataset.
    """
    dataset_name = "house_prices"
    data = load_data(dataset_name)
    assert isinstance(data, pd.DataFrame)
    assert data.shape[0] > 0
    assert data.shape[1] > 0


# Test the load_data function with column_to_lower=True
def test_load_data_with_column_to_lower_false():
    """
    Test the load_data function with column_to_lower set to True.

    Validates if the load_data function correctly loads data and keeps column names in lowercase.
    """
    dataset_name = "house_prices"
    data = load_data(dataset_name, column_to_lower=True)
    assert isinstance(data, pd.DataFrame)
    assert data.shape[0] > 0
    assert data.shape[1] > 0
    assert all(col.islower() for col in data.columns)  # Check if columns are in lowercase


# Test the load_data function with invalid dataset name
def test_load_data_with_invalid_dataset():
    """
    Test the load_data function with an invalid dataset name.

    Validates if the load_data function raises a ValueError when an unrecognized dataset name is provided.
    """
    dataset_name = "invalid_dataset"
    with pytest.raises(ValueError):
        data = load_data(dataset_name)

# Add more tests as needed

if __name__ == '__main__':
    pytest.main()