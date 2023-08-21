from src.utils import (filter_variables_by_completion_rate, 
                         split_dataset, 
                         remove_single_modality_categorical_variables,
                         save_object_with_dill, save_dataset, 
                         load_dataset)

from ..src.make_dataset import load_data
from pathlib import Path
import os
import pandas as pd


data = load_data("house_prices")


def test_filter_variables_by_completion_rate():
    """
    Test the filter_variables_by_completion_rate function to verify if it correctly filters out
    variables with completion rates below the minimum threshold.
    """

    filtered_data = filter_variables_by_completion_rate(data)

    assert filtered_data.shape[1] >= 50  # Expected number of columns after filtering
    assert "salecondition" in filtered_data.columns  # Expected column to be present
    assert "miscfeature" not in filtered_data.columns  # Expected column to be removed
    assert "poolqc" not in filtered_data.columns  # Expected column to be removed


def test_remove_single_modality_categorical_variables():
    """
    Test the remove_single_modality_categorical_variables function to verify if it correctly removes
    categorical variables with a single modality.
    """

    filtered_data = remove_single_modality_categorical_variables(data)

    assert filtered_data.shape[1] >= 50  # Expected number of columns after filtering


def test_split_dataset():
    """
    Test the split_dataset function to verify if it correctly splits the data into training and test sets,
    while maintaining consistency in the number of rows and columns.
    """

    X_train, X_test, y_train, y_test = split_dataset(data)

    assert X_train.shape[0] == y_train.shape[0]  # Ensure matching number of samples
    assert X_test.shape[0] == y_test.shape[0]  # Ensure matching number of samples
    assert X_train.shape[1] == X_test.shape[1]  # Ensure matching number of features


def test_save_object_with_dill(tmpdir):
    """
    Test the save_object_with_dill function to verify if it properly saves an object using the dill module.

    Args:
        tmpdir: A temporary directory provided by pytest to ensure isolated file operations during testing.
    """

    object_path = os.path.join(tmpdir, "test_object.pkl")
    sample_object = {"data": [1, 2, 3]}
    
    save_object_with_dill(sample_object, object_path)
    assert Path(object_path).is_file()


def test_save_and_load_dataset(tmpdir):
    """
    Test the save_dataset and load_dataset functions to verify if they correctly save and load a dataset.

    Args:
        tmpdir: A temporary directory provided by pytest to ensure isolated file operations during testing.
    """

    filename = "test_dataset"
    sample_data = [1, 2, 3, 4]
    save_dataset(sample_data, filename)
    
    loaded_data = load_dataset(filename)
    assert loaded_data == sample_data
