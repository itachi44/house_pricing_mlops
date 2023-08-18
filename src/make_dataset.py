"""Data collection"""
from typing import Optional

import pandas as pd
from loguru import logger
from sklearn.datasets import fetch_openml


def load_data(dataset_name: str,
              column_to_lower: Optional[bool] = True,
              ) -> pd.DataFrame:
    """Load data from OpenML.

    Args:
        dataset_name (str): dataset name to load
        column_to_lower (Optional[bool]): default is True
            It True, we transform column names to lower
            Otherwise, we return the raw column names

    Returns:
        pd.DataFrame: data to use for training House price

    """
    logger.info(f"\n======================================================================="
                f"\nArgs: dataset name: {dataset_name} \ncolumn to lower: {column_to_lower}"
                f"\n=======================================================================")
    if dataset_name == "house_prices":
        dframe = fetch_openml(name=dataset_name, as_frame=True, version="active", target_column=None)
    data = dframe.data
    logger.info(f"Shape of raw input features: {data.shape}")
    logger.info(f"Full description of the dataset\n{dframe.DESCR}")
    if column_to_lower:
        data.columns = data.columns.str.lower()
    return data
