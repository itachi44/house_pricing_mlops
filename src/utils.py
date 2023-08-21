import pandas as pd
import missingno as msno
import dill
import pickle



from typing import List
from loguru import logger
from settings.params import MODEL_PARAMS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from pathlib import Path
from ..settings.params import DATA_DIR, DATA_DIR_INPUT, MODEL_DIR


def filter_variables_by_completion_rate(
        data: pd.DataFrame
        )-> pd.DataFrame:

    """Filter variables by completion rate (0.5).

    Args:
        data (Dataframe): dataset in which we realize filter

    Returns:
        pd.DataFrame: data to use for training House price

    """

    missing_values = data.isnull().mean()  # Calculate the percentage of missing values for each column

    logger.info(f"\n======================================================================="
                f"\n percentage of missing values: {missing_values}"
                f"\n=======================================================================")
    incomplete_columns = missing_values[missing_values > MODEL_PARAMS["MIN_COMPLETION_RATE"]].index  # Filter columns with completion rate < MIN_COMPLETION_RATE

    filtered_data = data.drop(incomplete_columns, axis=1)  # Drop the columns with low completion rate

    return filtered_data


def remove_single_modality_categorical_variables(
        data: pd.DataFrame
    )-> pd.DataFrame:
    
    """remove categorical variables that have only one modality.

    Args:
        data (Dataframe): dataset in which we realize filter

    Returns:
        pd.DataFrame: data to use for training House price

    """

    categorical_features = data.select_dtypes(include="object").columns
    variables_to_remove = []
    
    for feature in categorical_features:
        unique_values = data[feature].nunique()  # Count the number of unique values for each categorical feature
        logger.info(f"\n unique values count for {feature}: {unique_values}")
        if unique_values == 1:  # Check if the feature has only one modality
            variables_to_remove.append(feature)
    
    filtered_data = data.drop(variables_to_remove, axis=1)  # Drop the variables with a single modality
    
    return filtered_data


def split_dataset(
        data: pd.DataFrame
        )-> pd.DataFrame:
    
    # selected_features = data.columns.drop(MODEL_PARAMS["TARGET"])

    FEATURES = set(MODEL_PARAMS["FEATURES"])
    data_columns = set(data.columns)
    intersection_result = FEATURES.intersection(data_columns)
    selected_columns_index = pd.Index(intersection_result, dtype='object')


    X_train, X_test, y_train, y_test = train_test_split(
        data[selected_columns_index],
        data[MODEL_PARAMS["TARGET"]], 
        test_size=MODEL_PARAMS["TEST_SIZE"],
        random_state=23
    )
    
    logger.info(f"\nx train: {X_train.shape}\nY train: {y_train.shape} \n" f"X test: {X_test.shape}\nY test: {y_test.shape}")
    return X_train, X_test, y_train, y_test


def save_object_with_dill(object_to_save, object_path):
    """
    Sauvegarde un objet en utilisant le module dill.
    
    Args:
        object_to_save: L'objet que vous souhaitez sauvegarder.
        object_path (Path): Le chemin complet vers l'emplacement où l'objet sera sauvegardé.
    """

    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(object_path, "wb") as f:
        dill.dump(object_to_save, f)


def save_dataset(dataset, filename):
    """
    Sauvegarde le dataset prétraité dans l'emplacement DATA_DIR_INPUT.
    
    Args:
        dataset (object): Le dataset prétraité que vous souhaitez sauvegarder.
        filename (str): Le nom du fichier de sauvegarde (sans extension).
    """

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_DIR_INPUT.exists():
        DATA_DIR_INPUT.mkdir(parents=True, exist_ok=True)

    save_path = DATA_DIR_INPUT / (filename + ".pkl")
    
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
        
    print(f"Dataset sauvegardé avec succès sous {save_path}")


def load_dataset(filename):
    """
    Charge le dataset prétraité depuis l'emplacement DATA_DIR_INPUT.
    
    Args:
        filename (str): Le nom du fichier de sauvegarde (sans extension).
        
    Returns:
        dataset (object): Le dataset prétraité chargé depuis le fichier.
    """

    load_path = DATA_DIR_INPUT / (filename + ".pkl")
    
    with open(load_path, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset