from typing import Dict, Union
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer, make_column_selector, TransformedTargetRegressor
from sklearn.metrics import (r2_score,
                             mean_squared_error,
                             mean_absolute_percentage_error,
                             mean_absolute_error,
                             max_error,
                            )
from sklearn.pipeline import Pipeline,make_pipeline



def eval_metrics(y_actual: Union[pd.DataFrame, pd.Series, np.ndarray],
                 y_pred: Union[pd.DataFrame, pd.Series, np.ndarray]
                 ) -> Dict[str, float]:
    """ Compute evaluation metrics

    Args:
        y_actual: Ground truth (correct) target values
        y_pred: Estimated target values.

    Returns:
        Dict[str, float]: dictionary of evaluation metrics.
            Expected keys are: "rmse", "mae", "r2", "max_error"

    """
    # Root mean squared error
    rmse = mean_squared_error(y_actual, y_pred, squared=False)
    # mean absolute error
    mae = mean_absolute_error(y_actual, y_pred)
    # R-squared: coefficient of determination
    r2 = r2_score(y_actual, y_pred)
    # max error: maximum value of absolute error (y_actual - y_pred)
    maxerror = max_error(y_actual, y_pred)
    return {"rmse": rmse,
            "mae": mae,
            "r2": r2,
            "max_error": maxerror
           }



def define_pipeline(numerical_transformer: list,
                    categorical_transformer: list,
                    target_transformer,
                    estimator: Pipeline,
                    **kwargs: dict) -> Pipeline:
    """ Define pipeline for modeling

    Args:
        **kwargs:

    Returns:
        Pipeline: sklearn pipeline
    """
    numerical_transformer = make_pipeline(*numerical_transformer)

    categorical_transformer = make_pipeline(*categorical_transformer)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, make_column_selector(dtype_include=["number"])),
            ("cat", categorical_transformer, make_column_selector(dtype_include=["object", "bool"])),
        ],
        remainder="drop",  # non-specified columns are dropped
        verbose_feature_names_out=False,  # will not prefix any feature names with the name of the transformer
    )
    # Append regressor to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    if target_transformer:
        model_pipe = Pipeline(steps=[("preprocessor", preprocessor),
                                     ("estimator", TransformedTargetRegressor(regressor=estimator,
                                                                              func=np.log,
                                                                              inverse_func=np.exp))])
    
    else:
        
        model_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
        
    logger.info(f"{model_pipe}")
    return model_pipe