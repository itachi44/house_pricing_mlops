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
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from mlflow.models import infer_signature



try:
    from ..settings.params import ESTIMATORS, EXECUTION_DATE
except Exception:
    from settings.params import ESTIMATORS, EXECUTION_DATE



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
        model_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                     ("estimator", TransformedTargetRegressor(regressor=estimator,
                                                                              func=np.log,
                                                                              inverse_func=np.exp))])
    
    else:
        
        model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
        
    logger.info(f"{model_pipeline}")
    return model_pipeline


# TODO : écrire le docstring

def train_models(
        data, 
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        categorical_features,
        numerical_features,
        artifact_path,
        experiment_id,
        target_transformer
        ):
    
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=ESTIMATORS),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=ESTIMATORS)
    }

    model_results = {}


    for model_name, model in models.items():

        with mlflow.start_run(
            run_name=f"{EXECUTION_DATE.strftime('%Y%m%d_%H%m%S')}-house_pricing",
            experiment_id=experiment_id,
            tags={"version": "v1", "priority": "P1"},
            description="house price modeling",) as mlf_run:

            # Model definition
            reg = define_pipeline(numerical_transformer=[SimpleImputer(strategy="median"),
                                                        RobustScaler()],
                                categorical_transformer=[SimpleImputer(strategy="constant", fill_value="undefined"),
                                                        OneHotEncoder(drop="if_binary", handle_unknown="ignore")],
                                target_transformer=target_transformer,
                                estimator=model
                            )

            reg.fit(X_train, y_train)

            # Evaluate Metrics
            y_train_pred = reg.predict(X_train)
            y_test_pred = reg.predict(X_test)
            train_metrics = eval_metrics(y_train , y_train_pred)
            test_metrics = eval_metrics(y_test , y_test_pred)


            logger.info(f"Model: {model_name}")
            logger.info(f"run_id: {mlf_run.info.run_id}")
            logger.info(f"version tag value: {mlf_run.data.tags.get('version')}")
            logger.info("--")
            logger.info(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
            logger.info(f"Train: {train_metrics}")
            logger.info(f"Test: {test_metrics}")

            # Log parameter, metrics, and model to MLflow

            if model_name!="LinearRegression":
                mlflow.log_param("n_estimators", ESTIMATORS)
            mlflow.log_param("model_name", model_name)

            # Infer model signature
            # Converting train features into a DataFrame
            X_train_df = pd.DataFrame(data=X_train, columns=data.columns)

            X_train_df.loc[:, categorical_features] = X_train_df.loc[:, categorical_features].astype(str)
            X_train_df.loc[:, numerical_features] = X_train_df.loc[:, numerical_features].astype(str)

            signature = infer_signature(model_input=X_train_df,model_output=y_train_pred)


            # Log parameter, metrics, and model to MLflow
            for group_name, set_metrics in [("train", train_metrics),("test", test_metrics),]:

                for metric_name, metric_value in set_metrics.items():
                    mlflow.log_metric(f"{group_name}_{metric_name}", metric_value)

            model_results[model_name] = {
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "run_id": mlf_run.info.run_id
            }


            mlflow.sklearn.log_model(reg, artifact_path=artifact_path,signature=signature, registered_model_name=f"{model_name}Model")

    return model_results