import numpy as np
import pandas as pd
from ..src.trainer import eval_metrics, define_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def test_eval_metrics():
    """
    Test the eval_metrics function to ensure it computes evaluation metrics accurately.

    This test uses example actual and predicted values and verifies that the metrics are correctly computed.
    """

    y_actual = np.array([3, 7, 2, 5])
    y_pred = np.array([2.8, 7.2, 2.5, 4.9])
    
    metrics = eval_metrics(y_actual, y_pred)
    
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "max_error" in metrics
    
    # Test specific values or ranges of values based on your data
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert metrics["r2"] >= -1 and metrics["r2"] <= 1
    assert metrics["max_error"] >= 0


def test_define_pipeline():
    """
    Test the define_pipeline function to ensure it constructs a pipeline with the specified steps.

    This test creates a pipeline using example transformer and estimator configurations and verifies
    the presence of the expected steps in the pipeline.
    """
    
    numerical_transformer = [StandardScaler()]
    categorical_transformer = [OneHotEncoder()]
    target_transformer = True
    estimator = RandomForestRegressor()
    
    pipeline = define_pipeline(numerical_transformer, categorical_transformer, target_transformer, estimator)
    
    # Test that the pipeline is created and contains the specified steps
    assert "preprocessor" in pipeline.named_steps
    assert "estimator" in pipeline.named_steps