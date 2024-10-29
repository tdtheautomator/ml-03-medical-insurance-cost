import os
import sys
import time
from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging
from src.config.artifacts_schema import RegressionModelMetrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error


#function for getting performance metrics of regression models
def regression_metrics(true, predicted)->RegressionModelMetrics:
    try:
        logging.info("getting performance metrics")
        o_mae = round(mean_absolute_error(true, predicted),4)
        o_mse = round(mean_squared_error(true, predicted),4)
        o_rmse = round(root_mean_squared_error(true, predicted),4)
        o_r2_score = round(r2_score(true, predicted),4)
        performance_metrics = RegressionModelMetrics(MAE=o_mae, MSE=o_mse,RMSE=o_rmse,R2Score=o_r2_score)
        return performance_metrics
    except Exception as e:
            raise CustomException(e,sys)