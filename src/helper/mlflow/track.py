import os
import sys
import mlflow

from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging
from dataclasses import asdict

from mlflow.models import infer_signature
from urllib.parse import urlparse



def track_experiment(exp_name,best_model,metrics,url):
    try:
        metrics=asdict(metrics)
        mlflow.set_experiment(exp_name)
        if url == "local":
            mlflow.set_tracking_uri("http://127.0.0.1:5000/")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        else:
            mlflow.set_tracking_uri(url)
        with mlflow.start_run():
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            #if tracking_url_type_store != "file":
            #    mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model)
            #else:
            #    mlflow.sklearn.log_model(best_model, "model")
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)