import os
import sys

from datetime import datetime
from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging
from dataclasses import asdict

import mlflow
import src.vars as vars
from mlflow.models import infer_signature
from urllib.parse import urlparse


def track_experiment(metrics,tags=[],params=[],log_model:bool=False):
    try:
        #metrics=asdict(metrics)
        #run_name= f"{exp_name} : {datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        run_name= f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        mlflow.set_experiment(vars.MLFLOW_EXP_NAME)
        mlflow.set_tracking_uri(vars.MLFLOW_TRACKING_URI)
        with mlflow.start_run(run_name=run_name) as run:
            if (metrics):
                logging.info("registering metrics mlflow")
                mlflow.log_metrics(metrics)
            if (tags):
                logging.info("registering tags in mlflow")
                mlflow.set_tags(tags)
            if (params):
                logging.info("registering parameters mlflow")
                mlflow.log_params(params)
            if log_model:
                logging.info("registering model in mlflow")
            run_id = run.info.run_id
        return run_id, run_name
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)