import os
import sys
import mlflow
from datetime import datetime
from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging
from dataclasses import asdict

from mlflow.models import infer_signature
from urllib.parse import urlparse



def track_experiment(exp_name,metrics,url,tags=[],parms=[]):
    try:
        metrics=asdict(metrics)
        #run_name= f"{exp_name} : {datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        run_name= f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        mlflow.set_experiment(exp_name)
        if url == "local":
            mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        else:
            mlflow.set_tracking_uri(url)
        with mlflow.start_run(run_name=run_name) as run:
            if (metrics):
                logging.info("registering mlflow metrics")
                mlflow.log_metrics(metrics)
            if (tags):
                logging.info("registering mlflow tags")
                mlflow.log_parms(tags)
            if (parms):
                logging.info("registering mlflow parameters")
                mlflow.log_parms(parms)
            run_id = run.info.run_id
        return run_id, run_name
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)