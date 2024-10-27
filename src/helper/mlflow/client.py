import os
import sys

from datetime import datetime
from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging
from dataclasses import asdict

import mlflow
import src.vars as vars
from mlflow.tracking import MlflowClient

def get_best_run_id(exp_id,metric_name):
    try:
        client = MlflowClient()
        runs = client.search_runs(experiment_ids=[exp_id], order_by=[f"metrics.{metric_name} DESC"])
        if runs:
            best_run = runs[0]
            return best_run.info.run_id, best_run.info.run_name
        else:
            return None
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)
    
def get_exp_info_from_run_id(run_id):
    try:
        client = MlflowClient()
        run = client.get_run(run_id)
        exp_id = run.info.experiment_id
        experiment = client.get_experiment(exp_id)
        exp_name = experiment.name
        return exp_id, exp_name
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)
