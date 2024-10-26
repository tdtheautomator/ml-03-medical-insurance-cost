import os
import sys
from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging
from dataclasses import asdict
from datetime import datetime

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.helper.ml_metrics.metrics import regression_metrics
from src.helper.mlflow.track import track_experiment

import mlflow
import src.vars as vars
from mlflow.models import infer_signature
from urllib.parse import urlparse


#function for evaluating models for best parameters using grid search
def evaluate_reg_model_perf(X_train, y_train,X_test,y_test,models,params,searcher,track_in_mlflow:bool,log_model_in_mlflow:bool,register_model_in_mlflow:str="None"):
    try:
        report = {}
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]
            if searcher == "gsv":
                logging.info(f"evaluating {model_name} using GridSearchCV")
                logging.info(f"hyper tuning parameters : {param}")
                s = GridSearchCV(model,param)
            else:
                logging.info(f"evaluating {model_name} using RandomSearchCV")
                logging.info(f"Using Parameters : {param}")
                s = RandomizedSearchCV(model,param)
            s.fit(X_train,y_train)
            model.set_params(**s.best_params_)
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            performance_metrics = asdict(regression_metrics(y_test,y_test_pred))
            tags={
                    "Algorithm" : model_name
                }
            (mlflow_run_id, mlflow_run_name) = track_experiment(metrics=performance_metrics,params=s.best_params_,tags=tags)
            #if track_in_mlflow:
            #    logging.info("tracking test predection metrics in mlflow")
            #    tags={
            #        "Algorithm" : model_name
            #    }
            #    run_name= f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
            #    mlflow.set_experiment(vars.MLFLOW_EXP_NAME)
            #    mlflow.set_tracking_uri(vars.MLFLOW_TRACKING_URI)
            #    with mlflow.start_run(run_name=run_name) as run:
            #        if (performance_metrics):
            #            logging.info("registering metrics mlflow")
            #            mlflow.log_metrics(performance_metrics)
            #        if (tags):
            #            logging.info("adding tags in mlflow")
            #            mlflow.set_tags(tags)
            #        if (s.best_params_):
            #            logging.info("registering parameters mlflow")
            #            params=s.best_params_
            #            mlflow.log_params(params)
            #        if log_model_in_mlflow:
            #            logging.info("logging model in mlflow")
            #            mlflow.sklearn.log_model(
            #            sk_model=model_name,
            #            artifact_path=vars.MLFLOW_EXP_NAME.replace(' ','_').lower(),
            #            signature=infer_signature(X_train,model.predict(X_train))
            #        )
            #        run_id = run.info.run_id
            #        logging.info(f"Mlflow Run ID: {run_id}, Mlflow Run Name: {run_name}")
            report[list(models.keys())[i]] = performance_metrics,s.best_params_
            logging.info("final model performance report after hyper tuning")
            logging.info(report)
        return report
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)