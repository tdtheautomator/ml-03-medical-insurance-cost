import os
import sys
from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging
from dataclasses import asdict
from datetime import datetime

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.helper.ml_metrics.metrics import regression_metrics
from src.helper.mlflow.client import get_best_run_id, get_exp_info_from_run_id

import mlflow
import src.vars as vars
from mlflow.models import infer_signature


#function for evaluating models for best parameters using grid search
def evaluate_reg_model_perf(X_train, y_train,X_test,y_test,models,params,searcher,track_in_mlflow:bool=False,log_model_in_mlflow:bool=False,register_best_model_in_mlflow:bool=False):
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
            #(mlflow_run_id, mlflow_run_name) = track_experiment(metrics=performance_metrics,params=s.best_params_,tags=tags)
            if track_in_mlflow:
                logging.info("tracking test predection metrics in mlflow")
                tags={
                    "Algorithm" : model_name
                }
                run_name= f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
                mlflow.set_experiment(vars.MLFLOW_EXP_NAME)
                mlflow.set_tracking_uri(vars.MLFLOW_TRACKING_URI)
                with mlflow.start_run(run_name=run_name) as run:
                    if (performance_metrics):
                        logging.info("registering metrics mlflow")
                        mlflow.log_metrics(performance_metrics)
                    if (tags):
                        logging.info("adding tags in mlflow")
                        mlflow.set_tags(tags)
                    if (s.best_params_):
                        logging.info("registering parameters mlflow")
                        best_params=s.best_params_
                        mlflow.log_params(best_params)
                    if log_model_in_mlflow:
                        logging.info("logging model in mlflow")
                        mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path=vars.MLFLOW_EXP_NAME.replace(' ','_').lower(),
                        signature=infer_signature(X_train,model.predict(X_train))
                    )
                    run_id = run.info.run_id
                    experiment_id = run.info.experiment_id
                    #r2Score = performance_metrics['R2Score']
                    logging.info(f"Experiment ID: {experiment_id},  Mlflow Run ID: {run_id}, Mlflow Run Name: {run_name}")
            report[list(models.keys())[i]] = performance_metrics,s.best_params_
        logging.info("final model performance report after hyper tuning")
        logging.info(report)
        if register_best_model_in_mlflow:
            logging.info("registering best model in mlflow")
            (best_run_id,best_run_name) = get_best_run_id(experiment_id,'R2Score')
            best_model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(model_uri=best_model_uri, name=vars.MLFLOW_REG_MODLE_NAME, tags={"environment": "development" })
            logging.info(f"Mlflow Best Run ID : {best_run_id}, Mlflow Best Run Name : {best_run_name}, MLFlow Best Model URI : {best_model_uri}")
        return report
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)