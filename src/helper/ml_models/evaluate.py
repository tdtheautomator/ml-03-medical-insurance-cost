import os
import sys
from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging
from dataclasses import asdict

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from src.helper.ml_metrics.metrics import regression_metrics
from src.helper.mlflow.track import track_experiment

#function for evaluating models for best parameters using grid search
def evaluate_reg_model_perf(X_train, y_train,X_test,y_test,models,params,searcher):
    try:
        report = {}
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]
            if searcher == "gsv":
                logging.info(f"evaluating {model_name} using GridSearchCV")
                logging.info(f"Using Parameters : {param}")
                s = GridSearchCV(model,param)
            else:
                logging.info(f"evaluating {model_name} using RandomSearchCV")
                logging.info(f"Using Parameters : {param}")
                s = RandomizedSearchCV(model,param)
            s.fit(X_train,y_train)
            model.set_params(**s.best_params_)
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            performance_metrics = regression_metrics(y_test,y_test_pred)
            p_metrics = asdict(performance_metrics)
            logging.info("loading test predection metrics to mlflow")
            (mlflow_run_id, mlflow_run_name) = track_experiment(metrics=performance_metrics,params=s.best_params_)
            logging.info(f"Mlflow Run ID: {mlflow_run_id}, Mlflow Run Name: {mlflow_run_name}")
            logging.info("performing predection on training data (optional)")
            report[list(models.keys())[i]] = p_metrics,s.best_params_
            logging.info("final model performance report after hyper tuning")
            logging.info(report)
        return report
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)