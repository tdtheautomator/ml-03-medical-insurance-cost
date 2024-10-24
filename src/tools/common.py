#file used to create code for common functions
import os
import sys
import time
import pickle
import dill
import pandas as pd
import numpy as np
import yaml
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from src.tools.custom_exception import CustomException
from src.tools.custom_logger import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse


#function for saving pickle file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)

 #function for loading pickle file   

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)

 #function for evaluating models for best parameters using grid search

def evaluate_model_best_param_gsv(X_train, y_train,X_test,y_test,models,params):
    logging.info("using grid search")
    performance_metrics = {}
    m_name=[]
    mae=[]
    mse=[]
    rmse=[]
    r2=[]
    dur=[]
    try:
        report = {}
        for i in range(len(list(models))):
            t1=time.time()
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]
            logging.info(f'evaluating {model_name}')
            gs = GridSearchCV(model,param)
            gs.fit(X_train,y_train)
            logging.info(f'best estimator : {gs.best_estimator_}')
            logging.info(f'best score : {gs.best_score_}')
            logging.info(f'best params : {gs.best_params_}')
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            performance_metrics = get_model_performance_metrics(y_test, y_test_pred)
            test_model_score = performance_metrics[3]
            report[list(models.keys())[i]] = test_model_score
            t2=time.time()
            logging.info(f'{model_name} | MAE : {performance_metrics[0]}, MSE : {performance_metrics[1]}, RMSE : {performance_metrics[2]}, R2 Score : {performance_metrics[3]}')
            m_name.append(model_name)
            mae.append(performance_metrics[0])
            mse.append(performance_metrics[1])
            rmse.append(performance_metrics[2])
            r2.append(performance_metrics[3])
            dur.append(t2-t1)
        performance_metrics_report ={'Model Name': m_name, 'MAE': mae,'MSE': mse, 'RMSE': rmse, 'R2 Score': r2,'Duration': dur}
        df_ModelPerformance = pd.DataFrame(performance_metrics_report)
        filepath = f'./outputs/{time.strftime("%Y%m%d_%H%M%S")}_ModelPerformance.csv'
        df_ModelPerformance.to_csv(filepath)
        return report
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)

#function for evaluating models for best parameters using random grid search
def evaluate_model_best_param_rsv(X_train, y_train,X_test,y_test,models,params):
    logging.info("using randomized search")
    performance_metrics = {}
    m_name=[]
    mae=[]
    mse=[]
    rmse=[]
    r2=[]
    dur=[]
    try:
        report = {}
        for i in range(len(list(models))):
            t1=time.time()
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]
            logging.info(f'evaluating {model_name}')
            rgs = RandomizedSearchCV(model,param)
            rgs.fit(X_train,y_train)
            logging.info(f'best estimator : {rgs.best_estimator_}')
            logging.info(f'best score : {rgs.best_score_}')
            logging.info(f'best params : {rgs.best_params_}')
            model.set_params(**rgs.best_params_)
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            performance_metrics = get_model_performance_metrics(y_test, y_test_pred)
            test_model_score = performance_metrics[3]
            report[list(models.keys())[i]] = test_model_score
            t2=time.time()
            logging.info(f'{model_name} | MAE : {performance_metrics[0]}, MSE : {performance_metrics[1]}, RMSE : {performance_metrics[2]}, R2 Score : {performance_metrics[3]}')
            m_name.append(model_name)
            mae.append(performance_metrics[0])
            mse.append(performance_metrics[1])
            rmse.append(performance_metrics[2])
            r2.append(performance_metrics[3])
            dur.append(t2-t1)
        performance_metrics_report ={'Model Name': m_name, 'MAE': mae,'MSE': mse, 'RMSE': rmse, 'R2 Score': r2,'Duration': dur}
        df_ModelPerformance = pd.DataFrame(performance_metrics_report)
        df_ModelPerformance.sort_values(['RMSE','R2 Score'], ascending=[True, False], inplace=True)
        filepath = f'./outputs/{time.strftime("%Y%m%d_%H%M%S")}_ModelPerformance.csv'
        df_ModelPerformance.to_csv(filepath)
        return report
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)

#function for getting performance metrics of regression models
def get_model_performance_metrics(true, predicted):
    logging.info("getting performance metrics")
    mae = float(round(mean_absolute_error(true, predicted),4))
    mse = float(round(mean_squared_error(true, predicted),4))
    rmse = float(round(root_mean_squared_error(true, predicted),4))
    r2_sc = float(round(r2_score(true, predicted),4))
    return mae, mse, rmse, r2_sc

#function to read yaml file
@ensure_annotations
def read_yaml(file_path: Path) -> ConfigBox:
    try:
        with open(file_path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"{file_path} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"{path} directory created")

 #function for evaluating models for best parameters using grid search and MLOps
def evaluate_model_best_param_gsv_mlflow(X_train, y_train,X_test,y_test,models,params):
    logging.info("using grid search and mlflow")
    performance_metrics = {}
    m_name=[]
    mae=[]
    mse=[]
    rmse=[]
    r2=[]
    dur=[]
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    try:
        report = {}
        for i in range(len(list(models))):
            mlflow.set_experiment("Medical Insurance")
            with mlflow.start_run():
                signature=infer_signature(X_train,y_train)
                model_name = list(models.keys())[i]
                model = list(models.values())[i]
                param=params[list(models.keys())[i]]
                logging.info(f'evaluating {model_name}')
                gs = GridSearchCV(model,param)
                gs.fit(X_train,y_train)
                logging.info(f'best estimator : {gs.best_estimator_}')
                logging.info(f'best score : {gs.best_score_}')
                logging.info(f'best params : {gs.best_params_}')
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                (mae, mse, rmse, r2_sc) = get_model_performance_metrics(y_test, y_test_pred)
                test_model_score = r2_sc
                report[list(models.keys())[i]] = test_model_score
                logging.info(f'Model Name: {model_name} | MAE : {mae}, MSE : {mse}, RMSE : {rmse}, R2 Score : {r2_sc}')
                mlflow.set_tag('Model Name', model_name)
                mlflow.log_metric("MAE", float(mae))
                mlflow.log_metric("MSE", float(mse))
                mlflow.log_metric("RMSE",float(rmse))
                mlflow.log_metric("R2 Score", float(r2_sc))
        return report
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)