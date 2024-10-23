#file used to create code for common functions
import os
import sys
import time
import pickle
import dill
import pandas as pd
import numpy as np
import yaml

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
def read_yaml(file_path: str):
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise CustomException(e, sys) from e

#function to write yaml file   
def write_yaml(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys) from e

#function to save numpy array to file
def save_np_arr(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) from e
    

#function to load numpy array from file
def load_np_arr(file_path: str):
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e
    


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
                performance_metrics = get_model_performance_metrics(y_test, y_test_pred)
                test_model_score = performance_metrics[3]
                report[list(models.keys())[i]] = test_model_score
                logging.info(f'{model_name} | MAE : {performance_metrics[0]}, MSE : {performance_metrics[1]}, RMSE : {performance_metrics[2]}, R2 Score : {performance_metrics[3]}')
                #m1=model_name
                #m2={performance_metrics[0]}
                #m3={performance_metrics[1]}
                #m4={performance_metrics[2]}
                #m5={performance_metrics[3]}
                #mlflow.log_metric("Model Name", m1)
                #mlflow.log_metric("mae", m2)
                #mlflow.log_metric("mse",m3)
                #mlflow.log_metric("rmse",m4)
                #mlflow.log_metric("r2",m5)
        return report
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)