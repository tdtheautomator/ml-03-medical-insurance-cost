#file used to create code for common functions
import os
import sys
import time
import pickle
import dill
import pandas as pd
import numpy as np

from src.tools.custom_exception import CustomException
from src.tools.custom_logger import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

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

 #function for evaluating models for best parameters
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

def get_model_performance_metrics(true, predicted):
    logging.info("getting performance metrics")
    mae = round(mean_absolute_error(true, predicted),4)
    mse = round(mean_squared_error(true, predicted),4)
    rmse = round(root_mean_squared_error(true, predicted),4)
    r2_sc = round(r2_score(true, predicted),4)
    return mae, mse, rmse, r2_sc