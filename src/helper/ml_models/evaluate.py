import os
import sys
import time
from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from src.helper.ml_metrics.metrics import regression_metrics

#function for evaluating models for best parameters using grid search
def evaluate_best_model(X_train, y_train,X_test,y_test,models,params,searcher):
    try:
        report = {}
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]
            if searcher == "gsv":
                logging.info(f"evaluating {model_name} using GridSearchCV")
                s = GridSearchCV(model,param)
            else:
                logging.info(f"evaluating {model_name} using RandomSearchCV")
                s = RandomizedSearchCV(model,param)
            s.fit(X_train,y_train)
            model.set_params(**s.best_params_)
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)