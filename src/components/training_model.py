#file used to create code for training model
import os
import sys

from dataclasses import dataclass
from src.tools.custom_exception import CustomException
from src.tools.custom_logger import logging
from src.tools.common import save_object, evaluate_model_best_param_gsv, evaluate_model_best_param_rsv, evaluate_model_best_param_gsv_mlflow
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


@dataclass

@dataclass
class TrainingModelConfig:
    trained_model_file_path=os.path.join("outputs","trained_model.pkl")

class TrainingModel:
    def __init__(self):
        self.training_model_config=TrainingModelConfig()

    def initiate_training_model(self,training_array,test_array):
        logging.info("initiated model training")
        try:
            logging.info("assigning training and test data")

            X_train,y_train,X_test,y_test=(
                training_array[:,:-1], #all rows and columns except column
                training_array[:,-1],  #only last column
                test_array[:,:-1],     #all rows and columns except column
                test_array[:,-1]       #only last column
            )
            models = {
               "Linear Regression": LinearRegression(),
               "Catagory Boost Regressor": CatBoostRegressor(verbose=False),
               "Bagging Regressor": BaggingRegressor(),
               "Decision Tree Regressor": DecisionTreeRegressor(),
               "Random Forest Regressor": RandomForestRegressor(),
                "Extra Trees Regressor": ExtraTreesRegressor(),
                "Lasso Regression": Lasso(),
                "Ridge Regression": Ridge(),
                "XG Boost Regressor": XGBRegressor(), 
                "Gradient Boos Regressor": GradientBoostingRegressor(),
                "Adaptive Boost Regressor": AdaBoostRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(n_neighbors=5),
            }
            
            #used for hyper tuning
            params={
               "Linear Regression": {},
               "Catagory Boost Regressor": {
                   'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
               },
               "Bagging Regressor": {
                   'n_estimators': [8,16,32,64,128,256]
               },
               "Decision Tree Regressor": {
                   'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
               },
               "Random Forest Regressor": {
                   'n_estimators': [8,16,32,64,128,256],
               },
               "Extra Trees Regressor": {
                    'n_estimators': [256],
                    'criterion':['poisson'],
                },
                "Lasso Regression": {},
                "Ridge Regression": {},
                "XG Boost Regressor": {}, 
                "Gradient Boos Regressor": {},
                "Adaptive Boost Regressor": {},
                "K-Neighbors Regressor": {},
            }
            
            model_report:dict=evaluate_model_best_param_gsv_mlflow(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

            logging.info("evaluating best model name and score using")
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            for k,v in model_report.items():
                v = round(v,2)
                logging.info(f"Model: {k}, R2 Score: {v}")
            logging.info(f"best model selected is {best_model_name} with accuracy of {best_model_score}")

            if best_model_score<0.5:
                logging.error("unable to find any model with 0.5 and above accuracy,exiting")
                raise CustomException("unable to find any model with 0.5 and above accuracy, exiting")

            logging.info("saving trained model objects")
            save_object(
                file_path=self.training_model_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info("completed model training")
            return r2_square, best_model_name
        
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)