#file used to create code for training model
import os
import sys

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging
from src.helper.common import save_object, load_np_array
from src.helper.ml_models.evaluate import evaluate_best_model
from src.helper.ml_metrics.metrics import regression_metrics
from src.config.config_variables import TrainingModelConfig
from src.config.artifacts_shema import TrainingModelArtifact, DataTransformationArtifact

class TrainingModel:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,training_model_config:TrainingModelConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.training_model_config = training_model_config
            self.trained_model_file_path = self.training_model_config.trained_model_file_path
        except Exception as e:
            raise CustomException(e,sys)
        
    def train_model(self,X_train,y_train,X_test,y_test):
        try:
            models = {
               "Catagory Boost Regressor": CatBoostRegressor(verbose=False),
               "Decision Tree Regressor": DecisionTreeRegressor(),
               "Random Forest Regressor": RandomForestRegressor(),
            }
            
            #used for hyper tuning
            params={
               "Catagory Boost Regressor": {
                   'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
               },
               "Decision Tree Regressor": {
                   'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
               },
               "Random Forest Regressor": {
                   'n_estimators': [8,16,32,64,128,256],
               }
            }
            
            model_report:dict = evaluate_best_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params,searcher="gsv")

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

            predicted_y_test=best_model.predict(X_test)
            test_metrics=regression_metrics(true=y_test,predicted=predicted_y_test)

            predicted_y_train=best_model.predict(X_train)
            train_metrics=regression_metrics(true=y_train,predicted=predicted_y_train)

            logging.info("saving trained model objects")
            save_object(file_path=self.training_model_config.trained_model_file_path,obj=best_model)
            logging.info("completed model training")
            trained_model_artifact = TrainingModelArtifact(
                trained_model_file_path=self.training_model_config.trained_model_file_path,
                train_metrics=train_metrics, test_metrics=test_metrics)

            return trained_model_artifact
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_training_model(self)->TrainingModelArtifact:
        try:
            logging.info("loading training and test array")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            training_array = load_np_array(train_file_path)
            test_array = load_np_array(test_file_path)
            logging.info("assigning training and test data")

            X_train,y_train,X_test,y_test=(
                training_array[:,:-1], #all rows and columns except column
                training_array[:,-1],  #only last column
                test_array[:,:-1],     #all rows and columns except column
                test_array[:,-1]       #only last column
            )
            trained_model_artifact = self.train_model(X_train,y_train,X_test,y_test)
            return trained_model_artifact
        
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)