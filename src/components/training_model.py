#file used to create code for training model
import os
import sys

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging
from src.helper.common import save_object, load_np_array
from src.helper.ml_models.evaluate import evaluate_reg_model_perf
from src.helper.ml_metrics.metrics import regression_metrics
from src.helper.mlflow.track import track_experiment
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
            
            #hyper tuning parameters
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
            
            model_report:dict = evaluate_reg_model_perf(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params,searcher="gsv")
            r2_score_report = {}
            for key,value in model_report.items():
                name=key
                metrics = value[0]
                r2score = metrics['R2Score']
                r2_score_report[name] = r2score

            logging.info("evaluating best model name and score using")
            best_model_score = max(sorted(r2_score_report.values()))
            best_model_name = list(r2_score_report.keys())[
                list(r2_score_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            for k,v in r2_score_report.items():
                v = round(v,2)
                logging.info(f"Model: {k}, R2 Score: {v}")
            logging.info(f"best model selected is {best_model_name} with accuracy of {best_model_score}")
            logging.info("performing predection on test data")
            predicted_y_test=best_model.predict(X_test)
            test_metrics=regression_metrics(true=y_test,predicted=predicted_y_test)

            logging.info("loading test predection metrics to mlflow")
            exp_name = "Medical Insurance"
            (mlflow_run_id, mlflow_run_name) = track_experiment(exp_name,test_metrics, "local")
            logging.info(f"Mlflow Experiment: {exp_name}, Mlflow Run ID: {mlflow_run_id}, Mlflow Run Name: {mlflow_run_name}")
            logging.info("performing predection on training data (optional)")
            predicted_y_train=best_model.predict(X_train)
            train_metrics=regression_metrics(true=y_train,predicted=predicted_y_train)
            #track_experiment("Medical Insurance", best_model,train_metrics, "local")

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