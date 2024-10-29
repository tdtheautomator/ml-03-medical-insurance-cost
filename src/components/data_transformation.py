#file used to create code for transforming data
import os
import sys
import numpy as np 
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging
from src.helper.common import save_object, read_yaml

from src.config.config_variables import DataTransformationConfig
from src.config.artifacts_schema import DataTransformationArtifact, DataValidationArtifact

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
            self.schema_file_path=self.data_transformation_config.input_data_schema_path
            self.train_data_path = self.data_validation_artifact.valid_train_file_path
            self.test_data_path = self.data_validation_artifact.valid_test_file_path
        except Exception as e:
            raise CustomException(e,sys)
    
    @staticmethod
    def read_csv_as_df(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
        
    def data_transformer(self)->Pipeline:
        logging.info("initiated data transformation")
        try:
            logging.info("defining numeric columns")
            numeric_columns = read_yaml(self.schema_file_path)['numerical_columns']
            logging.info("defining category columns")
            category_columns = read_yaml(self.schema_file_path)['categorical_columns']
            logging.info("defining numeric pipeline")
            numeric_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")), #handle missing values
                ("scaler",StandardScaler())
                ]
            )
            logging.info("defining category pipeline")
            category_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")), #handle missing values
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("initiating column transformer")
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",numeric_pipeline,numeric_columns),
                ("cat_pipelines",category_pipeline,category_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)

    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info("loading training data into dataframe")
            training_data_path = self.train_data_path
            test_data_path = self.test_data_path
            if os.path.isfile(training_data_path) and os.path.isfile(test_data_path):
                logging.info("test and training data files found")
                logging.info('loading test and training data into dataframe')
                training_df=pd.read_csv(training_data_path)
                test_df=pd.read_csv(test_data_path)
            else:
                logging.error("test and training data files missing")
                raise CustomException("test and training data files missing",sys)
            
            logging.info("transforming data")
            preprocessing_obj=self.data_transformer()

            target_column_name = read_yaml(self.schema_file_path)['target_column'][0]
            logging.info(f"dropping target column {target_column_name} from training dataframe" )
            input_feature_training_df=training_df.drop(columns=[target_column_name],axis=1) #dataframe
            target_feature_training_df=training_df[target_column_name] #pd series

            logging.info(f"dropping target column {target_column_name} from test dataframe" )
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1) #dataframe
            target_feature_test_df=test_df[target_column_name] #pd series

            logging.info("performing preprocessing")
            input_feature_training_arr=preprocessing_obj.fit_transform(input_feature_training_df)  #numpy array
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)  #numpy array

            logging.info("slicing objects to concatenation along the second axis")
            training_arr = np.c_[input_feature_training_arr, np.array(target_feature_training_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("saving transformed training and test data")
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_file_path),exist_ok=True)
            np.save(self.data_transformation_config.transformed_train_file_path, training_arr)
            np.save(self.data_transformation_config.transformed_test_file_path, test_arr)

            logging.info("saving encoded object")
            save_object(
                file_path=self.data_transformation_config.encoded_file_path,
                obj=preprocessing_obj
            )
            logging.info("data transformation completed")

            data_transformation_artifact=DataTransformationArtifact(
                encoded_file_path=self.data_transformation_config.encoded_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)