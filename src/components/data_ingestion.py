#file used to create code for ingesting data
import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging
from src.config.config_variables import DataIngestionConfig
from src.config.artifacts_shema import DataIngestionArtifact

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise CustomException(e,sys)

    def export_data_as_dataframe(self):
        try:
            logging.info("reading datset as dataframe")
            df=pd.read_csv(self.data_ingestion_config.input_data_path) #update with dataset file
            numerical_features = [feature for feature in df.columns if df[feature].dtype != 'O']
            categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
            logging.info(f'numerical features : {format(len(numerical_features))} : {numerical_features}')
            logging.info(f'categorical features : {format(len(categorical_features))} : {categorical_features}')
            return df
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
        
    def export_data_to_outputs(self,df: pd.DataFrame):
        try:
            logging.info("creating outputs folders if doesn't exist")
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_data_path),exist_ok=True)
            logging.info("exporting dataframe to csv to outputs")
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            return df
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)

    def split_data_as_train_test(self,df: pd.DataFrame):
        try:
            logging.info("spliting training and test data")
            training_set,test_set=train_test_split(df,test_size=self.data_ingestion_config.train_test_split_ratio,random_state=None)
            
            logging.info("exporting training dataset to outputs")
            training_set.to_csv(self.data_ingestion_config.training_data_path,index=False,header=True)
            
            logging.info("exporting test dataset to outputs")
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info(f'shape of original data: {df.shape}')
            logging.info(f'shape of training data: {training_set.shape}')
            logging.info(f'shape of test data: {test_set.shape}')
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            df=self.export_data_as_dataframe()
            df=self.export_data_to_outputs(df)
            self.split_data_as_train_test(df)
            DataIngestionOutput = DataIngestionArtifact(
                train_file_path = self.data_ingestion_config.training_data_path,
                test_file_path = self.data_ingestion_config.test_data_path
            )
            return DataIngestionOutput
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)