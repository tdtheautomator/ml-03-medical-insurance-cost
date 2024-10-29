import os
import sys
import time

from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.training_model import TrainingModel

from src.config.config_variables import DataIngestionConfig, VarsConfig, DataValidationConfig, DataTransformationConfig,TrainingModelConfig
from src.config.artifacts_schema import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, TrainingModelArtifact


class TrainingPipeline:
    def __init__(self):
        try:
            self.varsConfig = VarsConfig()
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)

    def start_data_ingestion(self):
        try:
            logging.info("initiated data ingestion")
            data_ingestion_start_time = time.time()
            dataIngestionConfig = DataIngestionConfig(self.varsConfig)
            dataIngestion = DataIngestion(data_ingestion_config=dataIngestionConfig)
            dataIngestionOutput = dataIngestion.initiate_data_ingestion()
            data_ingestion_end_time = time.time()
            data_ingestion_time = round((data_ingestion_end_time - data_ingestion_start_time),2)
            logging.info(f"data ingestion completed in {data_ingestion_time} secs")
            return dataIngestionOutput
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
        
    def start_data_validation(self,dataIngestionOutput:DataIngestionArtifact):
        try:
            logging.info("initiated data validation")
            data_validation_start_time = time.time()
            dataValidationConfig=DataValidationConfig(self.varsConfig)
            dataValidation=DataValidation(dataIngestionOutput,dataValidationConfig)
            dataValidationOutput=dataValidation.initiate_data_validation()
            data_validation_end_time = time.time()
            data_validation_time = round((data_validation_end_time - data_validation_start_time),2)
            logging.info(f"data validation completed in {data_validation_time} secs")
            return dataValidationOutput
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
        
    def start_data_transformation(self,dataValidationOutput:DataValidationArtifact):
        try:
            logging.info("initiated data transformation")
            data_transformation_start_time = time.time()
            dataTransformationConfig=DataTransformationConfig(self.varsConfig)
            dataTransformation=DataTransformation(dataValidationOutput,dataTransformationConfig)
            dataTransformationOutput=dataTransformation.initiate_data_transformation()
            data_transformation_end_time = time.time()
            data_transformation_time = round((data_transformation_end_time - data_transformation_start_time),2)
            logging.info(f"data transformation completed in {data_transformation_time} secs")
            return dataTransformationOutput
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
        
    def start_model_training(self,dataTransformationOutput:DataTransformationArtifact):
        try:
            logging.info("initiated model training")
            model_training_start_time = time.time()
            TrainModelConfig=TrainingModelConfig(self.varsConfig)
            TrainModel=TrainingModel(dataTransformationOutput,TrainModelConfig)
            TrainModelOutput=TrainModel.initiate_training_model()
            model_training_end_time = time.time()
            model_training_time = round((model_training_end_time - model_training_start_time),2)
            logging.info(f"model training completed in {model_training_time} secs")
            return TrainModelOutput
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
        
    def start_training_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(dataIngestionOutput=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(dataValidationOutput=data_validation_artifact)
            training_model_artifact=self.start_model_training(dataTransformationOutput=data_transformation_artifact)
            return training_model_artifact
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)

if __name__=='__main__':
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.start_training_pipeline()
    except Exception as e:
                logging.error(e)
                raise CustomException(e,sys)