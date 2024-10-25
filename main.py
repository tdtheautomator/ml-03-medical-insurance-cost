import sys
import time

from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging

from src.config.config_variables import DataIngestionConfig, VarsConfig, DataValidationConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation

if __name__=='__main__':
    try:
          print('-'*100)
          
          logging.info("initiated data ingestion")
          data_ingestion_start_time = time.time()
          varsConfig = VarsConfig()
          dataIngestionConfig = DataIngestionConfig(varsConfig)
          dataIngestion = DataIngestion(dataIngestionConfig)
          dataIngestionOutput = dataIngestion.initiate_data_ingestion()
          data_ingestion_end_time = time.time()
          data_ingestion_time = round((data_ingestion_end_time - data_ingestion_start_time),2)
          logging.info(f"data ingestion completed in {data_ingestion_time} secs")
          
          logging.info("initiated data validation")
          data_validation_start_time = time.time()
          dataValidationConfig=DataValidationConfig(varsConfig)
          dataValidation=DataValidation(dataIngestionOutput,dataValidationConfig)
          dataValidationOutput=dataValidation.initiate_data_validation()
          data_validation_end_time = time.time()
          data_validation_time = round((data_ingestion_end_time - data_ingestion_start_time),2)
          logging.info(f"data validation completed in {data_validation_time} secs")

          print('-'*100)
    except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)