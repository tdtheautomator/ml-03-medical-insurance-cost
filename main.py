import sys
import time

from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging

from src.config.config_variables import DataIngestionConfig, VarsConfig
from src.components.data_ingestion import DataIngestion

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

          print('-'*100)
    except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)