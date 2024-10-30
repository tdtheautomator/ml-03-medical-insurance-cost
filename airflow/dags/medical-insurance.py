import sys
import time

from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging

from datetime import datetime
from src.config.config_variables import DataIngestionConfig, VarsConfig, DataValidationConfig, DataTransformationConfig,TrainingModelConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.training_model import TrainingModel

from airflow.models import DAG
from airflow.operators.python import PythonOperator

def data_ingestion(**context):
      try:
            logging.info("initiated data ingestion")
            data_ingestion_start_time = time.time()
            varsConfig = VarsConfig()
            dataIngestionConfig = DataIngestionConfig(varsConfig)
            dataIngestion = DataIngestion(dataIngestionConfig)
            dataIngestionOutput = dataIngestion.initiate_data_ingestion()
            data_ingestion_end_time = time.time()
            data_ingestion_time = round((data_ingestion_end_time - data_ingestion_start_time),2)
            logging.info(f"data ingestion completed in {data_ingestion_time} secs")
            context["ti"].xcom_push(key="output", value=dataIngestionOutput)
      except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)


def data_validation(**context):
      try:
            logging.info("initiated data validation")
            data_validation_start_time = time.time()
            varsConfig = VarsConfig()
            dataIngestionOutput = context['ti'].xcom_pull(key='output', task_ids='data_ingestion')
            dataValidationConfig=DataValidationConfig(varsConfig)
            dataValidation=DataValidation(dataIngestionOutput,dataValidationConfig)
            dataValidationOutput=dataValidation.initiate_data_validation()
            data_validation_end_time = time.time()
            data_validation_time = round((data_validation_end_time - data_validation_start_time),2)
            logging.info(f"data validation completed in {data_validation_time} secs")
            context["ti"].xcom_push(key="output", value=dataValidationOutput)
      except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
      
def data_transformation(**context):
      try:
            logging.info("initiated data transformation")
            data_transformation_start_time = time.time()
            varsConfig = VarsConfig()
            dataValidationOutput = context['ti'].xcom_pull(key='output', task_ids='data_ingestion')
            dataTransformationConfig=DataTransformationConfig(varsConfig)
            dataTransformation=DataTransformation(dataValidationOutput,dataTransformationConfig)
            dataTransformationOutput=dataTransformation.initiate_data_transformation()
            data_transformation_end_time = time.time()
            data_transformation_time = round((data_transformation_end_time - data_transformation_start_time),2)
            logging.info(f"data transformation completed in {data_transformation_time} secs")
            context["ti"].xcom_push(key="output", value=dataTransformationOutput)
      except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)

def model_training(**context):
      try:
            logging.info("initiated model training")
            model_training_start_time = time.time()
            dataTransformationOutput = context['ti'].xcom_pull(key='output', task_ids='data_transformation')
            TrainModelConfig=TrainingModelConfig(varsConfig)
            varsConfig = VarsConfig()
            TrainModel=TrainingModel(dataTransformationOutput,TrainModelConfig)
            trainModelOutput=TrainModel.initiate_training_model()
            model_training_end_time = time.time()
            model_training_time = round((model_training_end_time - model_training_start_time),2)
            logging.info(f"model training completed in {model_training_time} secs")
            context["ti"].xcom_push(key="output", value=trainModelOutput)
      except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)


with DAG(
    'medical-insurance-pipeline',
    start_date=datetime(2024,1,1),
    schedule_interval="@daily",
    catchup=False
) as dag:
    
    data_ingestion=PythonOperator(task_id="data_ingestion",python_callable=data_ingestion,provide_context=True)
    data_validation=PythonOperator(task_id="data_validation",python_callable=data_validation,provide_context=True)
    data_transformation=PythonOperator(task_id="data_transformation",python_callable=data_transformation,provide_context=True)
    model_training=PythonOperator(task_id="model_training",python_callable=model_training,provide_context=True)


    data_ingestion >> data_validation >> data_transformation >> model_training