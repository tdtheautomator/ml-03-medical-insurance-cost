import os
import sys

# Common Variables
OUT_DIR: str = "outputs"
OUT_LOGS_DIR: str = "logs"
IN_DIR: str = "data"
IN_FILE_NAME: str = "insurance.csv"
IN_DATA_SCHEMA: str = "data_schema.yaml"

# Data Ingestion Variables
OUT_DATA_INGESTION_DIR: str = "01.data ingestion"
OUT_RAW_FILE_NAME: str = "raw_data.csv"
OUT_TRAIN_FILE_NAME: str = "train_data.csv"
OUT_TEST_FILE_NAME: str = "test_data.csv"
TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Data Validation Variables
OUT_DATA_VALIDATION_DIR: str = "02.data validation"
OUT_DATA_VALID_DIR: str = "valid"
OUT_DATA_INVALID_DIR: str = "invalid"
OUT_DATA_DRIFT_REPORT_DIR: str = "drift"
OUT_DATA_DRIFT_REPORT_FILE_NAME: str = "data_drift_report.html"

# Data Transformation Variables
OUT_DATA_TRANSFORMATION_DIR: str = "03.data transformation"
OUT_ENCODED_FILE_NAME: str = "encoded_data.pkl"
OUT_TRANSFORMED_TRAIN_FILE_NAME: str = "transformed_train_data.npy"
OUT_TRANSFORMED_TEST_FILE_NAME: str = "transformed_test_data.npy"

# Model Training Variables
OUT_TRAINED_MODEL_DIR: str = "04.trained model"
OUT_MODEL_FILE_NAME: str = "trained_model.pkl"

# MLFlow Variables
MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000/"
MLFLOW_EXP_NAME:str = "Medical Insurance"
MLFLOW_REG_MODLE_NAME:str = "medical-insurance-predictor"