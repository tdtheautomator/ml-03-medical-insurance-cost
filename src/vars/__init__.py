import os
import sys



OUT_DIR: str = "outputs"
OUT_LOGS_DIR: str = "logs"

IN_DIR: str = "data"
IN_FILE_NAME: str = "insurance.csv"
IN_DATA_SCHEMA: str = "data_schema.yaml"


OUT_DATA_INGESTION_DIR: str = "01.data ingestion"
OUT_DATA_VALIDATION_DIR: str = "02.data validation"
OUT_DATA_TRANSFORMATION_DIR: str = "03.data transformation"
OUT_TRAINED_MODEL_DIR: str = "04.trained model"

OUT_RAW_FILE_NAME: str = "raw_data.csv"
OUT_TRAIN_FILE_NAME: str = "train_data.csv"
OUT_TEST_FILE_NAME: str = "test_data.csv"
OUT_ENCODED_FILE_NAME: str = "encoded_data.pkl"
OUT_MODEL_FILE_NAME: str = "trained_model.pkl"

OUT_DATA_VALID_DIR: str = "valid"
OUT_DATA_INVALID_DIR: str = "invalid"
OUT_DATA_DRIFT_REPORT_DIR: str = "drift"
OUT_DATA_DRIFT_REPORT_FILE_NAME: str = "daat_drift_report.html"

TRAIN_TEST_SPLIT_RATIO: float = 0.2