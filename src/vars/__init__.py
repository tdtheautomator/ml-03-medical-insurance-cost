import os
import sys


IN_DIR: str = "data"
LOGS_DIR: str = "logs"
OUT_DIR: str = "outputs"
OUT_DATA_INGESTION_DIR: str = "01.data ingestion"
OUT_DATA_VALIDATION_DIR: str = "02.data validation"
OUT_DATA_TRANSFORMATION_DIR: str = "03.data transformation"
OUT_DATA_MODEL_DIR: str = "04.trained model"
IN_FILE_NAME: str = "insurance.csv"

OUT_RAW_FILE_NAME: str = "raw_data.csv"
OUT_TRAIN_FILE_NAME: str = "train_data.csv"
OUT_TEST_FILE_NAME: str = "test_data.csv"
OUT_ENCODED_FILE_NAME: str = "encoded_data.pkl"
OUT_MODEL_FILE_NAME: str = "trained_model.pkl"

TRAIN_TEST_SPLIT_RATIO: float = 0.2