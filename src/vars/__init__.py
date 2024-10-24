import os
import sys


IN_DIR: str = "data"
LOGS_DIR: str = "logs"
OUT_DIR: str = "outputs"

IN_FILE_NAME: str = "insurance.csv"

OUT_RAW_FILE_NAME: str = "raw_data.csv"
OUT_TRAIN_FILE_NAME: str = "train_data.csv"
OUT_TEST_FILE_NAME: str = "test_data.csv"
OUT_ENCODED_FILE_NAME: str = "encoded_data.pkl"
OUT_MODEL_FILE_NAME: str = "trained_model.pkl"

TRAIN_TEST_SPLIT_RATIO: float = 0.2