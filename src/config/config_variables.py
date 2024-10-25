import os

from datetime import datetime
import src.vars as vars



class VarsConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp = timestamp.strftime("%Y-%m-%d")
        self.input_dir:str = vars.IN_DIR
        self.output_dir:str = os.path.join(vars.OUT_DIR,timestamp)

class DataIngestionConfig:
    def __init__(self,vars_config:VarsConfig):
        self.input_dir:str = vars.IN_DIR
        self.data_ingestion_dir:str=os.path.join(vars_config.output_dir,vars.OUT_DATA_INGESTION_DIR)
        self.input_data_path:str = os.path.join(vars.IN_DIR,vars.IN_FILE_NAME)
        self.training_data_path:str = os.path.join(self.data_ingestion_dir,vars.OUT_TRAIN_FILE_NAME)
        self.test_data_path:str = os.path.join(self.data_ingestion_dir,vars.OUT_TEST_FILE_NAME)
        self.raw_data_path:str = os.path.join(self.data_ingestion_dir,vars.OUT_RAW_FILE_NAME)
        self.train_test_split_ratio:float = vars.TRAIN_TEST_SPLIT_RATIO

class DataValidationConfig:
    def __init__(self,vars_config:VarsConfig):
        self.data_validaton_dir:str=os.path.join(vars_config.output_dir,vars.OUT_DATA_VALIDATION_DIR)
        self.input_data_schema_path:str = os.path.join(vars.IN_DIR,vars.IN_DATA_SCHEMA)
        self.valid_data_dir: str = os.path.join(self.data_validaton_dir, vars.OUT_DATA_VALID_DIR)
        self.invalid_data_dir: str = os.path.join(self.data_validaton_dir, vars.OUT_DATA_INVALID_DIR)
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, vars.OUT_TRAIN_FILE_NAME)
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, vars.OUT_TEST_FILE_NAME)
        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, vars.OUT_TRAIN_FILE_NAME)
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, vars.OUT_TEST_FILE_NAME)
        self.drift_report_file_path: str = os.path.join(self.data_validaton_dir,vars.OUT_DATA_DRIFT_REPORT_DIR, vars.OUT_DATA_DRIFT_REPORT_FILE_NAME)

class DataTransformationConfig:
     def __init__(self,vars_config:VarsConfig):
        self.data_transformation_dir:str=os.path.join(vars_config.output_dir,vars.OUT_DATA_TRANSFORMATION_DIR)
        self.encoded_file_path:str = os.path.join(self.data_transformation_dir,vars.OUT_ENCODED_FILE_NAME)
        self.transformed_train_file_path:str = os.path.join(self.data_transformation_dir,vars.OUT_TRANSFORMED_TRAIN_FILE_NAME)
        self.transformed_test_file_path:str = os.path.join(self.data_transformation_dir,vars.OUT_TRANSFORMED_TEST_FILE_NAME)
        self.input_data_schema_path:str = os.path.join(vars.IN_DIR,vars.IN_DATA_SCHEMA)
     
class TrainingModelConfig:
    def __init__(self,vars_config:VarsConfig):
        self.trained_mode_dir:str=os.path.join(vars_config.output_dir,vars.OUT_TRAINED_MODEL_DIR)
        self.trained_model_file_path:str = os.path.join(self.trained_mode_dir,vars.OUT_MODEL_FILE_NAME)
