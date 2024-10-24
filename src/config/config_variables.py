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
        self.data_ingestion_dir:str=os.path.join(
            vars_config.output_dir,vars.OUT_DATA_INGESTION_DIR
        )

        self.input_data_path:str = os.path.join(vars.IN_DIR,vars.IN_FILE_NAME)
        self.training_data_path:str = os.path.join(self.data_ingestion_dir,vars.OUT_TRAIN_FILE_NAME)
        self.test_data_path:str = os.path.join(self.data_ingestion_dir,vars.OUT_TEST_FILE_NAME)
        self.raw_data_path:str = os.path.join(self.data_ingestion_dir,vars.OUT_RAW_FILE_NAME)
        self.train_test_split_ratio:float = vars.TRAIN_TEST_SPLIT_RATIO 

class DataValidationConfig:
    def __init__(self,vars_config:VarsConfig):
        self.input_dir:str = vars.IN_DIR
        self.output_dir:str = vars.OUT_DIR

class DataTransformationConfig:
     def __init__(self,vars_config:VarsConfig):
        self.input_dir:str = vars.IN_DIR
        self.output_dir:str = vars.OUT_DIR

        self.preprocessor_obj_file_path:str = os.path.join(vars.OUT_DIR,vars.OUT_ENCODED_FILE_NAME)
     
class TrainingModelConfig:
    def __init__(self,vars_config:VarsConfig):
        self.input_dir:str = vars.IN_DIR
        self.output_dir:str = vars.OUT_DIR

        self.trained_model_file_path:str = os.path.join(vars.OUT_DIR,vars.OUT_MODEL_FILE_NAME)
