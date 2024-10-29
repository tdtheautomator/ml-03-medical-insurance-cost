import os
import sys
import json
import pandas as pd

from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging
from src.config.config_variables import DataValidationConfig
from src.config.artifacts_schema import DataValidationArtifact, DataIngestionArtifact
from src.helper.common import read_yaml
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self.schema_file_path=self.data_validation_config.input_data_schema_path
        except Exception as e:
            raise CustomException(e,sys)
    
    @staticmethod
    def read_csv_as_df(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)    
    
    def validate_columns(self,df:pd.DataFrame)->bool:
        try:
            src_number_of_columns=len(read_yaml(self.schema_file_path)['columns'])
            tar_number_of_columns=len(df.columns)
            logging.info(f"columns in input dataframe: {src_number_of_columns}")
            logging.info(f"columns in validation dataframe: {tar_number_of_columns}")
            if tar_number_of_columns==src_number_of_columns:
                return True
            return False
        except Exception as e:
            raise CustomException(e,sys)

    def detect_dataset_drift(self, ref_df: pd.DataFrame, target_df: pd.DataFrame, ) -> bool:
        try:
            data_drift_report = Report(metrics=[DataDriftPreset()])
            data_drift_report.run(reference_data=ref_df, current_data=target_df)
            report = data_drift_report.as_dict()
            os.makedirs(os.path.dirname(self.data_validation_config.drift_report_file_path),exist_ok=True)
            drift_status = report["metrics"][0]["result"]["dataset_drift"]
            return drift_status
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path
            
            train_df=DataValidation.read_csv_as_df(train_file_path)
            test_df=DataValidation.read_csv_as_df(test_file_path)

            tarin_data_status=self.validate_columns(df=train_df)
            if not tarin_data_status:
                logging.error("columns mismatched in training data")
            else:
                logging.info("training data validated sucessfully")

            test_data_status = self.validate_columns(df=test_df)
            if not test_data_status:
                logging.error("columns mismatched in test data")
            else:
                logging.info("training data validated sucessfully")

            drift_status=self.detect_dataset_drift(ref_df=train_df,target_df=test_df)

            if not drift_status:
                dir_path=os.path.dirname(self.data_validation_config.invalid_train_file_path)
                os.makedirs(dir_path,exist_ok=True)
                logging.error("data drift detected")
                logging.info("exporting invalid files")
                train_df.to_csv(self.data_validation_config.invalid_train_file_path, index=False, header=True)
                test_df.to_csv(self.data_validation_config.invalid_test_file_path, index=False, header=True)
                logging.info("building DataValidationArtifact")
                data_validation_artifact = DataValidationArtifact(
                    valid_train_file_path=self.data_ingestion_artifact.train_file_path,
                    valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                    invalid_train_file_path=None,
                    invalid_test_file_path=None,
                    drift_validation_status=drift_status,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path
                )
            else:
                dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
                os.makedirs(dir_path,exist_ok=True)
                logging.info("no data drift detected")
                logging.info("exporting valid files")
                train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
                test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)
                logging.info("building DataValidationArtifact")
                data_validation_artifact = DataValidationArtifact(
                    valid_train_file_path=self.data_ingestion_artifact.train_file_path,
                    valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                    invalid_train_file_path=None,
                    invalid_test_file_path=None,
                    drift_validation_status=drift_status,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path
                )
            return data_validation_artifact

        except Exception as e:
            raise CustomException(e,sys)