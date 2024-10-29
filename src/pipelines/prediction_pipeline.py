#file used to create code for predection
import os
import sys
import pandas as pd

from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging
from src.helper.common import load_object

from src.config.config_variables import PredictionPipelineConfig


class PredictPipeline:
    def __init__(self,prediction_pipeline_config:PredictionPipelineConfig):
        self.prediction_pipeline_config=prediction_pipeline_config
        self.final_encoded_file = self.prediction_pipeline_config.final_encoded_file
        self.final_trained_model_file = self.prediction_pipeline_config.final_tained_model_file

    def predict(self,features):
        try:
            logging.info("loading trained model file")
            model=load_object(self.final_trained_model_file)
            logging.info("loading encoded file")
            preprocessor=load_object(self.final_encoded_file)
            logging.info("transforming prediction data")
            data_scaled=preprocessor.transform(features)
            logging.info("performing prediction")
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
