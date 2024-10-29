import sys
import time

from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.training_model import TrainingModel

from src.config.config_variables import DataIngestionConfig, VarsConfig, DataValidationConfig, DataTransformationConfig,TrainingModelConfig
from src.config.artifacts_schema import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, TrainingModelArtifact


class PredictionPipeline:
    def __init__(self):
        pass
