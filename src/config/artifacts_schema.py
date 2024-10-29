from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_validation_status: bool
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    encoded_file_path:str
    transformed_train_file_path:str
    transformed_test_file_path:str

@dataclass
class RegressionModelMetrics:
    MAE: float
    MSE: float
    RMSE: float
    R2Score: float

@dataclass
class TrainingModelArtifact:
    trained_model_file_path:str
    train_metrics: RegressionModelMetrics
    test_metrics: RegressionModelMetrics