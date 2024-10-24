from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    pass

@dataclass
class DataTransformationArtifact:
    encoded_file_path:str

@dataclass
class TrainingModelArtifact:
    pass
