"""
Custom exception classes for the ML pipeline project.
"""

class CustomException(Exception):
    """Base exception class for the ML pipeline."""
    def __init__(self, error_message: str):
        super().__init__(error_message)
        self.error_message = error_message

class DataIngestionError(CustomException):
    """Raised when data ingestion fails."""
    pass

class DataProcessingError(CustomException):
    """Raised when data processing fails."""
    pass

class FeatureEngineeringError(CustomException):
    """Raised when feature engineering fails."""
    pass

class ModelBuildingError(CustomException):
    """Raised when model building fails."""
    pass

class ModelEvaluationError(CustomException):
    """Raised when model evaluation fails."""
    pass

class ConfigurationError(CustomException):
    """Raised when configuration loading fails."""
    pass

class FileNotFoundError(CustomException):
    """Raised when a required file is not found."""
    pass
