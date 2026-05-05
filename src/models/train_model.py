"""
Production-grade model building module for ML pipeline.

This module handles machine learning model training operations with
robust error handling and logging for enterprise environments.
"""

import os
import yaml
from typing import Dict, Any
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from ..exceptions import ModelBuildingError, ConfigurationError
from ..logger import get_logger


class ModelBuilder:
    """
    Handles machine learning model training with robust error handling and logging.
    
    Attributes:
        params (Dict): Configuration parameters for model building
        logger: Logger instance for tracking operations
        model: Trained machine learning model
    """
    
    def __init__(self, params_path: str = 'params.yaml'):
        """
        Initialize ModelBuilder with configuration parameters.
        
        Args:
            params_path: Path to the parameters YAML file
            
        Raises:
            ConfigurationError: If parameters cannot be loaded
        """
        self.logger = get_logger(__name__)
        self.logger.info("Initializing ModelBuilder class")
        
        try:
            self.params = self._load_params(params_path)
            self.model = None
            self.logger.info("Successfully initialized ModelBuilder")
        except Exception as e:
            self.logger.error(f"Failed to initialize ModelBuilder: {e}")
            raise ConfigurationError(f"Initialization failed: {e}")
    
    def _load_params(self, params_path: str) -> Dict:
        """
        Load configuration parameters from YAML file.
        
        Args:
            params_path: Path to the parameters file
            
        Returns:
            Dictionary containing model building parameters
            
        Raises:
            ConfigurationError: If parameters cannot be loaded or parsed
        """
        try:
            with open(params_path, 'r') as file:
                params = yaml.safe_load(file)
            
            model_building_params = params['model_building']
            self.logger.debug("Successfully parsed model building parameters")
            return model_building_params
            
        except FileNotFoundError:
            self.logger.error(f"Parameters file not found at path: {params_path}")
            raise ConfigurationError(f"Parameters file missing: {params_path}")
        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error: {e}")
            raise ConfigurationError(f"Invalid YAML format: {e}")
        except KeyError as e:
            self.logger.error(f"Missing required parameter section: {e}")
            raise ConfigurationError(f"Missing parameter section: {e}")
    
    def _create_model(self) -> Any:
        """
        Create and configure machine learning model based on parameters.
        
        Returns:
            Configured machine learning model
            
        Raises:
            ModelBuildingError: If model creation fails
        """
        try:
            algorithm = self.params['algorithm']
            self.logger.info(f"Creating model with algorithm: {algorithm}")
            
            if algorithm == 'gradient_boosting':
                model = GradientBoostingClassifier(
                    n_estimators=self.params['n_estimators'],
                    learning_rate=self.params['learning_rate'],
                    max_depth=self.params['max_depth'],
                    random_state=self.params['random_state']
                )
                
                self.logger.info(f"GradientBoostingClassifier configuration: "
                               f"n_estimators={self.params['n_estimators']}, "
                               f"learning_rate={self.params['learning_rate']}, "
                               f"max_depth={self.params['max_depth']}, "
                               f"random_state={self.params['random_state']}")
                
                return model
            else:
                raise ModelBuildingError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            self.logger.error(f"Failed to create model: {e}")
            raise ModelBuildingError(f"Model creation failed: {e}")
    
    def load_training_data(self, data_path: str) -> tuple:
        """
        Load training data for model building.
        
        Args:
            data_path: Base path containing feature data
            
        Returns:
            Tuple of (X_train, y_train)
            
        Raises:
            ModelBuildingError: If data loading fails
        """
        try:
            self.logger.info(f"Loading training data from {data_path}")
            
            train_path = os.path.join(data_path, 'features', 'train_bow.csv')
            
            if not os.path.exists(train_path):
                raise ModelBuildingError(f"Training data not found: {train_path}")
            
            train_data = pd.read_csv(train_path)
            
            self.logger.info(f"Loaded training data shape: {train_data.shape}")
            
            # Separate features and labels
            if 'label' not in train_data.columns:
                raise ModelBuildingError("Missing 'label' column in training data")
            
            X_train = train_data.iloc[:, :-1].values  # All columns except last
            y_train = train_data.iloc[:, -1].values    # Last column
            
            self.logger.info(f"Features shape: {X_train.shape}")
            self.logger.info(f"Labels shape: {y_train.shape}")
            
            # Check for data quality issues
            if len(X_train) == 0:
                raise ModelBuildingError("Empty training dataset")
            
            if len(set(y_train)) < 2:
                raise ModelBuildingError("Training data contains only one class")
            
            # Log class distribution
            unique_classes, class_counts = zip(*[(x, list(y_train).count(x)) for x in set(y_train)])
            self.logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
            
            return X_train, y_train
            
        except Exception as e:
            self.logger.error(f"Failed to load training data: {e}")
            raise ModelBuildingError(f"Data loading failed: {e}")
    
    def train_model(self, X_train, y_train) -> Any:
        """
        Train the machine learning model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
            
        Raises:
            ModelBuildingError: If training fails
        """
        try:
            self.logger.info("Starting model training")
            
            # Create model
            self.model = self._create_model()
            
            # Train model
            self.logger.info("Fitting model on training data")
            self.model.fit(X_train, y_train)
            
            self.logger.info("Model training completed successfully")
            
            # Perform cross-validation for model validation
            self.logger.info("Performing cross-validation")
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
            
            self.logger.info(f"Cross-validation scores: {cv_scores}")
            self.logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise ModelBuildingError(f"Training failed: {e}")
    
    def save_model(self, model_path: str = 'models/model.pkl') -> None:
        """
        Save the trained model to file.
        
        Args:
            model_path: Path to save the model
            
        Raises:
            ModelBuildingError: If saving fails
        """
        try:
            if self.model is None:
                raise ModelBuildingError("No trained model to save")
            
            self.logger.info(f"Saving model to: {model_path}")
            
            with open(model_path, 'wb') as file:
                pickle.dump(self.model, file)
            
            self.logger.info(f"Model saved successfully to: {model_path}")
            
            # Log model size for monitoring
            file_size = os.path.getsize(model_path)
            self.logger.info(f"Model file size: {file_size} bytes")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise ModelBuildingError(f"Model saving failed: {e}")
    
    def run_model_building(self) -> None:
        """
        Execute the complete model building pipeline.
        
        Raises:
            ModelBuildingError: If any step in the pipeline fails
        """
        try:
            self.logger.info("=" * 50)
            self.logger.info("STARTING MODEL BUILDING PIPELINE")
            self.logger.info("=" * 50)
            
            # Load training data
            X_train, y_train = self.load_training_data('data')
            
            # Train model
            trained_model = self.train_model(X_train, y_train)
            
            # Save model
            self.save_model()
            
            self.logger.info("=" * 50)
            self.logger.info("MODEL BUILDING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 50)
            
        except Exception as e:
            self.logger.error(f"Model building pipeline failed: {e}")
            raise ModelBuildingError(f"Pipeline execution failed: {e}")


def main() -> None:
    """
    Main function to execute model building pipeline.
    """
    try:
        builder = ModelBuilder()
        builder.run_model_building()
        
    except Exception as e:
        print(f"Fatal error in model building: {e}")
        raise


if __name__ == '__main__':
    main()