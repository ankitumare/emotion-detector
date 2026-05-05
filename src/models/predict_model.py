"""
Production-grade model evaluation module for ML pipeline.

This module handles machine learning model evaluation operations with
robust error handling and logging for enterprise environments.
"""

import os
import yaml
from typing import Dict, Any, List
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np

from ..exceptions import ModelEvaluationError, ConfigurationError
from ..logger import get_logger


class ModelEvaluator:
    """
    Handles machine learning model evaluation with robust error handling and logging.
    
    Attributes:
        params (Dict): Configuration parameters for model evaluation
        logger: Logger instance for tracking operations
        model: Loaded machine learning model
    """
    
    def __init__(self, params_path: str = 'params.yaml'):
        """
        Initialize ModelEvaluator with configuration parameters.
        
        Args:
            params_path: Path to the parameters YAML file
            
        Raises:
            ConfigurationError: If parameters cannot be loaded
        """
        self.logger = get_logger(__name__)
        self.logger.info("Initializing ModelEvaluator class")
        
        try:
            self.params = self._load_params(params_path)
            self.model = None
            self.logger.info("Successfully initialized ModelEvaluator")
        except Exception as e:
            self.logger.error(f"Failed to initialize ModelEvaluator: {e}")
            raise ConfigurationError(f"Initialization failed: {e}")
    
    def _load_params(self, params_path: str) -> Dict:
        """
        Load configuration parameters from YAML file.
        
        Args:
            params_path: Path to the parameters file
            
        Returns:
            Dictionary containing model evaluation parameters
            
        Raises:
            ConfigurationError: If parameters cannot be loaded or parsed
        """
        try:
            with open(params_path, 'r') as file:
                params = yaml.safe_load(file)
            
            model_evaluation_params = params['model_evaluation']
            self.logger.debug("Successfully parsed model evaluation parameters")
            return model_evaluation_params
            
        except FileNotFoundError:
            self.logger.error(f"Parameters file not found at path: {params_path}")
            raise ConfigurationError(f"Parameters file missing: {params_path}")
        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error: {e}")
            raise ConfigurationError(f"Invalid YAML format: {e}")
        except KeyError as e:
            self.logger.error(f"Missing required parameter section: {e}")
            raise ConfigurationError(f"Missing parameter section: {e}")
    
    def load_model(self, model_path: str = 'model.pkl') -> Any:
        """
        Load trained model for evaluation.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded machine learning model
            
        Raises:
            ModelEvaluationError: If model loading fails
        """
        try:
            self.logger.info(f"Loading model from: {model_path}")
            
            if not os.path.exists(model_path):
                raise ModelEvaluationError(f"Model file not found: {model_path}")
            
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            
            self.logger.info("Model loaded successfully")
            return self.model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise ModelEvaluationError(f"Model loading failed: {e}")
    
    def load_test_data(self, data_path: str) -> tuple:
        """
        Load test data for model evaluation.
        
        Args:
            data_path: Base path containing feature data
            
        Returns:
            Tuple of (X_test, y_test)
            
        Raises:
            ModelEvaluationError: If data loading fails
        """
        try:
            self.logger.info(f"Loading test data from {data_path}")
            
            test_path = os.path.join(data_path, 'features', 'test_bow.csv')
            
            if not os.path.exists(test_path):
                raise ModelEvaluationError(f"Test data not found: {test_path}")
            
            test_data = pd.read_csv(test_path)
            
            self.logger.info(f"Loaded test data shape: {test_data.shape}")
            
            # Separate features and labels
            if 'label' not in test_data.columns:
                raise ModelEvaluationError("Missing 'label' column in test data")
            
            X_test = test_data.iloc[:, :-1].values  # All columns except last
            y_test = test_data.iloc[:, -1].values    # Last column
            
            self.logger.info(f"Test features shape: {X_test.shape}")
            self.logger.info(f"Test labels shape: {y_test.shape}")
            
            # Check for data quality issues
            if len(X_test) == 0:
                raise ModelEvaluationError("Empty test dataset")
            
            # Log class distribution
            unique_classes, class_counts = zip(*[(x, list(y_test).count(x)) for x in set(y_test)])
            self.logger.info(f"Test class distribution: {dict(zip(unique_classes, class_counts))}")
            
            return X_test, y_test
            
        except Exception as e:
            self.logger.error(f"Failed to load test data: {e}")
            raise ModelEvaluationError(f"Test data loading failed: {e}")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary containing evaluation metrics
            
        Raises:
            ModelEvaluationError: If metric calculation fails
        """
        try:
            self.logger.info("Calculating evaluation metrics")
            
            metrics = {}
            average_method = self.params.get('average_method', 'binary')
            
            # Calculate each metric
            for metric_name in self.params['metrics']:
                if metric_name == 'accuracy':
                    metrics['accuracy'] = accuracy_score(y_true, y_pred)
                elif metric_name == 'precision':
                    metrics['precision'] = precision_score(y_true, y_pred, average=average_method)
                elif metric_name == 'recall':
                    metrics['recall'] = recall_score(y_true, y_pred, average=average_method)
                elif metric_name == 'auc':
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                else:
                    self.logger.warning(f"Unknown metric: {metric_name}")
            
            self.logger.info("Metrics calculated successfully")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate metrics: {e}")
            raise ModelEvaluationError(f"Metric calculation failed: {e}")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the loaded model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
            
        Raises:
            ModelEvaluationError: If evaluation fails
        """
        try:
            self.logger.info("Starting model evaluation")
            
            if self.model is None:
                raise ModelEvaluationError("No model loaded for evaluation")
            
            # Make predictions
            self.logger.info("Making predictions on test data")
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            self.logger.info(f"Predictions completed. Shape: {y_pred.shape}")
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Log metrics
            self.logger.info("Evaluation Results:")
            for metric_name, metric_value in metrics.items():
                self.logger.info(f"  {metric_name}: {metric_value:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            raise ModelEvaluationError(f"Evaluation failed: {e}")
    
    def save_metrics(self, metrics: Dict[str, float], metrics_path: str = 'reports/metrics.json') -> None:
        """
        Save evaluation metrics to JSON file.
        
        Args:
            metrics: Dictionary of evaluation metrics
            metrics_path: Path to save the metrics
            
        Raises:
            ModelEvaluationError: If saving fails
        """
        try:
            self.logger.info(f"Saving metrics to: {metrics_path}")
            
            # Add metadata
            metrics_with_metadata = {
                'metrics': metrics,
                'metadata': {
                    'model_type': 'gradient_boosting',
                    'evaluation_date': pd.Timestamp.now().isoformat(),
                    'parameters': self.params
                }
            }
            
            with open(metrics_path, 'w') as file:
                json.dump(metrics_with_metadata, file, indent=4)
            
            self.logger.info(f"Metrics saved successfully to: {metrics_path}")
            
            # Log file size for monitoring
            file_size = os.path.getsize(metrics_path)
            self.logger.info(f"Metrics file size: {file_size} bytes")
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
            raise ModelEvaluationError(f"Metrics saving failed: {e}")
    
    def run_evaluation(self) -> None:
        """
        Execute the complete model evaluation pipeline.
        
        Raises:
            ModelEvaluationError: If any step in the pipeline fails
        """
        try:
            self.logger.info("=" * 50)
            self.logger.info("STARTING MODEL EVALUATION PIPELINE")
            self.logger.info("=" * 50)
            
            # Load model
            self.load_model()
            
            # Load test data
            X_test, y_test = self.load_test_data('data')
            
            # Evaluate model
            metrics = self.evaluate_model(X_test, y_test)
            
            # Save metrics
            self.save_metrics(metrics)
            
            self.logger.info("=" * 50)
            self.logger.info("MODEL EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 50)
            
        except Exception as e:
            self.logger.error(f"Model evaluation pipeline failed: {e}")
            raise ModelEvaluationError(f"Pipeline execution failed: {e}")


def main() -> None:
    """
    Main function to execute model evaluation pipeline.
    """
    try:
        evaluator = ModelEvaluator()
        evaluator.run_evaluation()
        
    except Exception as e:
        print(f"Fatal error in model evaluation: {e}")
        raise


if __name__ == '__main__':
    main()