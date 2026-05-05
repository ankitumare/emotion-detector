"""
Production-grade feature engineering module for ML pipeline.

This module handles text feature extraction and transformation operations
for machine learning tasks with robust error handling and logging.
"""

import os
import yaml
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib

from ..exceptions import FeatureEngineeringError, ConfigurationError
from ..logger import get_logger


class FeatureEngineer:
    """
    Handles feature engineering operations with robust error handling and logging.
    
    Attributes:
        params (Dict): Configuration parameters for feature engineering
        logger: Logger instance for tracking operations
        vectorizer: Fitted CountVectorizer instance
    """
    
    def __init__(self, params_path: str = 'params.yaml'):
        """
        Initialize FeatureEngineer with configuration parameters.
        
        Args:
            params_path: Path to the parameters YAML file
            
        Raises:
            ConfigurationError: If parameters cannot be loaded
        """
        self.logger = get_logger(__name__)
        self.logger.info("Initializing FeatureEngineer class")
        
        try:
            self.params = self._load_params(params_path)
            self.vectorizer = None
            self.logger.info("Successfully initialized FeatureEngineer")
        except Exception as e:
            self.logger.error(f"Failed to initialize FeatureEngineer: {e}")
            raise ConfigurationError(f"Initialization failed: {e}")
    
    def _load_params(self, params_path: str) -> Dict:
        """
        Load configuration parameters from YAML file.
        
        Args:
            params_path: Path to the parameters file
            
        Returns:
            Dictionary containing feature engineering parameters
            
        Raises:
            ConfigurationError: If parameters cannot be loaded or parsed
        """
        try:
            with open(params_path, 'r') as file:
                params = yaml.safe_load(file)
            
            feature_engineering_params = params['feature_engineering']
            self.logger.debug("Successfully parsed feature engineering parameters")
            return feature_engineering_params
            
        except FileNotFoundError:
            self.logger.error(f"Parameters file not found at path: {params_path}")
            raise ConfigurationError(f"Parameters file missing: {params_path}")
        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error: {e}")
            raise ConfigurationError(f"Invalid YAML format: {e}")
        except KeyError as e:
            self.logger.error(f"Missing required parameter section: {e}")
            raise ConfigurationError(f"Missing parameter section: {e}")
    
    def _create_vectorizer(self) -> CountVectorizer:
        """
        Create and configure CountVectorizer based on parameters.
        
        Returns:
            Configured CountVectorizer instance
            
        Raises:
            FeatureEngineeringError: If vectorizer creation fails
        """
        try:
            self.logger.info("Creating CountVectorizer with parameters")
            
            vectorizer = CountVectorizer(
                max_features=self.params['max_features'],
                ngram_range=tuple(self.params['ngram_range']),
                min_df=self.params['min_df'],
                max_df=self.params['max_df']
            )
            
            self.logger.info(f"Vectorizer configuration: max_features={self.params['max_features']}, "
                           f"ngram_range={self.params['ngram_range']}, "
                           f"min_df={self.params['min_df']}, max_df={self.params['max_df']}")
            
            return vectorizer
            
        except Exception as e:
            self.logger.error(f"Failed to create vectorizer: {e}")
            raise FeatureEngineeringError(f"Vectorizer creation failed: {e}")
    
    def load_processed_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load processed training and testing data.
        
        Args:
            data_path: Base path containing processed data
            
        Returns:
            Tuple of (train_data, test_data)
            
        Raises:
            FeatureEngineeringError: If data loading fails
        """
        try:
            self.logger.info(f"Loading processed data from {data_path}")
            
            train_path = os.path.join(data_path, 'processed', 'train_processed.csv')
            test_path = os.path.join(data_path, 'processed', 'test_processed.csv')
            
            if not os.path.exists(train_path):
                raise FeatureEngineeringError(f"Training data not found: {train_path}")
            if not os.path.exists(test_path):
                raise FeatureEngineeringError(f"Testing data not found: {test_path}")
            
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            self.logger.info(f"Loaded training data shape: {train_data.shape}")
            self.logger.info(f"Loaded testing data shape: {test_data.shape}")
            
            # Check for required columns
            required_columns = ['content', 'sentiment']
            for col in required_columns:
                if col not in train_data.columns:
                    raise FeatureEngineeringError(f"Missing column '{col}' in training data")
                if col not in test_data.columns:
                    raise FeatureEngineeringError(f"Missing column '{col}' in testing data")
            
            return train_data, test_data
            
        except Exception as e:
            self.logger.error(f"Failed to load processed data: {e}")
            raise FeatureEngineeringError(f"Data loading failed: {e}")
    
    def _prepare_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for feature extraction.
        
        Args:
            train_data: Training DataFrame
            test_data: Testing DataFrame
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
            
        Raises:
            FeatureEngineeringError: If data preparation fails
        """
        try:
            self.logger.info("Preparing data for feature extraction")
            
            # Handle missing values
            train_data = train_data.fillna('')
            test_data = test_data.fillna('')
            
            # Extract features and labels
            X_train = train_data['content'].values
            y_train = train_data['sentiment'].values
            
            X_test = test_data['content'].values
            y_test = test_data['sentiment'].values
            
            self.logger.info(f"Training features shape: {X_train.shape}")
            self.logger.info(f"Training labels shape: {y_train.shape}")
            self.logger.info(f"Testing features shape: {X_test.shape}")
            self.logger.info(f"Testing labels shape: {y_test.shape}")
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            self.logger.error(f"Failed to prepare data: {e}")
            raise FeatureEngineeringError(f"Data preparation failed: {e}")
    
    def extract_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features using Bag of Words vectorization.
        
        Args:
            X_train: Training text data
            X_test: Testing text data
            
        Returns:
            Tuple of (X_train_bow, X_test_bow)
            
        Raises:
            FeatureEngineeringError: If feature extraction fails
        """
        try:
            self.logger.info("Starting feature extraction with Bag of Words")
            
            # Create vectorizer
            self.vectorizer = self._create_vectorizer()
            
            # Fit and transform training data
            self.logger.info("Fitting vectorizer on training data")
            X_train_bow = self.vectorizer.fit_transform(X_train)
            
            # Transform test data
            self.logger.info("Transforming test data")
            X_test_bow = self.vectorizer.transform(X_test)
            
            self.logger.info(f"Training features shape after vectorization: {X_train_bow.shape}")
            self.logger.info(f"Testing features shape after vectorization: {X_test_bow.shape}")
            
            # Get feature names for logging
            feature_names = self.vectorizer.get_feature_names_out()
            self.logger.info(f"Number of features extracted: {len(feature_names)}")
            
            return X_train_bow, X_test_bow
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            raise FeatureEngineeringError(f"Feature extraction failed: {e}")
    
    def _create_feature_dataframes(self, X_train_bow: np.ndarray, y_train: np.ndarray, 
                                 X_test_bow: np.ndarray, y_test: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create DataFrames from extracted features.
        
        Args:
            X_train_bow: Training feature matrix
            y_train: Training labels
            X_test_bow: Testing feature matrix
            y_test: Testing labels
            
        Returns:
            Tuple of (train_df, test_df)
            
        Raises:
            FeatureEngineeringError: If DataFrame creation fails
        """
        try:
            self.logger.info("Creating feature DataFrames")
            
            # Convert sparse matrices to dense arrays
            X_train_dense = X_train_bow.toarray()
            X_test_dense = X_test_bow.toarray()
            
            # Create DataFrames
            train_df = pd.DataFrame(X_train_dense)
            train_df['label'] = y_train
            
            test_df = pd.DataFrame(X_test_dense)
            test_df['label'] = y_test
            
            self.logger.info(f"Training DataFrame shape: {train_df.shape}")
            self.logger.info(f"Testing DataFrame shape: {test_df.shape}")
            
            return train_df, test_df
            
        except Exception as e:
            self.logger.error(f"Failed to create feature DataFrames: {e}")
            raise FeatureEngineeringError(f"DataFrame creation failed: {e}")
    
    def save_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str) -> None:
        """
        Save extracted features to specified directory.
        
        Args:
            train_df: Training features DataFrame
            test_df: Testing features DataFrame
            data_path: Base path for saving features
            
        Raises:
            FeatureEngineeringError: If saving fails
        """
        try:
            features_path = os.path.join(data_path, 'features')
            os.makedirs(features_path, exist_ok=True)
            
            train_file = os.path.join(features_path, "train_bow.csv")
            test_file = os.path.join(features_path, "test_bow.csv")
            vectorizer_file = os.path.join(features_path, "vectorizer.pkl")
            
            # Save feature data
            train_df.to_csv(train_file, index=False)
            test_df.to_csv(test_file, index=False)
            
            # Save vectorizer for future use
            joblib.dump(self.vectorizer, vectorizer_file)
            
            self.logger.info(f"Saved training features to: {train_file}")
            self.logger.info(f"Saved testing features to: {test_file}")
            self.logger.info(f"Saved vectorizer to: {vectorizer_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save features: {e}")
            raise FeatureEngineeringError(f"Feature saving failed: {e}")
    
    def run_feature_engineering(self) -> None:
        """
        Execute the complete feature engineering pipeline.
        
        Raises:
            FeatureEngineeringError: If any step in the pipeline fails
        """
        try:
            self.logger.info("=" * 50)
            self.logger.info("STARTING FEATURE ENGINEERING PIPELINE")
            self.logger.info("=" * 50)
            
            # Load processed data
            train_data, test_data = self.load_processed_data('data')
            
            # Prepare data
            X_train, y_train, X_test, y_test = self._prepare_data(train_data, test_data)
            
            # Extract features
            X_train_bow, X_test_bow = self.extract_features(X_train, X_test)
            
            # Create feature DataFrames
            train_df, test_df = self._create_feature_dataframes(X_train_bow, y_train, X_test_bow, y_test)
            
            # Save features
            self.save_features(train_df, test_df, 'data')
            
            self.logger.info("=" * 50)
            self.logger.info("FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 50)
            
        except Exception as e:
            self.logger.error(f"Feature engineering pipeline failed: {e}")
            raise FeatureEngineeringError(f"Pipeline execution failed: {e}")


def main() -> None:
    """
    Main function to execute feature engineering pipeline.
    """
    try:
        engineer = FeatureEngineer()
        engineer.run_feature_engineering()
        
    except Exception as e:
        print(f"Fatal error in feature engineering: {e}")
        raise


if __name__ == '__main__':
    main()