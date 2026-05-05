"""
Production-grade data ingestion module for ML pipeline.

This module handles downloading, preprocessing, and splitting data
for machine learning tasks with robust error handling and logging.
"""

import os
import yaml
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from ..exceptions import DataIngestionError, ConfigurationError, FileNotFoundError
from ..logger import get_logger


class DataIngestion:
    """
    Handles data ingestion operations including loading, preprocessing, and saving.
    
    Attributes:
        params (Dict): Configuration parameters for data ingestion
        logger: Logger instance for tracking operations
    """
    
    def __init__(self, params_path: str = 'params.yaml'):
        """
        Initialize DataIngestion with configuration parameters.
        
        Args:
            params_path: Path to the parameters YAML file
            
        Raises:
            ConfigurationError: If parameters cannot be loaded
        """
        self.logger = get_logger(__name__)
        self.logger.info("Initializing DataIngestion class")
        
        try:
            self.params = self._load_params(params_path)
            self.logger.info("Successfully loaded data ingestion parameters")
        except Exception as e:
            self.logger.error(f"Failed to initialize DataIngestion: {e}")
            raise ConfigurationError(f"Initialization failed: {e}")
    
    def _load_params(self, params_path: str) -> Dict:
        """
        Load configuration parameters from YAML file.
        
        Args:
            params_path: Path to the parameters file
            
        Returns:
            Dictionary containing data ingestion parameters
            
        Raises:
            ConfigurationError: If parameters cannot be loaded or parsed
        """
        try:
            with open(params_path, 'r') as file:
                params = yaml.safe_load(file)
            
            data_ingestion_params = params['data_ingestion']
            self.logger.debug("Successfully parsed data ingestion parameters")
            return data_ingestion_params
            
        except FileNotFoundError:
            self.logger.error(f"Parameters file not found at path: {params_path}")
            raise ConfigurationError(f"Parameters file missing: {params_path}")
        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error: {e}")
            raise ConfigurationError(f"Invalid YAML format: {e}")
        except KeyError as e:
            self.logger.error(f"Missing required parameter section: {e}")
            raise ConfigurationError(f"Missing parameter section: {e}")
    
    def load_data(self, data_url: str) -> pd.DataFrame:
        """
        Load data from specified URL.
        
        Args:
            data_url: URL to fetch data from
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            DataIngestionError: If data loading fails
        """
        try:
            self.logger.info(f"Loading data from URL: {data_url}")
            df = pd.read_csv(data_url)
            self.logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
            
        except pd.errors.ParserError as e:
            self.logger.error(f"CSV parsing error from {data_url}: {e}")
            raise DataIngestionError(f"Failed to parse CSV data: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading data from {data_url}: {e}")
            raise DataIngestionError(f"Data loading failed: {e}")
    
    def preprocess_data(self, df: pd.DataFrame, target_sentiments: List[str]) -> pd.DataFrame:
        """
        Preprocess the loaded data by filtering and mapping sentiments.
        
        Args:
            df: Raw DataFrame to preprocess
            target_sentiments: List of sentiment categories to include
            
        Returns:
            Preprocessed DataFrame with binary sentiment labels
            
        Raises:
            DataIngestionError: If preprocessing fails
        """
        try:
            self.logger.info(f"Starting data preprocessing with target sentiments: {target_sentiments}")
            self.logger.info(f"Input data shape: {df.shape}")
            
            # Drop unnecessary columns
            if 'tweet_id' in df.columns:
                df = df.drop(columns=['tweet_id'])
                self.logger.debug("Dropped 'tweet_id' column")
            
            # Filter for target sentiments
            initial_count = len(df)
            final_df = df[df['sentiment'].isin(target_sentiments)].copy()
            filtered_count = len(final_df)
            
            self.logger.info(f"Filtered data: {initial_count} -> {filtered_count} rows")
            
            if filtered_count == 0:
                raise DataIngestionError("No data remaining after sentiment filtering")
            
            # Map sentiments to binary labels
            if len(target_sentiments) != 2:
                raise DataIngestionError("Exactly 2 target sentiments required for binary classification")
            
            sentiment_map = {target_sentiments[0]: 1, target_sentiments[1]: 0}
            final_df['sentiment'] = final_df['sentiment'].replace(sentiment_map)
            
            self.logger.info(f"Successfully mapped sentiments: {sentiment_map}")
            self.logger.info(f"Preprocessed data shape: {final_df.shape}")
            
            return final_df
            
        except KeyError as e:
            self.logger.error(f"Missing required column in DataFrame: {e}")
            raise DataIngestionError(f"Missing column: {e}")
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise DataIngestionError(f"Preprocessing error: {e}")
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            df: DataFrame to split
            
        Returns:
            Tuple of (train_data, test_data)
            
        Raises:
            DataIngestionError: If data splitting fails
        """
        try:
            test_size = self.params['test_size']
            random_state = self.params['random_state']
            
            self.logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
            
            train_data, test_data = train_test_split(
                df, 
                test_size=test_size, 
                random_state=random_state,
                stratify=df['sentiment']  # Maintain class balance
            )
            
            self.logger.info(f"Train set shape: {train_data.shape}")
            self.logger.info(f"Test set shape: {test_data.shape}")
            
            # Verify class distribution
            train_dist = train_data['sentiment'].value_counts().to_dict()
            test_dist = test_data['sentiment'].value_counts().to_dict()
            
            self.logger.info(f"Train class distribution: {train_dist}")
            self.logger.info(f"Test class distribution: {test_dist}")
            
            return train_data, test_data
            
        except Exception as e:
            self.logger.error(f"Data splitting failed: {e}")
            raise DataIngestionError(f"Splitting error: {e}")
    
    def save_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
        """
        Save processed data to specified directory.
        
        Args:
            train_data: Training DataFrame to save
            test_data: Testing DataFrame to save
            data_path: Base directory path for saving data
            
        Raises:
            DataIngestionError: If saving fails
        """
        try:
            raw_data_path = os.path.join(data_path, 'raw')
            os.makedirs(raw_data_path, exist_ok=True)
            
            train_file = os.path.join(raw_data_path, "train.csv")
            test_file = os.path.join(raw_data_path, "test.csv")
            
            train_data.to_csv(train_file, index=False)
            test_data.to_csv(test_file, index=False)
            
            self.logger.info(f"Saved training data to: {train_file}")
            self.logger.info(f"Saved testing data to: {test_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
            raise DataIngestionError(f"Data saving failed: {e}")
    
    def run_ingestion(self) -> None:
        """
        Execute the complete data ingestion pipeline.
        
        Raises:
            DataIngestionError: If any step in the pipeline fails
        """
        try:
            self.logger.info("=" * 50)
            self.logger.info("STARTING DATA INGESTION PIPELINE")
            self.logger.info("=" * 50)
            
            # Load data
            df = self.load_data(self.params['data_url'])
            
            # Preprocess data
            processed_df = self.preprocess_data(df, self.params['target_sentiments'])
            
            # Split data
            train_data, test_data = self.split_data(processed_df)
            
            # Save data
            self.save_data(train_data, test_data, 'data')
            
            self.logger.info("=" * 50)
            self.logger.info("DATA INGESTION PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 50)
            
        except Exception as e:
            self.logger.error(f"Data ingestion pipeline failed: {e}")
            raise DataIngestionError(f"Pipeline execution failed: {e}")


def main() -> None:
    """
    Main function to execute data ingestion pipeline.
    """
    try:
        ingestion = DataIngestion()
        ingestion.run_ingestion()
        
    except Exception as e:
        print(f"Fatal error in data ingestion: {e}")
        raise


if __name__ == '__main__':
    main()
