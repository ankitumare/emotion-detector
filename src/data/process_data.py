"""
Production-grade data preprocessing module for ML pipeline.

This module handles text preprocessing operations including cleaning,
normalization, and transformation for machine learning tasks.
"""

import os
import re
import yaml
from typing import Dict, List
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from ..exceptions import DataProcessingError, ConfigurationError
from ..logger import get_logger


class DataProcessor:
    """
    Handles text preprocessing operations with robust error handling and logging.
    
    Attributes:
        params (Dict): Configuration parameters for data processing
        logger: Logger instance for tracking operations
        lemmatizer: NLTK lemmatizer instance
        stop_words: Set of English stop words
    """
    
    def __init__(self, params_path: str = 'params.yaml'):
        """
        Initialize DataProcessor with configuration parameters.
        
        Args:
            params_path: Path to the parameters YAML file
            
        Raises:
            ConfigurationError: If parameters cannot be loaded
        """
        self.logger = get_logger(__name__)
        self.logger.info("Initializing DataProcessor class")
        
        try:
            self.params = self._load_params(params_path)
            self._setup_nltk_resources()
            self.logger.info("Successfully initialized DataProcessor")
        except Exception as e:
            self.logger.error(f"Failed to initialize DataProcessor: {e}")
            raise ConfigurationError(f"Initialization failed: {e}")
    
    def _load_params(self, params_path: str) -> Dict:
        """
        Load configuration parameters from YAML file.
        
        Args:
            params_path: Path to the parameters file
            
        Returns:
            Dictionary containing data processing parameters
            
        Raises:
            ConfigurationError: If parameters cannot be loaded or parsed
        """
        try:
            with open(params_path, 'r') as file:
                params = yaml.safe_load(file)
            
            data_processing_params = params['data_processing']
            self.logger.debug("Successfully parsed data processing parameters")
            return data_processing_params
            
        except FileNotFoundError:
            self.logger.error(f"Parameters file not found at path: {params_path}")
            raise ConfigurationError(f"Parameters file missing: {params_path}")
        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error: {e}")
            raise ConfigurationError(f"Invalid YAML format: {e}")
        except KeyError as e:
            self.logger.error(f"Missing required parameter section: {e}")
            raise ConfigurationError(f"Missing parameter section: {e}")
    
    def _setup_nltk_resources(self) -> None:
        """
        Download and setup required NLTK resources.
        
        Raises:
            DataProcessingError: If NLTK resources cannot be downloaded
        """
        try:
            self.logger.info("Downloading NLTK resources")
            nltk.download('wordnet', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words("english"))
            
            self.logger.info("NLTK resources setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup NLTK resources: {e}")
            raise DataProcessingError(f"NLTK setup failed: {e}")
    
    def _lower_case(self, text: str) -> str:
        """
        Convert text to lowercase.
        
        Args:
            text: Input text to convert
            
        Returns:
            Lowercase text
        """
        try:
            words = text.split()
            lower_words = [word.lower() for word in words]
            return " ".join(lower_words)
        except Exception as e:
            self.logger.error(f"Error in lower_case transformation: {e}")
            raise DataProcessingError(f"Lower case transformation failed: {e}")
    
    def _remove_stop_words(self, text: str) -> str:
        """
        Remove stop words from text.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with stop words removed
        """
        try:
            if not self.params.get('remove_stopwords', True):
                return text
                
            words = str(text).split()
            filtered_words = [word for word in words if word not in self.stop_words]
            return " ".join(filtered_words)
        except Exception as e:
            self.logger.error(f"Error removing stop words: {e}")
            raise DataProcessingError(f"Stop words removal failed: {e}")
    
    def _remove_numbers(self, text: str) -> str:
        """
        Remove numbers from text.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with numbers removed
        """
        try:
            if not self.params.get('remove_numbers', True):
                return text
                
            return ''.join([char for char in text if not char.isdigit()])
        except Exception as e:
            self.logger.error(f"Error removing numbers: {e}")
            raise DataProcessingError(f"Number removal failed: {e}")
    
    def _remove_punctuations(self, text: str) -> str:
        """
        Remove punctuations from text.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with punctuations removed
        """
        try:
            if not self.params.get('remove_punctuation', True):
                return text
                
            # Remove punctuations
            punctuation_pattern = r'[%s]' % re.escape(r"""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~""")
            text = re.sub(punctuation_pattern, ' ', text)
            text = text.replace(';', "")
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            text = " ".join(text.split())
            
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error removing punctuations: {e}")
            raise DataProcessingError(f"Punctuation removal failed: {e}")
    
    def _remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with URLs removed
        """
        try:
            if not self.params.get('remove_urls', True):
                return text
                
            url_pattern = re.compile(r'https?://\S+|www\.\S+')
            return url_pattern.sub(r'', text)
        except Exception as e:
            self.logger.error(f"Error removing URLs: {e}")
            raise DataProcessingError(f"URL removal failed: {e}")
    
    def _lemmatize(self, text: str) -> str:
        """
        Apply lemmatization to text.
        
        Args:
            text: Input text to process
            
        Returns:
            Lemmatized text
        """
        try:
            if not self.params.get('lemmatization', True):
                return text
                
            words = text.split()
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            return " ".join(lemmatized_words)
        except Exception as e:
            self.logger.error(f"Error in lemmatization: {e}")
            raise DataProcessingError(f"Lemmatization failed: {e}")
    
    def _remove_small_sentences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove sentences that are too short.
        
        Args:
            df: DataFrame with text data
            
        Returns:
            DataFrame with short sentences removed
        """
        try:
            min_length = self.params.get('min_sentence_length', 3)
            self.logger.info(f"Removing sentences with less than {min_length} words")
            
            initial_count = len(df)
            
            for i in range(len(df)):
                try:
                    if len(df.content.iloc[i].split()) < min_length:
                        df.content.iloc[i] = np.nan
                except (AttributeError, IndexError):
                    continue
            
            # Remove rows with NaN content
            df = df.dropna(subset=['content']).reset_index(drop=True)
            final_count = len(df)
            
            self.logger.info(f"Removed {initial_count - final_count} short sentences")
            return df
            
        except Exception as e:
            self.logger.error(f"Error removing small sentences: {e}")
            raise DataProcessingError(f"Small sentence removal failed: {e}")
    
    def _normalize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all text normalization steps to DataFrame.
        
        Args:
            df: DataFrame with text data
            
        Returns:
            DataFrame with normalized text
        """
        try:
            self.logger.info("Starting text normalization")
            self.logger.info(f"Input DataFrame shape: {df.shape}")
            
            # Apply transformations in sequence
            transformations = [
                ("lower case", self._lower_case),
                ("stop words", self._remove_stop_words),
                ("numbers", self._remove_numbers),
                ("punctuations", self._remove_punctuations),
                ("URLs", self._remove_urls),
                ("lemmatization", self._lemmatize)
            ]
            
            for transform_name, transform_func in transformations:
                self.logger.info(f"Applying {transform_name} transformation")
                df['content'] = df['content'].apply(transform_func)
            
            self.logger.info(f"Text normalization completed. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Text normalization failed: {e}")
            raise DataProcessingError(f"Text normalization failed: {e}")
    
    def load_data(self, data_path: str) -> tuple:
        """
        Load training and testing data from specified path.
        
        Args:
            data_path: Base path containing raw data
            
        Returns:
            Tuple of (train_data, test_data)
            
        Raises:
            DataProcessingError: If data loading fails
        """
        try:
            self.logger.info(f"Loading data from {data_path}")
            
            train_path = os.path.join(data_path, 'raw', 'train.csv')
            test_path = os.path.join(data_path, 'raw', 'test.csv')
            
            if not os.path.exists(train_path):
                raise DataProcessingError(f"Training data not found: {train_path}")
            if not os.path.exists(test_path):
                raise DataProcessingError(f"Testing data not found: {test_path}")
            
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            self.logger.info(f"Loaded training data shape: {train_data.shape}")
            self.logger.info(f"Loaded testing data shape: {test_data.shape}")
            
            return train_data, test_data
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise DataProcessingError(f"Data loading failed: {e}")
    
    def save_processed_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
        """
        Save processed data to specified directory.
        
        Args:
            train_data: Processed training DataFrame
            test_data: Processed testing DataFrame
            data_path: Base path for saving processed data
            
        Raises:
            DataProcessingError: If saving fails
        """
        try:
            processed_path = os.path.join(data_path, 'processed')
            os.makedirs(processed_path, exist_ok=True)
            
            train_file = os.path.join(processed_path, "train_processed.csv")
            test_file = os.path.join(processed_path, "test_processed.csv")
            
            train_data.to_csv(train_file, index=False)
            test_data.to_csv(test_file, index=False)
            
            self.logger.info(f"Saved processed training data to: {train_file}")
            self.logger.info(f"Saved processed testing data to: {test_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save processed data: {e}")
            raise DataProcessingError(f"Data saving failed: {e}")
    
    def run_processing(self) -> None:
        """
        Execute the complete data processing pipeline.
        
        Raises:
            DataProcessingError: If any step in the pipeline fails
        """
        try:
            self.logger.info("=" * 50)
            self.logger.info("STARTING DATA PROCESSING PIPELINE")
            self.logger.info("=" * 50)
            
            # Load data
            train_data, test_data = self.load_data('data')
            
            # Normalize text
            train_processed = self._normalize_text(train_data)
            test_processed = self._normalize_text(test_data)
            
            # Remove small sentences
            train_processed = self._remove_small_sentences(train_processed)
            test_processed = self._remove_small_sentences(test_processed)
            
            # Save processed data
            self.save_processed_data(train_processed, test_processed, 'data')
            
            self.logger.info("=" * 50)
            self.logger.info("DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 50)
            
        except Exception as e:
            self.logger.error(f"Data processing pipeline failed: {e}")
            raise DataProcessingError(f"Pipeline execution failed: {e}")


def main() -> None:
    """
    Main function to execute data processing pipeline.
    """
    try:
        processor = DataProcessor()
        processor.run_processing()
        
    except Exception as e:
        print(f"Fatal error in data processing: {e}")
        raise


if __name__ == '__main__':
    main()
