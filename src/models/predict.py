"""
Production-grade model inference script for ML pipeline.

This module handles loading trained models and making predictions on new data
with robust error handling and logging for enterprise environments.
"""

import os
import pickle
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from ..exceptions import ModelEvaluationError, ConfigurationError
from ..logger import get_logger


class SentimentPredictor:
    """
    Handles model inference with robust error handling and logging.
    
    Attributes:
        model: Loaded machine learning model
        vectorizer: Loaded feature vectorizer
        logger: Logger instance for tracking operations
        lemmatizer: NLTK lemmatizer instance
        stop_words: Set of English stop words
    """
    
    def __init__(self, model_path: str = 'models/model.pkl', vectorizer_path: str = 'data/features/vectorizer.pkl'):
        """
        Initialize SentimentPredictor with loaded model and vectorizer.
        
        Args:
            model_path: Path to the saved model
            vectorizer_path: Path to the saved vectorizer
            
        Raises:
            ModelEvaluationError: If model or vectorizer loading fails
        """
        self.logger = get_logger(__name__)
        self.logger.info("Initializing SentimentPredictor class")
        
        try:
            self.model = self._load_model(model_path)
            self.vectorizer = self._load_vectorizer(vectorizer_path)
            self._setup_nltk_resources()
            self.logger.info("Successfully initialized SentimentPredictor")
        except Exception as e:
            self.logger.error(f"Failed to initialize SentimentPredictor: {e}")
            raise ModelEvaluationError(f"Initialization failed: {e}")
    
    def _load_model(self, model_path: str):
        """
        Load trained model from file.
        
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
                model = pickle.load(file)
            
            self.logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise ModelEvaluationError(f"Model loading failed: {e}")
    
    def _load_vectorizer(self, vectorizer_path: str):
        """
        Load fitted vectorizer from file.
        
        Args:
            vectorizer_path: Path to the saved vectorizer
            
        Returns:
            Loaded vectorizer
            
        Raises:
            ModelEvaluationError: If vectorizer loading fails
        """
        try:
            self.logger.info(f"Loading vectorizer from: {vectorizer_path}")
            
            if not os.path.exists(vectorizer_path):
                raise ModelEvaluationError(f"Vectorizer file not found: {vectorizer_path}")
            
            vectorizer = joblib.load(vectorizer_path)
            
            self.logger.info("Vectorizer loaded successfully")
            return vectorizer
            
        except Exception as e:
            self.logger.error(f"Failed to load vectorizer: {e}")
            raise ModelEvaluationError(f"Vectorizer loading failed: {e}")
    
    def _setup_nltk_resources(self) -> None:
        """
        Download and setup required NLTK resources.
        
        Raises:
            ModelEvaluationError: If NLTK resources cannot be downloaded
        """
        try:
            self.logger.info("Setting up NLTK resources")
            nltk.download('wordnet', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words("english"))
            
            self.logger.info("NLTK resources setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup NLTK resources: {e}")
            raise ModelEvaluationError(f"NLTK setup failed: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text input for prediction.
        
        Args:
            text: Raw text input
            
        Returns:
            Preprocessed text
            
        Raises:
            ModelEvaluationError: If preprocessing fails
        """
        try:
            if not isinstance(text, str):
                text = str(text)
            
            # Convert to lowercase
            words = text.lower().split()
            
            # Remove stop words
            words = [word for word in words if word not in self.stop_words]
            
            # Remove numbers
            words = [word for word in words if not word.isdigit()]
            
            # Remove punctuations
            punctuation_pattern = r'[%s]' % re.escape(r"""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~""")
            words = [re.sub(punctuation_pattern, ' ', word) for word in words]
            
            # Remove URLs
            url_pattern = re.compile(r'https?://\S+|www\.\S+')
            words = [url_pattern.sub(r'', word) for word in words]
            
            # Lemmatization
            words = [self.lemmatizer.lemmatize(word) for word in words]
            
            # Remove extra whitespace and join
            processed_text = ' '.join(words).strip()
            processed_text = re.sub(r'\s+', ' ', processed_text)
            
            return processed_text
            
        except Exception as e:
            self.logger.error(f"Text preprocessing failed: {e}")
            raise ModelEvaluationError(f"Preprocessing failed: {e}")
    
    def predict_single(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Make prediction on single text input.
        
        Args:
            text: Text input for prediction
            
        Returns:
            Dictionary containing prediction and confidence scores
            
        Raises:
            ModelEvaluationError: If prediction fails
        """
        try:
            self.logger.info(f"Making prediction on text: {text[:100]}...")
            
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            if not processed_text:
                raise ModelEvaluationError("Text became empty after preprocessing")
            
            # Vectorize text
            text_vector = self.vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = self.model.predict(text_vector)[0]
            prediction_proba = self.model.predict_proba(text_vector)[0]
            
            # Convert prediction to sentiment label
            sentiment = "happiness" if prediction == 1 else "sadness"
            confidence = float(max(prediction_proba))
            
            # Create result dictionary
            result = {
                "text": text,
                "processed_text": processed_text,
                "sentiment": sentiment,
                "prediction": int(prediction),
                "confidence": confidence,
                "probabilities": {
                    "sadness": float(prediction_proba[0]),
                    "happiness": float(prediction_proba[1])
                }
            }
            
            self.logger.info(f"Prediction completed: {sentiment} (confidence: {confidence:.4f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Single prediction failed: {e}")
            raise ModelEvaluationError(f"Prediction failed: {e}")
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Make predictions on batch of text inputs.
        
        Args:
            texts: List of text inputs for prediction
            
        Returns:
            List of prediction dictionaries
            
        Raises:
            ModelEvaluationError: If batch prediction fails
        """
        try:
            self.logger.info(f"Making batch predictions on {len(texts)} texts")
            
            if not texts:
                raise ModelEvaluationError("Empty text list provided")
            
            results = []
            
            for i, text in enumerate(texts):
                try:
                    result = self.predict_single(text)
                    result["batch_index"] = i
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to predict text at index {i}: {e}")
                    # Add error result for failed prediction
                    results.append({
                        "text": text,
                        "error": str(e),
                        "batch_index": i
                    })
            
            successful_predictions = sum(1 for r in results if "error" not in r)
            self.logger.info(f"Batch prediction completed: {successful_predictions}/{len(texts)} successful")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            raise ModelEvaluationError(f"Batch prediction failed: {e}")
    
    def predict_from_file(self, file_path: str, text_column: str = 'text') -> pd.DataFrame:
        """
        Make predictions on texts from a CSV file.
        
        Args:
            file_path: Path to CSV file containing texts
            text_column: Name of column containing text data
            
        Returns:
            DataFrame with original texts and predictions
            
        Raises:
            ModelEvaluationError: If file prediction fails
        """
        try:
            self.logger.info(f"Making predictions from file: {file_path}")
            
            if not os.path.exists(file_path):
                raise ModelEvaluationError(f"File not found: {file_path}")
            
            # Load data
            df = pd.read_csv(file_path)
            
            if text_column not in df.columns:
                raise ModelEvaluationError(f"Column '{text_column}' not found in file")
            
            # Get texts
            texts = df[text_column].fillna('').tolist()
            
            # Make predictions
            results = self.predict_batch(texts)
            
            # Create results DataFrame
            predictions_df = pd.DataFrame(results)
            
            # Merge with original data
            final_df = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)
            
            self.logger.info(f"File prediction completed: {len(final_df)} predictions")
            
            return final_df
            
        except Exception as e:
            self.logger.error(f"File prediction failed: {e}")
            raise ModelEvaluationError(f"File prediction failed: {e}")
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        try:
            model_info = {
                "model_type": type(self.model).__name__,
                "vectorizer_type": type(self.vectorizer).__name__,
                "feature_count": len(self.vectorizer.get_feature_names_out()),
                "model_file_size": os.path.getsize('models/model.pkl'),
                "vectorizer_file_size": os.path.getsize('data/features/vectorizer.pkl'),
                "supported_sentiments": ["sadness", "happiness"]
            }
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}


def main():
    """
    Main function to demonstrate model inference.
    """
    try:
        # Initialize predictor
        predictor = SentimentPredictor()
        
        # Print model information
        print("Model Information:")
        model_info = predictor.get_model_info()
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        print()
        
        # Example predictions
        test_texts = [
            "I am so happy today! This is amazing!",
            "I feel really sad and disappointed.",
            "The weather is okay I guess.",
            "This is the best day of my life!",
            "I'm feeling quite down about this."
        ]
        
        print("Example Predictions:")
        print("=" * 50)
        
        for text in test_texts:
            result = predictor.predict_single(text)
            print(f"Text: {text}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities: Sadness={result['probabilities']['sadness']:.4f}, "
                  f"Happiness={result['probabilities']['happiness']:.4f}")
            print("-" * 50)
        
    except Exception as e:
        print(f"Error in model inference: {e}")
        raise


if __name__ == '__main__':
    main()
