ml-pipeline-production
==============================

Production-grade ML pipeline with DVC for text sentiment analysis

Project Organization
------------

```
ml-pipeline-production/
├── src/                          # Source code
│   ├── __init__.py
│   ├── exceptions.py               # Custom exception classes
│   ├── logger.py                  # Centralized logging configuration
│   ├── data/
│   │   ├── make_dataset.py       # Data ingestion pipeline
│   │   └── process_data.py       # Data preprocessing
│   ├── features/
│   │   └── build_features.py     # Feature engineering
│   ├── models/
│   │   ├── train_model.py        # Model training
│   │   └── predict_model.py      # Model evaluation
│   └── visualization/             # Data visualization
├── data/                         # Data directories
│   ├── raw/
│   ├── processed/
│   └── features/
├── logs/                         # Log files
├── params.yaml                   # Configuration parameters
├── dvc.yaml                     # DVC pipeline definition
├── requirements.txt              # Dependencies
├── tests/                       # Unit tests
├── notebooks/                   # Jupyter notebooks
└── docs/                        # Documentation
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

## Quick Start

1. **Setup Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train Model**:
   ```bash
   dvc init --no-scm
   dvc repro
   ```

3. **Make Predictions**:
   ```bash
   # Interactive demo
   python demo.py
   
   # Direct model usage
   python -m src.models.predict
   ```

4. **Configuration**:
   Edit `params.yaml` to customize pipeline parameters

## Model Inference

### **Using the Sentiment Predictor**

The trained model can be used for real-time sentiment analysis:

```python
from src.models.predict import SentimentPredictor

# Initialize predictor
predictor = SentimentPredictor()

# Single prediction
result = predictor.predict_single("I love this product!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch predictions
texts = ["Great service!", "Terrible experience"]
results = predictor.predict_batch(texts)
```

### **Interactive Demo**

Run the interactive demo to test the model:
```bash
python demo.py
```

Features:
- **Real-time predictions** with confidence scores
- **Batch processing** for multiple texts
- **File-based predictions** from CSV files
- **Comprehensive error handling** and logging

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
