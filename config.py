"""
AeroGuard Configuration
========================
Central configuration file for the AeroGuard application.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data paths
DATA_DIR = BASE_DIR / 'data'
TRAIN_DATA_PATH = DATA_DIR / 'train_FD001.txt'
TEST_DATA_PATH = DATA_DIR / 'test_FD001.txt'

# Model paths
MODELS_DIR = BASE_DIR / 'models'
MODEL_PATH = MODELS_DIR / 'balanced_xgboost_model.pkl'
SCALER_PATH = MODELS_DIR / 'balanced_scaler.pkl'
ENGINEER_PATH = MODELS_DIR / 'balanced_engineer.pkl'

# Fallback model paths
FALLBACK_MODEL_PATH = MODELS_DIR / 'xgboost_model.pkl'
FALLBACK_SCALER_PATH = MODELS_DIR / 'scaler.pkl'

# Flask configuration
FLASK_CONFIG = {
    'DEBUG': True,
    'HOST': '0.0.0.0',
    'PORT': 5000,
    'THREADED': True
}

# Model configuration
MODEL_CONFIG = {
    'max_rul': 125,  # Maximum RUL value for normalization
    'health_thresholds': {
        'healthy': 70,
        'warning': 40
    }
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'rolling_windows': [10, 20],
    'ema_spans': [10],
    'diff_periods': [5]
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}
