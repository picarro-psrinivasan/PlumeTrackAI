"""
Wind Model Package

This package contains all wind model related functionality including:
- LSTM model implementation
- Data loading utilities
- Prediction functionality
- Training scripts
- Wind data API operations
- Validation and testing tools
"""

__version__ = "1.0.0"
__author__ = "PlumeTrackAI Team"

# Import main functionality
from .api import (
    predict_wind_with_validation,
    get_prediction_confidence,
    format_prediction_results,
    extract_forecast_wind_data,
    validate_prediction_with_forecast
)

from .validation import (
    predict_and_validate,
    predict_with_confidence_threshold
)

# Import core model functionality
from .src.lstm_model import WindLSTM
from .src.predict_wind import predict_wind_6hours_ahead, load_trained_model
from .src.load_data import load_and_preprocess_data

__all__ = [
    # API functions
    'predict_wind_with_validation',
    'get_prediction_confidence', 
    'format_prediction_results',
    'extract_forecast_wind_data',
    'validate_prediction_with_forecast',
    
    # Validation functions
    'predict_and_validate',
    'predict_with_confidence_threshold',
    
    # Core model functions
    'WindLSTM',
    'predict_wind_6hours_ahead',
    'load_trained_model',
    'load_and_preprocess_data'
] 