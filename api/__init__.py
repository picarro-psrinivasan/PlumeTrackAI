"""
Wind Model API Package

This package contains API-related functionality for wind predictions:
- Wind data API operations
- Prediction API functions
- Forecast validation API
"""

from .ops_wind_data_api import (
    setup_openmeteo_client,
    load_wind_data,
    extract_forecast_wind_data,
    validate_prediction_with_forecast
)

from .api_predict import (
    predict_wind_with_validation,
    get_prediction_confidence,
    format_prediction_results
)

__all__ = [
    'setup_openmeteo_client',
    'load_wind_data', 
    'extract_forecast_wind_data',
    'validate_prediction_with_forecast',
    'predict_wind_with_validation',
    'get_prediction_confidence',
    'format_prediction_results'
] 