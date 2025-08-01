#!/usr/bin/env python3
"""
FastAPI endpoint for Forecast-Weighted Prediction
Provides REST API access to forecast-weighted wind predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import sys
import os
from datetime import datetime
import numpy as np

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('.')

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types.
    This is needed because Pydantic cannot serialize numpy types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

try:
    from scripts.forecast_weighted_prediction import forecast_weighted_prediction
    # Fix data path for API usage
    import sys
    import os
    # Add the correct data path
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))
    
    # Monkey patch the data loading function to use correct path
    import pandas as pd
    from prediction.wind_predictor import get_recent_wind_data as original_get_recent_wind_data
    
    def get_recent_wind_data_fixed(data_file='../../data/15_min_avg_1site_1ms.csv', hours_back=6):
        """Fixed version of get_recent_wind_data that works from API directory."""
        return original_get_recent_wind_data(data_file, hours_back)
    
    # Replace the function in the module
    import prediction.wind_predictor
    prediction.wind_predictor.get_recent_wind_data = get_recent_wind_data_fixed
    
except ImportError:
    # If running directly, adjust paths
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.forecast_weighted_prediction import forecast_weighted_prediction

# Initialize FastAPI app
app = FastAPI(
    title="PlumeTrackAI Forecast-Weighted Prediction API",
    description="API for making forecast-weighted wind predictions using LSTM models and weather forecasts",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    """Request model for wind prediction."""
    latitude: float = Field(30.452, ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(-91.188, ge=-180, le=180, description="Longitude coordinate")
    hours_ahead: int = Field(6, ge=1, le=24, description="Hours ahead for prediction")
    forecast_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for forecast data (0-1)")
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Minimum confidence threshold")
    model_path: Optional[str] = Field(None, description="Path to trained model (optional)")

class PredictionResponse(BaseModel):
    """Response model for wind prediction."""
    success: bool
    timestamp: str
    location: Dict[str, float]
    hours_ahead: int
    
    # Base LSTM prediction
    base_prediction: Dict[str, Any]
    
    # Forecast-weighted prediction
    weighted_prediction: Dict[str, Any]
    
    # Forecast data
    forecast_data: Optional[Dict[str, Any]]
    forecast_confidence: Optional[float]
    forecast_weight_used: Optional[float]
    
    # Validation results
    validation: Optional[Dict[str, Any]]
    base_validation: Optional[Dict[str, Any]]
    improvement: Optional[float]
    
    # Error handling
    error: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PlumeTrackAI Forecast-Weighted Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make forecast-weighted wind prediction",
            "/health": "GET - Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "forecast-weighted-prediction"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_wind_forecast_weighted(request: PredictionRequest):
    """
    Make forecast-weighted wind prediction.
    
    This endpoint combines LSTM model predictions with weather forecast data
    to provide more accurate wind predictions.
    
    Args:
        request: PredictionRequest containing location and parameters
        
    Returns:
        PredictionResponse with base and weighted predictions
    """
    try:
        print(f"🔍 Received prediction request for {request.latitude}°N, {request.longitude}°E")
        
        # Call forecast-weighted prediction function with fixed data path
        # First, let's fix the data path issue by monkey patching the function
        import prediction.wind_predictor
        original_get_recent_wind_data = prediction.wind_predictor.get_recent_wind_data
        
        def get_recent_wind_data_fixed(data_file='../../data/15_min_avg_1site_1ms.csv', hours_back=6):
            return original_get_recent_wind_data(data_file, hours_back)
        
        # Replace the function in the module
        prediction.wind_predictor.get_recent_wind_data = get_recent_wind_data_fixed
        print("get_recent_wind_data_fixed success",get_recent_wind_data_fixed)
        
        # Set default model path for API usage
        if request.model_path is None:
            request.model_path = 'trained_models/wind_lstm_model.pth'
        
        # Create a custom forecast-weighted prediction function that works from API directory
        def custom_forecast_weighted_prediction(**kwargs):
            # Import the original function
            from scripts.forecast_weighted_prediction import forecast_weighted_prediction as original_func
            
            # Call the original function with fixed paths
            return original_func(**kwargs)
        
        results = custom_forecast_weighted_prediction(
            model_path=request.model_path,
            latitude=request.latitude,
            longitude=request.longitude,
            hours_ahead=request.hours_ahead,
            forecast_weight=request.forecast_weight,
            confidence_threshold=request.confidence_threshold
        )
        
        if results is None:
            raise HTTPException(
                status_code=500,
                detail="Prediction failed - could not generate results"
            )
        
        # Extract base prediction
        base_prediction = results.get('base_prediction', {})
        if not base_prediction:
            raise HTTPException(
                status_code=500,
                detail="Base prediction not available"
            )
        
        # Extract weighted prediction
        weighted_prediction = results.get('weighted_prediction', {})
        if not weighted_prediction:
            raise HTTPException(
                status_code=500,
                detail="Weighted prediction not available"
            )
        
        # Convert numpy types to Python native types for Pydantic serialization
        base_prediction_converted = convert_numpy_types(base_prediction)
        weighted_prediction_converted = convert_numpy_types(weighted_prediction)
        forecast_data_converted = convert_numpy_types(results.get('forecast_data'))
        validation_converted = convert_numpy_types(results.get('validation'))
        base_validation_converted = convert_numpy_types(results.get('base_validation'))
        
        # Build response
        response = PredictionResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            location={
                'latitude': request.latitude,
                'longitude': request.longitude
            },
            hours_ahead=request.hours_ahead,
            base_prediction=base_prediction_converted,
            weighted_prediction=weighted_prediction_converted,
            forecast_data=forecast_data_converted,
            forecast_confidence=convert_numpy_types(results.get('forecast_confidence')),
            forecast_weight_used=convert_numpy_types(results.get('forecast_weight_used')),
            validation=validation_converted,
            base_validation=base_validation_converted,
            improvement=convert_numpy_types(results.get('improvement'))
        )
        
        print(f"✅ Prediction completed successfully")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/predict/simple")
async def simple_prediction(
    latitude: float = 30.452,
    longitude: float = -91.188,
    hours_ahead: int = 6
):
    """
    Simple prediction endpoint with default parameters.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        hours_ahead: Hours ahead for prediction
        
    Returns:
        Simplified prediction results
    """
    try:
        results = forecast_weighted_prediction(
            latitude=latitude,
            longitude=longitude,
            hours_ahead=hours_ahead
        )
        
        if results is None:
            raise HTTPException(
                status_code=500,
                detail="Simple prediction failed"
            )
        
        # Convert numpy types to Python native types
        base_prediction = convert_numpy_types(results['base_prediction'])
        weighted_prediction = convert_numpy_types(results['weighted_prediction'])
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "base_prediction": {
                "wind_speed_mph": base_prediction['wind_speed_mph'],
                "wind_direction_degrees": base_prediction['wind_direction_degrees']
            },
            "weighted_prediction": {
                "wind_speed_mph": weighted_prediction['wind_speed_mph'],
                "wind_direction_degrees": weighted_prediction['wind_direction_degrees']
            },
            "forecast_confidence": convert_numpy_types(results.get('forecast_confidence')),
            "improvement": convert_numpy_types(results.get('improvement'))
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Simple prediction failed: {str(e)}"
        )

# Example usage and testing
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting PlumeTrackAI Forecast-Weighted Prediction API...")
    print("📖 API Documentation available at: http://localhost:8000/docs")
    print("🔗 Health check: http://localhost:8000/health")
    
    uvicorn.run(
        "fastapi_forecast_weighted:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 