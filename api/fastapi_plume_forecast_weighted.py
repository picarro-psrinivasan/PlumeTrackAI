#!/usr/bin/env python3
"""
FastAPI endpoint for Forecast-Weighted Plume Prediction
Provides REST API access to forecast-weighted plume travel predictions.
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
    from scripts.forecast_weighted_prediction import calculate_plume_forecast_weighted
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
    from scripts.forecast_weighted_prediction import calculate_plume_forecast_weighted

# Initialize FastAPI app
app = FastAPI(
    title="PlumeTrackAI Forecast-Weighted Plume Prediction API",
    description="API for making forecast-weighted plume travel predictions using LSTM models and weather forecasts",
    version="1.0.0"
)

class PlumePredictionRequest(BaseModel):
    """Request model for plume prediction."""
    source_latitude: float = Field(30.452, ge=-90, le=90, description="Source latitude coordinate")
    source_longitude: float = Field(-91.188, ge=-180, le=180, description="Source longitude coordinate")
    risk_latitude: float = Field(30.458, ge=-90, le=90, description="Risk zone latitude coordinate")
    risk_longitude: float = Field(-91.182, ge=-180, le=180, description="Risk zone longitude coordinate")
    forecast_latitude: float = Field(30.452, ge=-90, le=90, description="Latitude for forecast data")
    forecast_longitude: float = Field(-91.188, ge=-180, le=180, description="Longitude for forecast data")
    hours_ahead: int = Field(6, ge=1, le=24, description="Hours ahead for prediction")
    forecast_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for forecast data (0-1)")
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Minimum confidence threshold")
    model_path: Optional[str] = Field(None, description="Path to trained model (optional)")

class PlumePredictionResponse(BaseModel):
    """Response model for plume prediction."""
    success: bool
    timestamp: str
    
    # Location information
    source_location: Dict[str, float]
    risk_location: Dict[str, float]
    
    # Wind predictions
    wind_predictions: Dict[str, Any]
    
    # Plume travel results
    plume_travel: Dict[str, Any]
    
    # Forecast information
    forecast_weight_used: Optional[float]
    forecast_confidence: Optional[float]
    improvement: Optional[float]
    
    # Error handling
    error: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PlumeTrackAI Forecast-Weighted Plume Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make forecast-weighted plume prediction",
            "/health": "GET - Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "forecast-weighted-plume-prediction"
    }

@app.post("/predict", response_model=PlumePredictionResponse)
async def predict_plume_forecast_weighted(request: PlumePredictionRequest):
    """
    Make forecast-weighted plume travel prediction.
    
    This endpoint combines LSTM model predictions with weather forecast data
    to provide more accurate plume travel time predictions.
    
    Args:
        request: PlumePredictionRequest containing source, risk zone, and parameters
        
    Returns:
        PlumePredictionResponse with wind predictions and plume travel results
    """
    try:
        print(f"🔍 Received plume prediction request")
        print(f"Source: ({request.source_latitude}, {request.source_longitude})")
        print(f"Risk Zone: ({request.risk_latitude}, {request.risk_longitude})")
        
        # Set default model path for API usage
        if request.model_path is None:
            request.model_path = 'trained_models/wind_lstm_model.pth'
        
        # Call forecast-weighted plume prediction function
        results = calculate_plume_forecast_weighted(
            source_lat=request.source_latitude,
            source_lon=request.source_longitude,
            risk_lat=request.risk_latitude,
            risk_lon=request.risk_longitude,
            model_path=request.model_path,
            latitude=request.forecast_latitude,
            longitude=request.forecast_longitude,
            hours_ahead=request.hours_ahead,
            forecast_weight=request.forecast_weight,
            confidence_threshold=request.confidence_threshold
        )
        
        if results is None:
            raise HTTPException(
                status_code=500,
                detail="Plume prediction failed - could not generate results"
            )
        
        # Convert numpy types to Python native types for Pydantic serialization
        wind_predictions_converted = convert_numpy_types(results.get('wind_predictions', {}))
        plume_travel_converted = convert_numpy_types(results.get('plume_travel', {}))
        
        # Build response
        response = PlumePredictionResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            source_location=results.get('source_location', {}),
            risk_location=results.get('risk_location', {}),
            wind_predictions=wind_predictions_converted,
            plume_travel=plume_travel_converted,
            forecast_weight_used=convert_numpy_types(results.get('forecast_weight_used')),
            forecast_confidence=convert_numpy_types(results.get('forecast_confidence')),
            improvement=convert_numpy_types(results.get('improvement'))
        )
        
        print(f"✅ Plume prediction completed successfully")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"❌ Plume prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Plume prediction failed: {str(e)}"
        )

@app.get("/predict/simple")
async def simple_plume_prediction(
    source_lat: float = 30.452,
    source_lon: float = -91.188,
    risk_lat: float = 30.458,
    risk_lon: float = -91.182,
    hours_ahead: int = 6
):
    """
    Simple plume prediction endpoint with default parameters.
    
    Args:
        source_lat, source_lon: Source coordinates
        risk_lat, risk_lon: Risk zone coordinates
        hours_ahead: Hours ahead for prediction
        
    Returns:
        Simplified plume prediction results
    """
    try:
        results = calculate_plume_forecast_weighted(
            source_lat=source_lat,
            source_lon=source_lon,
            risk_lat=risk_lat,
            risk_lon=risk_lon,
            hours_ahead=hours_ahead
        )
        
        if results is None:
            raise HTTPException(
                status_code=500,
                detail="Simple plume prediction failed"
            )
        
        # Convert numpy types to Python native types
        plume_travel = convert_numpy_types(results.get('plume_travel', {}))
        wind_predictions = convert_numpy_types(results.get('wind_predictions', {}))
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "source_location": results.get('source_location', {}),
            "risk_location": results.get('risk_location', {}),
            "plume_travel": {
                "arrival_time_hours": plume_travel.get('arrival_time_hours'),
                "will_reach_destination": plume_travel.get('will_reach_destination'),
                "travel_log": plume_travel.get('travel_log', []),
                "base_arrival_time_hours": plume_travel.get('base_arrival_time_hours'),
                "weighted_arrival_time_hours": plume_travel.get('weighted_arrival_time_hours'),
                "base_will_reach_destination": plume_travel.get('base_will_reach_destination'),
                "weighted_will_reach_destination": plume_travel.get('weighted_will_reach_destination'),
                "total_distance_km": plume_travel.get('total_distance_km'),
                "bearing_degrees": plume_travel.get('bearing_degrees')
            },
            "wind_prediction": {
                "wind_speed_mph": wind_predictions.get('weighted_prediction', {}).get('wind_speed_mph'),
                "wind_direction_degrees": wind_predictions.get('weighted_prediction', {}).get('wind_direction_degrees')
            },
            "forecast_confidence": convert_numpy_types(results.get('forecast_confidence')),
            "improvement": convert_numpy_types(results.get('improvement'))
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Simple plume prediction failed: {str(e)}"
        )

# Example usage and testing
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting PlumeTrackAI Forecast-Weighted Plume Prediction API...")
    print("📖 API Documentation available at: http://localhost:8001/docs")
    print("🔗 Health check: http://localhost:8001/health")
    
    uvicorn.run(
        "fastapi_plume_forecast_weighted:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    ) 