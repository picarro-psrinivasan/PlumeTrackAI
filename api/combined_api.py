#!/usr/bin/env python3
"""
Combined PlumeTrackAI API
Provides both wind prediction and plume travel endpoints in a single service.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import sys
import os
from datetime import datetime
import numpy as np

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('.')

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
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

# Import prediction functions
try:
    from scripts.forecast_weighted_prediction import forecast_weighted_prediction, calculate_plume_forecast_weighted
    from prediction.wind_predictor import get_recent_wind_data as original_get_recent_wind_data
    
    def get_recent_wind_data_fixed(data_file='../../data/15_min_avg_1site_1ms.csv', hours_back=6):
        """Fixed version of get_recent_wind_data that works from API directory."""
        return original_get_recent_wind_data(data_file, hours_back)
    
    # Replace the function in the module
    import prediction.wind_predictor
    prediction.wind_predictor.get_recent_wind_data = get_recent_wind_data_fixed
    
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.forecast_weighted_prediction import forecast_weighted_prediction, calculate_plume_forecast_weighted

# Initialize FastAPI app
app = FastAPI(
    title="PlumeTrackAI Combined API",
    description="Combined API for wind predictions and plume travel calculations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ============================================================================
# WIND PREDICTION MODELS
# ============================================================================

class WindPredictionRequest(BaseModel):
    """Request model for wind prediction."""
    latitude: float = Field(30.452, ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(-91.188, ge=-180, le=180, description="Longitude coordinate")
    hours_ahead: int = Field(6, ge=1, le=24, description="Hours ahead for prediction")
    forecast_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for forecast data (0-1)")
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Minimum confidence threshold")
    model_path: Optional[str] = Field(None, description="Path to trained model (optional)")

class WindPredictionResponse(BaseModel):
    """Response model for wind prediction."""
    success: bool
    timestamp: str
    location: Dict[str, float]
    hours_ahead: int
    base_prediction: Dict[str, Any]
    weighted_prediction: Dict[str, Any]
    forecast_data: Optional[Dict[str, Any]]
    forecast_confidence: Optional[float]
    forecast_weight_used: Optional[float]
    validation: Optional[Dict[str, Any]]
    base_validation: Optional[Dict[str, Any]]
    improvement: Optional[float]
    hourly_predictions: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

# ============================================================================
# PLUME TRAVEL MODELS
# ============================================================================

class PlumePredictionRequest(BaseModel):
    """Request model for plume prediction."""
    source_latitude: float = Field(30.452, ge=-90, le=90, description="Source latitude coordinate")
    source_longitude: float = Field(-91.188, ge=-180, le=180, description="Source longitude coordinate")
    risk_latitude: float = Field(30.458, ge=-90, le=90, description="Risk zone latitude coordinate")
    risk_longitude: float = Field(-91.182, ge=-180, le=180, description="Risk zone longitude coordinate")
    hours_ahead: int = Field(6, ge=1, le=24, description="Hours ahead for prediction")
    forecast_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for forecast data (0-1)")
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Minimum confidence threshold")
    model_path: Optional[str] = Field(None, description="Path to trained model (optional)")

class PlumePredictionResponse(BaseModel):
    """Response model for plume prediction."""
    success: bool
    timestamp: str
    source_location: Dict[str, float]
    risk_location: Dict[str, float]
    plume_travel: Dict[str, Any]
    geojson: Optional[Dict[str, Any]] = None
    wind_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ============================================================================
# ROOT ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PlumeTrackAI Combined API",
        "version": "1.0.0",
        "description": "Combined API for wind predictions and plume travel calculations",
        "endpoints": {
            "wind_prediction": {
                "POST /wind/predict": "Full wind prediction with forecast weighting",
                "GET /wind/predict/simple": "Simple wind prediction"
            },
            "plume_travel": {
                "POST /plume/predict": "Full plume travel prediction",
                "GET /plume/predict/simple": "Simple plume travel prediction"
            },
            "health": {
                "GET /health": "Health check"
            }
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "wind_prediction": "available",
            "plume_travel": "available"
        }
    }

# ============================================================================
# WIND PREDICTION ENDPOINTS
# ============================================================================

@app.post("/wind/predict", response_model=WindPredictionResponse)
async def predict_wind_forecast_weighted(request: WindPredictionRequest):
    """Full wind prediction with forecast weighting."""
    try:
        # Call the forecast-weighted prediction function
        result = forecast_weighted_prediction(
            latitude=request.latitude,
            longitude=request.longitude,
            hours_ahead=request.hours_ahead,
            forecast_weight=request.forecast_weight,
            confidence_threshold=request.confidence_threshold,
            model_path=request.model_path
        )
        
        # Convert numpy types for JSON serialization
        result = convert_numpy_types(result)
        
        # Add hourly predictions to the response
        if 'forecast_data' in result and result['forecast_data']:
            forecast_data = result['forecast_data']
            
            # Create hourly predictions array
            hourly_predictions = []
            for hour in range(request.hours_ahead):
                if hour < len(forecast_data.get('wind_speed_forecast', [])):
                    # Use forecast data for hourly predictions
                    wind_speed = forecast_data['wind_speed_forecast'][hour]
                    wind_direction = forecast_data['wind_direction_forecast'][hour]
                    
                    # Apply the same weighting logic as the final prediction
                    if 'base_prediction' in result and 'weighted_prediction' in result:
                        base_speed = result['base_prediction'].get('wind_speed_mph', wind_speed)
                        base_direction = result['base_prediction'].get('wind_direction_degrees', wind_direction)
                        forecast_weight = result.get('forecast_weight_used', 0.0)
                        
                        # Calculate weighted values for this hour
                        weighted_speed = (1 - forecast_weight) * base_speed + forecast_weight * wind_speed
                        
                        # For wind direction, handle circular nature
                        import numpy as np
                        base_dir_rad = np.radians(base_direction)
                        forecast_dir_rad = np.radians(wind_direction)
                        
                        # Convert to Cartesian coordinates
                        base_x = np.cos(base_dir_rad)
                        base_y = np.sin(base_dir_rad)
                        forecast_x = np.cos(forecast_dir_rad)
                        forecast_y = np.sin(forecast_dir_rad)
                        
                        # Weighted average in Cartesian space
                        weighted_x = (1 - forecast_weight) * base_x + forecast_weight * forecast_x
                        weighted_y = (1 - forecast_weight) * base_y + forecast_weight * forecast_y
                        
                        # Convert back to degrees
                        weighted_direction = np.degrees(np.arctan2(weighted_y, weighted_x))
                        if weighted_direction < 0:
                            weighted_direction += 360
                    else:
                        weighted_speed = wind_speed
                        weighted_direction = wind_direction
                    
                    hourly_predictions.append({
                        "hour": hour + 1,
                        "wind_speed_mph": round(weighted_speed, 2),
                        "wind_direction_degrees": round(weighted_direction, 1),
                        "forecast_speed_mph": round(wind_speed, 2),
                        "forecast_direction_degrees": round(wind_direction, 1)
                    })
            
            # Add hourly predictions to the result
            result['hourly_predictions'] = hourly_predictions
        
        return WindPredictionResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            location={"latitude": request.latitude, "longitude": request.longitude},
            hours_ahead=request.hours_ahead,
            **result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/wind/predict/simple")
async def simple_wind_prediction(
    latitude: float = 30.452,
    longitude: float = -91.188,
    hours_ahead: int = 6
):
    """Simple wind prediction without forecast weighting."""
    try:
        # Call the forecast-weighted prediction function with default settings
        result = forecast_weighted_prediction(
            latitude=latitude,
            longitude=longitude,
            hours_ahead=hours_ahead,
            forecast_weight=0.0,  # No forecast weighting for simple prediction
            confidence_threshold=0.8
        )
        
        # Extract just the base prediction for simple response
        base_pred = result.get('base_prediction', {})
        
        return {
            "success": True,
            "wind_predictions": [
                {
                    "hour": i + 1,
                    "speed": base_pred.get('wind_speeds', [])[i] if i < len(base_pred.get('wind_speeds', [])) else 0,
                    "direction": base_pred.get('wind_directions', [])[i] if i < len(base_pred.get('wind_directions', [])) else 0
                }
                for i in range(hours_ahead)
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simple prediction failed: {str(e)}")

# ============================================================================
# PLUME TRAVEL ENDPOINTS
# ============================================================================

@app.post("/plume/predict", response_model=PlumePredictionResponse)
async def predict_plume_forecast_weighted(request: PlumePredictionRequest):
    """Full plume travel prediction with forecast weighting."""
    try:
        # Call the plume forecast-weighted prediction function
        result = calculate_plume_forecast_weighted(
            source_lat=request.source_latitude,
            source_lon=request.source_longitude,
            risk_lat=request.risk_latitude,
            risk_lon=request.risk_longitude,
            latitude=request.source_latitude,  # Use source location for forecast data
            longitude=request.source_longitude,  # Use source location for forecast data
            hours_ahead=request.hours_ahead,
            forecast_weight=request.forecast_weight,
            confidence_threshold=request.confidence_threshold,
            model_path=request.model_path
        )
        
        # Convert numpy types for JSON serialization
        result = convert_numpy_types(result)
        
        # Extract only the essential plume travel information
        plume_travel = result.get('plume_travel', {})
        
        # Create response with plume travel data and wind predictions
        wind_data = result.get('wind_predictions', {})
        
        # Add hourly predictions to match wind prediction endpoint
        if 'forecast_data' in wind_data and wind_data['forecast_data']:
            forecast_data = wind_data['forecast_data']
            
            # Create hourly predictions array (same logic as wind prediction endpoint)
            hourly_predictions = []
            for hour in range(request.hours_ahead):
                if hour < len(forecast_data.get('wind_speed_forecast', [])):
                    # Use forecast data for hourly predictions
                    wind_speed = forecast_data['wind_speed_forecast'][hour]
                    wind_direction = forecast_data['wind_direction_forecast'][hour]
                    
                    # Apply the same weighting logic as the wind prediction endpoint
                    if 'base_prediction' in wind_data and 'weighted_prediction' in wind_data:
                        base_speed = wind_data['base_prediction'].get('wind_speed_mph', wind_speed)
                        base_direction = wind_data['base_prediction'].get('wind_direction_degrees', wind_direction)
                        forecast_weight = wind_data.get('forecast_weight_used', 0.0)
                        
                        # Calculate weighted values for this hour
                        weighted_speed = (1 - forecast_weight) * base_speed + forecast_weight * wind_speed
                        
                        # For wind direction, handle circular nature
                        import numpy as np
                        base_dir_rad = np.radians(base_direction)
                        forecast_dir_rad = np.radians(wind_direction)
                        
                        # Convert to Cartesian coordinates
                        base_x = np.cos(base_dir_rad)
                        base_y = np.sin(base_dir_rad)
                        forecast_x = np.cos(forecast_dir_rad)
                        forecast_y = np.sin(forecast_dir_rad)
                        
                        # Weighted average in Cartesian space
                        weighted_x = (1 - forecast_weight) * base_x + forecast_weight * forecast_x
                        weighted_y = (1 - forecast_weight) * base_y + forecast_weight * forecast_y
                        
                        # Convert back to degrees
                        weighted_direction = np.degrees(np.arctan2(weighted_y, weighted_x))
                        if weighted_direction < 0:
                            weighted_direction += 360
                    else:
                        weighted_speed = wind_speed
                        weighted_direction = wind_direction
                    
                    hourly_predictions.append({
                        "hour": hour + 1,
                        "wind_speed_mph": round(weighted_speed, 2),
                        "wind_direction_degrees": round(weighted_direction, 1),
                        "forecast_speed_mph": round(wind_speed, 2),
                        "forecast_direction_degrees": round(wind_direction, 1)
                    })
            
            # Add hourly predictions to wind_data
            wind_data['hourly_predictions'] = hourly_predictions
        
        simplified_result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'source_location': {"latitude": request.source_latitude, "longitude": request.source_longitude},
            'risk_location': {"latitude": request.risk_latitude, "longitude": request.risk_longitude},
            'plume_travel': {
                'arrival_time_hours': plume_travel.get('arrival_time_hours'),
                'will_reach_destination': plume_travel.get('will_reach_destination', False),
                'total_distance_km': plume_travel.get('total_distance_km'),
                'bearing_degrees': plume_travel.get('bearing_degrees'),
                'travel_log': plume_travel.get('travel_log', [])
            },
            'geojson': result.get('geojson'),
            'wind_data': wind_data  # Include wind predictions with hourly_predictions
        }
        
        return simplified_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plume prediction failed: {str(e)}")

@app.get("/plume/predict/simple")
async def simple_plume_prediction(
    source_lat: float = 30.452,
    source_lon: float = -91.188,
    risk_lat: float = 30.458,
    risk_lon: float = -91.182,
    hours_ahead: int = 6
):
    """Simple plume travel prediction without forecast weighting."""
    try:
        # Call the plume forecast-weighted prediction function with default settings
        result = calculate_plume_forecast_weighted(
            source_lat=source_lat,
            source_lon=source_lon,
            risk_lat=risk_lat,
            risk_lon=risk_lon,
            latitude=source_lat,
            longitude=source_lon,
            hours_ahead=hours_ahead,
            forecast_weight=0.0,  # No forecast weighting for simple prediction
            confidence_threshold=0.8
        )
        
        # Extract just the essential plume travel info
        plume_travel = result.get('plume_travel', {})
        
        # Extract wind predictions if available
        hourly_wind_predictions = []
        if 'wind_predictions' in result and 'forecast_data' in result.get('wind_predictions', {}):
            wind_results = result['wind_predictions']
            forecast_data = wind_results.get('forecast_data', {})
            
            # Get base and weighted predictions
            base_prediction = wind_results.get('base_prediction', {})
            weighted_prediction = wind_results.get('weighted_prediction', {})
            forecast_weight = wind_results.get('forecast_weight_used', 0.0)
            
            # Create hourly predictions array
            for hour in range(hours_ahead):
                if hour < len(forecast_data.get('wind_speed_forecast', [])):
                    # Use forecast data for hourly predictions
                    wind_speed = forecast_data['wind_speed_forecast'][hour]
                    wind_direction = forecast_data['wind_direction_forecast'][hour]
                    
                    # Apply the same weighting logic as the final prediction
                    if base_prediction and weighted_prediction:
                        base_speed = base_prediction.get('wind_speed_mph', wind_speed)
                        base_direction = base_prediction.get('wind_direction_degrees', wind_direction)
                        
                        # Calculate weighted values for this hour
                        weighted_speed = (1 - forecast_weight) * base_speed + forecast_weight * wind_speed
                        
                        # For wind direction, handle circular nature
                        import numpy as np
                        base_dir_rad = np.radians(base_direction)
                        forecast_dir_rad = np.radians(wind_direction)
                        
                        # Convert to Cartesian coordinates
                        base_x = np.cos(base_dir_rad)
                        base_y = np.sin(base_dir_rad)
                        forecast_x = np.cos(forecast_dir_rad)
                        forecast_y = np.sin(forecast_dir_rad)
                        
                        # Weighted average in Cartesian space
                        weighted_x = (1 - forecast_weight) * base_x + forecast_weight * forecast_x
                        weighted_y = (1 - forecast_weight) * base_y + forecast_weight * forecast_y
                        
                        # Convert back to degrees
                        weighted_direction = np.degrees(np.arctan2(weighted_y, weighted_x))
                        if weighted_direction < 0:
                            weighted_direction += 360
                    else:
                        weighted_speed = wind_speed
                        weighted_direction = wind_direction
                    
                    hourly_wind_predictions.append({
                        "hour": hour + 1,
                        "wind_speed_mph": round(weighted_speed, 2),
                        "wind_direction_degrees": round(weighted_direction, 1),
                        "forecast_speed_mph": round(wind_speed, 2),
                        "forecast_direction_degrees": round(wind_direction, 1),
                        "base_speed_mph": round(base_prediction.get('wind_speed_mph', 0), 2),
                        "base_direction_degrees": round(base_prediction.get('wind_direction_degrees', 0), 1)
                    })
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "source_location": {"latitude": source_lat, "longitude": source_lon},
            "risk_location": {"latitude": risk_lat, "longitude": risk_lon},
            "plume_travel": {
                "arrival_time_hours": plume_travel.get('arrival_time_hours'),
                "will_reach_destination": plume_travel.get('will_reach_destination', False),
                "total_distance_km": plume_travel.get('total_distance_km'),
                "bearing_degrees": plume_travel.get('bearing_degrees'),
                "travel_log": plume_travel.get('travel_log', [])
            },
            "hourly_wind_predictions": hourly_wind_predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simple plume prediction failed: {str(e)}")

# ============================================================================
# CORS OPTIONS ENDPOINTS
# ============================================================================

@app.options("/wind/predict")
async def wind_prediction_options():
    """CORS options for wind prediction."""
    return {"message": "OK"}

@app.options("/plume/predict")
async def plume_prediction_options():
    """CORS options for plume prediction."""
    return {"message": "OK"}

@app.options("/{path:path}")
async def catch_all_options(path: str):
    """Catch-all CORS options handler."""
    return {"message": "OK"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 