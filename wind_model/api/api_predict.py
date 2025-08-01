#!/usr/bin/env python3
"""
PlumeTrackAI - Wind Prediction API
Simple API functions for making validated wind predictions.
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Add src directory to path
sys.path.append('../src')

from src.predict_wind import load_trained_model, get_recent_wind_data, prepare_input_sequence, predict_wind_6hours_ahead
from .ops_wind_data_api import validate_prediction_with_forecast

def predict_wind_with_validation(
    latitude: float = 30.452,
    longitude: float = -91.188,
    hours_ahead: int = 6
) -> Dict[str, Any]:
    """
    Make wind prediction with automatic forecast validation.
    
    Args:
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        hours_ahead (int): Number of hours to predict ahead
    
    Returns:
        Dict containing prediction and validation results
    """
    try:
        # Load model
        model, scaler = load_trained_model()
        if model is None or scaler is None:
            return {'error': 'Model loading failed'}
        
        # Get recent data
        recent_data = get_recent_wind_data()
        if recent_data is None or recent_data.empty:
            return {'error': 'Data loading failed'}
        
        # Prepare input
        input_sequence = prepare_input_sequence(recent_data, scaler)
        if input_sequence is None:
            return {'error': 'Input preparation failed'}
        
        # Make prediction
        prediction = predict_wind_6hours_ahead(model, input_sequence, scaler)
        if prediction is None:
            return {'error': 'Prediction failed'}
        
        # Validate against forecast
        validation = validate_prediction_with_forecast(
            predicted_wind_speed=prediction['wind_speed_mph'],
            predicted_wind_direction=prediction['wind_direction_degrees'],
            latitude=latitude,
            longitude=longitude,
            hours_ahead=hours_ahead
        )
        
        return {
            'success': True,
            'prediction': prediction,
            'validation': validation,
            'timestamp': datetime.now().isoformat(),
            'location': {'latitude': latitude, 'longitude': longitude},
            'hours_ahead': hours_ahead
        }
        
    except Exception as e:
        return {
            'error': f'Prediction failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }

def get_prediction_confidence(results: Dict[str, Any]) -> str:
    """
    Get confidence level based on validation results.
    
    Args:
        results (Dict): Results from predict_wind_with_validation
    
    Returns:
        str: Confidence level (HIGH, MEDIUM, LOW, UNKNOWN)
    """
    if 'error' in results or 'validation' not in results:
        return 'UNKNOWN'
    
    validation = results['validation']
    if 'error' in validation:
        return 'UNKNOWN'
    
    if 'validation_metrics' not in validation:
        return 'UNKNOWN'
    
    overall_accuracy = validation['validation_metrics']['overall_accuracy']
    
    if overall_accuracy >= 90:
        return 'HIGH'
    elif overall_accuracy >= 80:
        return 'MEDIUM'
    elif overall_accuracy >= 70:
        return 'LOW'
    else:
        return 'VERY_LOW'

def format_prediction_results(results: Dict[str, Any]) -> str:
    """
    Format prediction results as a readable string.
    
    Args:
        results (Dict): Results from predict_wind_with_validation
    
    Returns:
        str: Formatted results string
    """
    if 'error' in results:
        return f"❌ Error: {results['error']}"
    
    prediction = results['prediction']
    validation = results['validation']
    
    output = []
    output.append("🎯 Wind Prediction Results")
    output.append("=" * 40)
    output.append(f"Wind Speed: {prediction['wind_speed_mph']:.2f} mph")
    output.append(f"Wind Direction: {prediction['wind_direction_degrees']:.1f}°")
    
    if 'error' not in validation:
        forecast = validation['forecast']
        metrics = validation['validation_metrics']
        
        output.append(f"\n🌤️ Forecast Comparison:")
        output.append(f"Forecast Speed: {forecast['wind_speed']:.2f} mph")
        output.append(f"Forecast Direction: {forecast['wind_direction']:.1f}°")
        output.append(f"\n📊 Validation Metrics:")
        output.append(f"Overall Accuracy: {metrics['overall_accuracy']:.1f}%")
        output.append(f"Wind Speed Accuracy: {metrics['wind_speed_accuracy']:.1f}%")
        output.append(f"Wind Direction Accuracy: {metrics['wind_direction_accuracy']:.1f}%")
        
        confidence = get_prediction_confidence(results)
        output.append(f"\n🎯 Confidence Level: {confidence}")
    else:
        output.append(f"\n⚠️ Validation: {validation['error']}")
    
    output.append(f"\n📍 Location: {results['location']['latitude']}°N, {results['location']['longitude']}°E")
    output.append(f"⏰ Prediction Horizon: {results['hours_ahead']} hours")
    output.append(f"🕐 Timestamp: {results['timestamp']}")
    
    return "\n".join(output)

# Example usage functions
def example_basic_prediction():
    """Example of basic prediction with validation."""
    print("=== Basic Prediction Example ===")
    results = predict_wind_with_validation()
    print(format_prediction_results(results))

def example_confidence_check():
    """Example of prediction with confidence checking."""
    print("=== Confidence Check Example ===")
    results = predict_wind_with_validation()
    confidence = get_prediction_confidence(results)
    print(f"Prediction Confidence: {confidence}")
    
    if confidence in ['HIGH', 'MEDIUM']:
        print("✅ Prediction is reliable")
    else:
        print("⚠️ Prediction may need verification")

if __name__ == "__main__":
    example_basic_prediction()
    print("\n" + "="*50)
    example_confidence_check() 