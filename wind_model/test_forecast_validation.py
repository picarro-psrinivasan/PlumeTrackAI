#!/usr/bin/env python3
"""
Test script for forecast validation functionality.
This script demonstrates how to extract wind forecast data and validate predictions.
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

from ops_wind_data_api import extract_forecast_wind_data, validate_prediction_with_forecast
from src.predict_wind import predict_wind_6hours_ahead, load_trained_model

def test_forecast_extraction():
    """
    Test the forecast extraction functionality.
    """
    print("=== Testing Forecast Extraction ===")
    
    try:
        # Extract forecast data for 6 hours ahead
        forecast_data = extract_forecast_wind_data(
            latitude=30.452,
            longitude=-91.188,
            hours_ahead=6
        )
        
        if forecast_data:
            print(f"\n✅ Successfully extracted forecast data!")
            print(f"Location: {forecast_data['location']['latitude']}°N, {forecast_data['location']['longitude']}°E")
            print(f"Hours ahead: {forecast_data['hours_ahead']}")
            print(f"Wind speeds (mph): {forecast_data['wind_speed_forecast']}")
            print(f"Wind directions (°): {forecast_data['wind_direction_forecast']}")
            print(f"Forecast times: {forecast_data['forecast_times']}")
        else:
            print("❌ Failed to extract forecast data")
            
        return forecast_data
        
    except Exception as e:
        print(f"❌ Error testing forecast extraction: {e}")
        return None

def test_prediction_validation():
    """
    Test the prediction validation functionality.
    """
    print("\n=== Testing Prediction Validation ===")
    
    try:
        # Load the trained model
        print("Loading trained model...")
        model, scaler = load_trained_model()
        
        if model is None or scaler is None:
            print("❌ Could not load trained model")
            return None
        
        # Make a prediction (this would normally use real recent data)
        # For testing, we'll use some sample predictions
        predicted_wind_speed = 5.2  # mph
        predicted_wind_direction = 180.0  # degrees
        
        print(f"Sample prediction: {predicted_wind_speed} mph, {predicted_wind_direction}°")
        
        # Validate the prediction against forecast
        validation_results = validate_prediction_with_forecast(
            predicted_wind_speed=predicted_wind_speed,
            predicted_wind_direction=predicted_wind_direction,
            latitude=30.452,
            longitude=-91.188,
            hours_ahead=6
        )
        
        if 'error' not in validation_results:
            print(f"\n✅ Validation completed successfully!")
            print(f"Overall accuracy: {validation_results['validation_metrics']['overall_accuracy']:.1f}%")
            return validation_results
        else:
            print(f"❌ Validation failed: {validation_results['error']}")
            return None
            
    except Exception as e:
        print(f"❌ Error testing prediction validation: {e}")
        return None

def test_real_prediction_validation():
    """
    Test validation with a real prediction from the model.
    """
    print("\n=== Testing Real Prediction Validation ===")
    
    try:
        # Load the trained model
        print("Loading trained model...")
        model, scaler = load_trained_model()
        
        if model is None or scaler is None:
            print("❌ Could not load trained model")
            return None
        
        # Get recent wind data and make a real prediction
        from src.predict_wind import get_recent_wind_data, prepare_input_sequence
        
        print("Getting recent wind data...")
        recent_data = get_recent_wind_data()
        
        if recent_data is None or recent_data.empty:
            print("❌ Could not get recent wind data")
            return None
        
        print(f"Recent data shape: {recent_data.shape}")
        
        # Prepare input sequence
        input_sequence = prepare_input_sequence(recent_data, scaler)
        
        if input_sequence is None:
            print("❌ Could not prepare input sequence")
            return None
        
        # Make prediction
        print("Making prediction...")
        prediction = predict_wind_6hours_ahead(model, input_sequence, scaler)
        
        if prediction is None:
            print("❌ Could not make prediction")
            return None
        
        predicted_wind_speed = prediction['wind_speed_mph']
        predicted_wind_direction = prediction['wind_direction_degrees']
        
        print(f"Real prediction: {predicted_wind_speed:.2f} mph, {predicted_wind_direction:.1f}°")
        
        # Validate the real prediction
        validation_results = validate_prediction_with_forecast(
            predicted_wind_speed=predicted_wind_speed,
            predicted_wind_direction=predicted_wind_direction,
            latitude=30.452,
            longitude=-91.188,
            hours_ahead=6
        )
        
        if 'error' not in validation_results:
            print(f"\n✅ Real prediction validation completed!")
            print(f"Overall accuracy: {validation_results['validation_metrics']['overall_accuracy']:.1f}%")
            return validation_results
        else:
            print(f"❌ Real prediction validation failed: {validation_results['error']}")
            return None
            
    except Exception as e:
        print(f"❌ Error testing real prediction validation: {e}")
        return None

def main():
    """
    Main function to run all tests.
    """
    print("=== PlumeTrackAI Forecast Validation Tests ===")
    
    # Test 1: Forecast extraction
    forecast_data = test_forecast_extraction()
    
    # Test 2: Sample prediction validation
    sample_validation = test_prediction_validation()
    
    # Test 3: Real prediction validation
    real_validation = test_real_prediction_validation()
    
    # Summary
    print("\n" + "="*50)
    print("=== Test Summary ===")
    print(f"Forecast extraction: {'✅ PASS' if forecast_data else '❌ FAIL'}")
    print(f"Sample validation: {'✅ PASS' if sample_validation else '❌ FAIL'}")
    print(f"Real validation: {'✅ PASS' if real_validation else '❌ FAIL'}")
    
    if real_validation and 'validation_metrics' in real_validation:
        print(f"\n🎯 Model Performance vs Forecast:")
        print(f"Wind Speed Accuracy: {real_validation['validation_metrics']['wind_speed_accuracy']:.1f}%")
        print(f"Wind Direction Accuracy: {real_validation['validation_metrics']['wind_direction_accuracy']:.1f}%")
        print(f"Overall Accuracy: {real_validation['validation_metrics']['overall_accuracy']:.1f}%")

if __name__ == "__main__":
    main() 