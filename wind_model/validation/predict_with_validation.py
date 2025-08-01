#!/usr/bin/env python3
"""
PlumeTrackAI - Wind Prediction with Forecast Validation
Integrated prediction system that automatically validates predictions against forecast API data.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add paths
sys.path.append('../src')
sys.path.append('../api')

# Handle imports based on how the script is run
try:
    from src.predict_wind import load_trained_model, get_recent_wind_data, prepare_input_sequence, predict_wind_6hours_ahead
    from api.ops_wind_data_api import validate_prediction_with_forecast, extract_forecast_wind_data
except ImportError:
    # If running directly, adjust paths
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.predict_wind import load_trained_model, get_recent_wind_data, prepare_input_sequence, predict_wind_6hours_ahead
    from api.ops_wind_data_api import validate_prediction_with_forecast, extract_forecast_wind_data

def predict_and_validate(
    latitude: float = 30.452,
    longitude: float = -91.188,
    hours_ahead: int = 6,
    show_forecast_details: bool = True
) -> dict:
    """
    Make wind prediction and automatically validate against forecast data.
    
    Args:
        latitude (float): Latitude coordinate for forecast validation
        longitude (float): Longitude coordinate for forecast validation
        hours_ahead (int): Number of hours to predict ahead
        show_forecast_details (bool): Whether to show detailed forecast information
    
    Returns:
        dict: Complete prediction and validation results
    """
    print("=== PlumeTrackAI Wind Prediction with Validation ===")
    print(f"Location: {latitude}°N, {longitude}°E")
    print(f"Prediction horizon: {hours_ahead} hours ahead")
    print("=" * 60)
    
    # Step 1: Load the trained model
    print("\n1️⃣ Loading trained model...")
    model, scaler = load_trained_model()
    
    if model is None or scaler is None:
        print("❌ Error: Could not load trained model")
        return {'error': 'Model loading failed'}
    
    print("✅ Model loaded successfully")
    
    # Step 2: Get recent wind data
    print("\n2️⃣ Loading recent wind data...")
    recent_data = get_recent_wind_data()
    
    if recent_data is None or recent_data.empty:
        print("❌ Error: Could not load recent wind data")
        return {'error': 'Data loading failed'}
    
    print(f"✅ Loaded {len(recent_data)} time steps of recent data")
    
    # Step 3: Prepare input sequence
    print("\n3️⃣ Preparing input sequence...")
    input_sequence = prepare_input_sequence(recent_data, scaler, sequence_length=24)
    
    if input_sequence is None:
        print("❌ Error: Could not prepare input sequence")
        return {'error': 'Input preparation failed'}
    
    print("✅ Input sequence prepared successfully")
    
    # Step 4: Make prediction
    print("\n4️⃣ Making wind prediction...")
    prediction = predict_wind_6hours_ahead(model, input_sequence, scaler)
    
    if prediction is None:
        print("❌ Error: Could not make prediction")
        return {'error': 'Prediction failed'}
    
    predicted_wind_speed = prediction['wind_speed_mph']
    predicted_wind_direction = prediction['wind_direction_degrees']
    
    print(f"✅ Prediction completed: {predicted_wind_speed:.2f} mph, {predicted_wind_direction:.1f}°")
    
    # Step 5: Get forecast data for validation
    print("\n5️⃣ Retrieving forecast data for validation...")
    forecast_data = extract_forecast_wind_data(latitude, longitude, hours_ahead)
    
    if forecast_data is None:
        print("⚠️ Warning: Could not retrieve forecast data for validation")
        return {
            'prediction': prediction,
            'validation': {'error': 'Forecast data unavailable'},
            'timestamp': datetime.now().isoformat()
        }
    
    print("✅ Forecast data retrieved successfully")
    
    # Step 6: Validate prediction against forecast
    print("\n6️⃣ Validating prediction against forecast...")
    validation_results = validate_prediction_with_forecast(
        predicted_wind_speed=predicted_wind_speed,
        predicted_wind_direction=predicted_wind_direction,
        latitude=latitude,
        longitude=longitude,
        hours_ahead=hours_ahead
    )
    
    if 'error' in validation_results:
        print(f"⚠️ Warning: Validation failed - {validation_results['error']}")
    else:
        print("✅ Validation completed successfully")
    
    # Step 7: Compile comprehensive results
    results = {
        'prediction': prediction,
        'validation': validation_results,
        'forecast_data': forecast_data if show_forecast_details else None,
        'model_info': {
            'location': {'latitude': latitude, 'longitude': longitude},
            'hours_ahead': hours_ahead,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Step 8: Display comprehensive results
    print("\n" + "=" * 60)
    print("🎯 PREDICTION & VALIDATION RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Model Prediction ({hours_ahead} hours ahead):")
    print(f"   Wind Speed: {predicted_wind_speed:.2f} mph")
    print(f"   Wind Direction: {predicted_wind_direction:.1f}°")
    
    if 'error' not in validation_results:
        forecast_wind_speed = validation_results['forecast']['wind_speed']
        forecast_wind_direction = validation_results['forecast']['wind_direction']
        
        print(f"\n🌤️ Forecast API ({hours_ahead} hours ahead):")
        print(f"   Wind Speed: {forecast_wind_speed:.2f} mph")
        print(f"   Wind Direction: {forecast_wind_direction:.1f}°")
        
        print(f"\n📈 Validation Metrics:")
        print(f"   Wind Speed Error: {validation_results['errors']['wind_speed_error']:.2f} mph ({validation_results['errors']['wind_speed_percentage_error']:.1f}%)")
        print(f"   Wind Direction Error: {validation_results['errors']['wind_direction_error']:.1f}° ({validation_results['errors']['wind_direction_percentage_error']:.1f}%)")
        print(f"   Overall Accuracy: {validation_results['validation_metrics']['overall_accuracy']:.1f}%")
        
        # Confidence assessment
        overall_accuracy = validation_results['validation_metrics']['overall_accuracy']
        if overall_accuracy >= 90:
            confidence = "🟢 HIGH"
            confidence_desc = "Excellent prediction accuracy"
        elif overall_accuracy >= 80:
            confidence = "🟡 MEDIUM"
            confidence_desc = "Good prediction accuracy"
        elif overall_accuracy >= 70:
            confidence = "🟠 MODERATE"
            confidence_desc = "Acceptable prediction accuracy"
        else:
            confidence = "🔴 LOW"
            confidence_desc = "Low prediction accuracy - consider retraining"
        
        print(f"\n🎯 Confidence Assessment:")
        print(f"   Level: {confidence}")
        print(f"   Description: {confidence_desc}")
        
    else:
        print(f"\n⚠️ Validation Status: {validation_results['error']}")
    
    if show_forecast_details and forecast_data:
        print(f"\n📋 Detailed Forecast ({hours_ahead} hours):")
        for i, (speed, direction, time) in enumerate(zip(
            forecast_data['wind_speed_forecast'],
            forecast_data['wind_direction_forecast'],
            forecast_data['forecast_times']
        ), 1):
            print(f"   Hour {i}: {speed:.2f} mph, {direction:.1f}° ({time.strftime('%H:%M UTC')})")
    
    print("\n" + "=" * 60)
    print("✅ Prediction with validation completed!")
    print("=" * 60)
    
    return results

def predict_with_confidence_threshold(
    confidence_threshold: float = 80.0,
    latitude: float = 30.452,
    longitude: float = -91.188,
    hours_ahead: int = 6
) -> dict:
    """
    Make prediction and only return results if confidence meets threshold.
    
    Args:
        confidence_threshold (float): Minimum accuracy percentage required
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        hours_ahead (int): Number of hours to predict ahead
    
    Returns:
        dict: Prediction results if confidence threshold met, otherwise error
    """
    print(f"=== PlumeTrackAI Prediction with Confidence Threshold ({confidence_threshold}%) ===")
    
    results = predict_and_validate(latitude, longitude, hours_ahead, show_forecast_details=False)
    
    if 'error' in results:
        return results
    
    if 'validation' in results and 'validation_metrics' in results['validation']:
        overall_accuracy = results['validation']['validation_metrics']['overall_accuracy']
        
        if overall_accuracy >= confidence_threshold:
            print(f"\n✅ Confidence threshold met! ({overall_accuracy:.1f}% >= {confidence_threshold}%)")
            return results
        else:
            print(f"\n❌ Confidence threshold not met! ({overall_accuracy:.1f}% < {confidence_threshold}%)")
            return {
                'error': f'Confidence threshold not met ({overall_accuracy:.1f}% < {confidence_threshold}%)',
                'prediction': results['prediction'],
                'validation': results['validation']
            }
    else:
        print("\n⚠️ Could not assess confidence - validation failed")
        return results

def main():
    """
    Main function to demonstrate prediction with validation.
    """
    print("=== PlumeTrackAI Wind Prediction with Forecast Validation ===")
    
    # Example 1: Standard prediction with validation
    print("\n" + "="*60)
    print("EXAMPLE 1: Standard Prediction with Validation")
    print("="*60)
    
    results1 = predict_and_validate()
    
    # Example 2: Prediction with confidence threshold
    print("\n" + "="*60)
    print("EXAMPLE 2: Prediction with Confidence Threshold (85%)")
    print("="*60)
    
    results2 = predict_with_confidence_threshold(confidence_threshold=85.0)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if 'error' not in results1:
        print("✅ Standard prediction completed successfully")
    else:
        print(f"❌ Standard prediction failed: {results1['error']}")
    
    if 'error' not in results2:
        print("✅ Confidence threshold prediction completed successfully")
    else:
        print(f"❌ Confidence threshold prediction failed: {results2['error']}")

if __name__ == "__main__":
    main() 