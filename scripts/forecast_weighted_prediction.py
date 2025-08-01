#!/usr/bin/env python3
"""
Forecast-Weighted Prediction
This module adjusts LSTM predictions based on forecast data without retraining.
"""

import torch
import numpy as np
import sys
import os

# Add paths
sys.path.append('..')
sys.path.append('../api')

# Handle imports based on how the script is run
try:
    from prediction.wind_predictor import load_trained_model, get_recent_wind_data, prepare_input_sequence, predict_wind_6hours_ahead
    from api.ops_wind_data_api import extract_forecast_wind_data, validate_prediction_with_forecast
except ImportError:
    # If running directly, adjust paths
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from prediction.wind_predictor import load_trained_model, get_recent_wind_data, prepare_input_sequence, predict_wind_6hours_ahead
    from api.ops_wind_data_api import extract_forecast_wind_data, validate_prediction_with_forecast

def forecast_weighted_prediction(
    model_path=None,
    latitude=30.452,
    longitude=-91.188,
    hours_ahead=6,
    forecast_weight=0.3,
    confidence_threshold=0.8
):
    # Set default model path based on how script is run
    if model_path is None:
        if os.path.exists('trained_models/wind_lstm_model.pth'):
            model_path = 'trained_models/wind_lstm_model.pth'
        else:
            model_path = 'trained_models/wind_lstm_model.pth'
    """
    Make forecast-weighted prediction by combining LSTM output with forecast data.
    
    Args:
        model_path: Path to trained model
        latitude: Latitude for forecast data
        longitude: Longitude for forecast data
        hours_ahead: Hours ahead for prediction
        forecast_weight: Weight given to forecast data (0-1)
        confidence_threshold: Minimum confidence for forecast weighting
        
    Returns:
        dict: Weighted prediction results
    """
    
    print("=== Forecast-Weighted Prediction ===")
    
    # Load model and make base prediction
    print("1️⃣ Loading model and making base prediction...")
    print("model_path",model_path, os.path.exists(model_path))
    model, scaler = load_trained_model(model_path)
    
    if model is None or scaler is None:
        print("❌ Could not load model")
        return None
    
    # Get recent data and make base prediction
    recent_data = get_recent_wind_data()
    if recent_data is None:
        print("❌ Could not load recent data")
        return None
    
    input_sequence = prepare_input_sequence(recent_data, scaler)
    if input_sequence is None:
        print("❌ Could not prepare input sequence")
        return None
    
    base_predictions = predict_wind_6hours_ahead(model, input_sequence, scaler)
    if base_predictions is None:
        print("❌ Could not make base prediction")
        return None
    
    # Get the prediction for the specific hour we want
    if hours_ahead <= len(base_predictions):
        base_prediction = base_predictions[hours_ahead - 1]  # 0-indexed, so hours_ahead-1
    else:
        # If hours_ahead is greater than available predictions, use the last one
        base_prediction = base_predictions[-1]
    
    print(f"✅ Base prediction for hour {hours_ahead}: {base_prediction['wind_speed_mph']:.2f} mph, {base_prediction['wind_direction_degrees']:.1f}°")
    
    # Get forecast data
    print("2️⃣ Retrieving forecast data...")
    forecast_data = extract_forecast_wind_data(latitude, longitude, hours_ahead)
    
    if forecast_data is None:
        print("⚠️ Could not retrieve forecast data, using base prediction only")
        return {
            'base_prediction': base_prediction,
            'weighted_prediction': base_prediction,  # Same as base when no forecast
            'method': 'base_only',
            'forecast_available': False,
            'forecast_confidence': 0.0,
            'forecast_weight_used': 0.0,
            'validation': None,
            'base_validation': None,
            'improvement': None
        }
    
    print("✅ Forecast data retrieved")
    
    # Calculate forecast confidence
    print("3️⃣ Calculating forecast confidence...")
    forecast_confidence = calculate_forecast_confidence(forecast_data)
    print(f"Forecast confidence: {forecast_confidence:.2f}")
    
    # Adjust weight based on confidence
    if forecast_confidence < confidence_threshold:
        adjusted_weight = forecast_weight * (forecast_confidence / confidence_threshold)
        print(f"⚠️ Low forecast confidence, reducing weight to: {adjusted_weight:.2f}")
    else:
        adjusted_weight = forecast_weight
        print(f"✅ High forecast confidence, using weight: {adjusted_weight:.2f}")
    
    # Combine predictions
    print("4️⃣ Combining predictions...")
    weighted_prediction = combine_predictions(
        base_prediction=base_prediction,
        forecast_data=forecast_data,
        forecast_weight=adjusted_weight,
        hours_ahead=hours_ahead
    )
    
    print(f"✅ Weighted prediction: {weighted_prediction['wind_speed_mph']:.2f} mph, {weighted_prediction['wind_direction_degrees']:.1f}°")
    
    # Validate weighted prediction
    print("5️⃣ Validating weighted prediction...")
    validation = validate_prediction_with_forecast(
        predicted_wind_speed=weighted_prediction['wind_speed_mph'],
        predicted_wind_direction=weighted_prediction['wind_direction_degrees'],
        latitude=latitude,
        longitude=longitude,
        hours_ahead=hours_ahead
    )
    
    # Compare with base prediction
    base_validation = validate_prediction_with_forecast(
        predicted_wind_speed=base_prediction['wind_speed_mph'],
        predicted_wind_direction=base_prediction['wind_direction_degrees'],
        latitude=latitude,
        longitude=longitude,
        hours_ahead=hours_ahead
    )
    
    results = {
        'base_prediction': base_prediction,
        'weighted_prediction': weighted_prediction,
        'forecast_data': forecast_data,
        'forecast_confidence': forecast_confidence,
        'forecast_weight_used': adjusted_weight,
        'validation': validation,
        'base_validation': base_validation,
        'improvement': None
    }
    
    # Calculate improvement
    if 'validation_metrics' in validation and 'validation_metrics' in base_validation:
        weighted_accuracy = validation['validation_metrics']['overall_accuracy']
        base_accuracy = base_validation['validation_metrics']['overall_accuracy']
        improvement = weighted_accuracy - base_accuracy
        
        results['improvement'] = improvement
        
        print(f"\n📊 Comparison Results:")
        print(f"Base prediction accuracy: {base_accuracy:.1f}%")
        print(f"Weighted prediction accuracy: {weighted_accuracy:.1f}%")
        print(f"Improvement: {improvement:+.1f}%")
        
        if improvement > 0:
            print("✅ Forecast weighting improved prediction!")
        else:
            print("⚠️ Forecast weighting did not improve prediction")
    
    return results

def calculate_forecast_confidence(forecast_data):
    """
    Calculate confidence in forecast data based on consistency and reliability.
    
    Args:
        forecast_data: Forecast data dictionary
        
    Returns:
        float: Confidence score (0-1)
    """
    
    wind_speeds = forecast_data['wind_speed_forecast']
    wind_directions = forecast_data['wind_direction_forecast']
    
    # Calculate consistency metrics
    speed_variance = np.var(wind_speeds)
    direction_variance = np.var(wind_directions)
    
    # Normalize variances (lower variance = higher confidence)
    max_speed_variance = 25.0  # mph^2
    max_direction_variance = 10000.0  # degrees^2
    
    speed_confidence = max(0, 1 - (speed_variance / max_speed_variance))
    direction_confidence = max(0, 1 - (direction_variance / max_direction_variance))
    
    # Overall confidence
    overall_confidence = (speed_confidence + direction_confidence) / 2
    
    return overall_confidence

def combine_predictions(base_prediction, forecast_data, forecast_weight, hours_ahead):
    """
    Combine base prediction with forecast data using weighted average.
    
    Args:
        base_prediction: Base LSTM prediction
        forecast_data: Forecast data
        forecast_weight: Weight for forecast data
        hours_ahead: Hours ahead for prediction
        
    Returns:
        dict: Combined prediction
    """
    
    # Get forecast values for the target time
    forecast_wind_speed = forecast_data['wind_speed_forecast'][hours_ahead - 1]
    forecast_wind_direction = forecast_data['wind_direction_forecast'][hours_ahead - 1]
    
    # Get base prediction values
    base_wind_speed = base_prediction['wind_speed_mph']
    base_wind_direction = base_prediction['wind_direction_degrees']
    
    # Calculate weighted averages
    weighted_wind_speed = (1 - forecast_weight) * base_wind_speed + forecast_weight * forecast_wind_speed
    
    # For wind direction, handle circular nature
    base_dir_rad = np.radians(base_wind_direction)
    forecast_dir_rad = np.radians(forecast_wind_direction)
    
    # Convert to Cartesian coordinates
    base_x = np.cos(base_dir_rad)
    base_y = np.sin(base_dir_rad)
    forecast_x = np.cos(forecast_dir_rad)
    forecast_y = np.sin(forecast_dir_rad)
    
    # Weighted average in Cartesian space
    weighted_x = (1 - forecast_weight) * base_x + forecast_weight * forecast_x
    weighted_y = (1 - forecast_weight) * base_y + forecast_weight * forecast_y
    
    # Convert back to degrees
    weighted_wind_direction = np.degrees(np.arctan2(weighted_y, weighted_x))
    
    # Ensure direction is between 0 and 360
    if weighted_wind_direction < 0:
        weighted_wind_direction += 360
    
    return {
        'wind_speed_mph': round(weighted_wind_speed, 2),
        'wind_direction_degrees': round(weighted_wind_direction, 1),
        'base_weight': 1 - forecast_weight,
        'forecast_weight': forecast_weight
    }

def adaptive_forecast_weighting(
    model_path=None,
    latitude=30.452,
    longitude=-91.188,
    hours_ahead=6
):
    # Set default model path based on how script is run
    if model_path is None:
        if os.path.exists('models/wind_lstm_model.pth'):
            model_path = 'models/wind_lstm_model.pth'
        else:
            model_path = '../models/wind_lstm_model.pth'
    """
    Use adaptive weighting based on forecast confidence and model performance.
    
    Args:
        model_path: Path to trained model
        latitude: Latitude for forecast data
        longitude: Longitude for forecast data
        hours_ahead: Hours ahead for prediction
        
    Returns:
        dict: Adaptive prediction results
    """
    
    print("=== Adaptive Forecast Weighting ===")
    
    # Test different forecast weights
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    for weight in weights:
        print(f"\nTesting forecast weight: {weight}")
        result = forecast_weighted_prediction(
            model_path=model_path,
            latitude=latitude,
            longitude=longitude,
            hours_ahead=hours_ahead,
            forecast_weight=weight
        )
        
        if result and 'improvement' in result and result['improvement'] is not None:
            results.append({
                'weight': weight,
                'improvement': result['improvement'],
                'accuracy': result['validation']['validation_metrics']['overall_accuracy']
            })
    
    # Find best weight
    if results:
        best_result = max(results, key=lambda x: x['improvement'])
        print(f"\n🎯 Best forecast weight: {best_result['weight']}")
        print(f"Best improvement: {best_result['improvement']:+.1f}%")
        print(f"Best accuracy: {best_result['accuracy']:.1f}%")
        
        # Make final prediction with best weight
        final_result = forecast_weighted_prediction(
            model_path=model_path,
            latitude=latitude,
            longitude=longitude,
            hours_ahead=hours_ahead,
            forecast_weight=best_result['weight']
        )
        
        return final_result
    
    return None

def main():
    """Main function to demonstrate forecast-weighted prediction."""
    print("=== PlumeTrackAI Forecast-Weighted Prediction ===")
    
    # Option 1: Fixed weight prediction
    print("\n1️⃣ Fixed Weight Prediction (30% forecast weight)")
    result1 = forecast_weighted_prediction(forecast_weight=0.3)
    
    # Option 2: Adaptive weight prediction
    print("\n2️⃣ Adaptive Weight Prediction")
    result2 = adaptive_forecast_weighting()
    
    if result1:
        print("\n✅ Fixed weight prediction completed!")
    
    if result2:
        print("\n✅ Adaptive weight prediction completed!")

if __name__ == "__main__":
    main() 