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
    from prediction.plume_calculator import time_stepped_plume_travel
    from utils.geo_utils import calculate_bearing, calculate_distance, calculate_effective_wind_speed
except ImportError:
    # If running directly, adjust paths
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from prediction.wind_predictor import load_trained_model, get_recent_wind_data, prepare_input_sequence, predict_wind_6hours_ahead
    from api.ops_wind_data_api import extract_forecast_wind_data, validate_prediction_with_forecast
    from prediction.plume_calculator import time_stepped_plume_travel
    from utils.geo_utils import calculate_bearing, calculate_distance, calculate_effective_wind_speed

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

def calculate_plume_forecast_weighted(
    source_lat: float,
    source_lon: float,
    risk_lat: float,
    risk_lon: float,
    model_path=None,
    latitude=30.452,
    longitude=-91.188,
    hours_ahead=6,
    forecast_weight=0.3,
    confidence_threshold=0.8
):
    """
    Calculate plume travel time using forecast-weighted wind predictions.
    
    Args:
        source_lat, source_lon: Source coordinates
        risk_lat, risk_lon: Risk destination coordinates
        model_path: Path to trained model
        latitude, longitude: Location for forecast data
        hours_ahead: Hours ahead for prediction
        forecast_weight: Weight given to forecast data (0-1)
        confidence_threshold: Minimum confidence for forecast weighting
        
    Returns:
        dict: Plume travel results with forecast-weighted predictions
    """
    
    print("=== Forecast-Weighted Plume Travel Calculation ===")
    
    # Set default model path based on how script is run
    if model_path is None:
        if os.path.exists('trained_models/wind_lstm_model.pth'):
            model_path = 'trained_models/wind_lstm_model.pth'
        else:
            model_path = '../trained_models/wind_lstm_model.pth'
    
    # Get forecast-weighted wind predictions
    wind_results = forecast_weighted_prediction(
        model_path=model_path,
        latitude=latitude,
        longitude=longitude,
        hours_ahead=hours_ahead,
        forecast_weight=forecast_weight,
        confidence_threshold=confidence_threshold
    )
    
    if wind_results is None:
        print("❌ Could not get wind predictions")
        return None
    
    # Extract wind predictions for plume calculation
    base_predictions = wind_results.get('base_prediction', {})
    weighted_predictions = wind_results.get('weighted_prediction', {})
    
    # Prepare both base and weighted predictions for plume calculation
    base_wind_predictions = []
    weighted_wind_predictions = []
    
    # Get base prediction
    if 'wind_speed_mph' in base_predictions and 'wind_direction_degrees' in base_predictions:
        base_wind_speed_kmh = base_predictions['wind_speed_mph'] * 1.60934
        base_wind_direction = base_predictions['wind_direction_degrees']
        base_wind_predictions.append((base_wind_speed_kmh, base_wind_direction))
    else:
        print("❌ No valid base wind predictions available for plume calculation")
        return None
    
    # Get weighted prediction
    if 'wind_speed_mph' in weighted_predictions and 'wind_direction_degrees' in weighted_predictions:
        weighted_wind_speed_kmh = weighted_predictions['wind_speed_mph'] * 1.60934
        weighted_wind_direction = weighted_predictions['wind_direction_degrees']
        weighted_wind_predictions.append((weighted_wind_speed_kmh, weighted_wind_direction))
    else:
        print("❌ No valid weighted wind predictions available for plume calculation")
        return None
    
    # For multiple hours, we need to generate predictions for each hour
    # This is a simplified version - in practice, you'd want to get predictions for all hours
    if hours_ahead > 1:
        # For now, we'll use the same prediction for all hours
        # In a full implementation, you'd get predictions for each hour
        for hour in range(1, hours_ahead):
            base_wind_predictions.append(base_wind_predictions[0])
            weighted_wind_predictions.append(weighted_wind_predictions[0])
    
    print(f"Base wind predictions for plume calculation:")
    for hour, (speed, direction) in enumerate(base_wind_predictions):
        print(f"Hour {hour + 1}: {speed:.1f} km/h @ {direction:.1f}°")
    
    print(f"Weighted wind predictions for plume calculation:")
    for hour, (speed, direction) in enumerate(weighted_wind_predictions):
        print(f"Hour {hour + 1}: {speed:.1f} km/h @ {direction:.1f}°")
    
    # Calculate plume travel time for both base and weighted predictions
    base_arrival_time, base_travel_log = time_stepped_plume_travel(
        source_lat, source_lon, risk_lat, risk_lon, base_wind_predictions
    )
    
    weighted_arrival_time, weighted_travel_log = time_stepped_plume_travel(
        source_lat, source_lon, risk_lat, risk_lon, weighted_wind_predictions
    )
    
    # Combine base and weighted predictions into a single travel log
    combined_travel_log = []
    for hour in range(len(base_travel_log)):
        base_step = base_travel_log[hour]
        weighted_step = weighted_travel_log[hour]
        
        combined_step = {
            'hour': hour + 1,
            'time': base_step['time'],
            'base_prediction': {
                'wind_speed': base_step['wind_speed'],
                'wind_direction': base_step['wind_direction'],
                'effective_speed': base_step['effective_speed'],
                'distance_moved': base_step['distance_moved'],
                'remaining_distance': base_step['remaining_distance'],
                'movement_status': base_step['movement_status']
            },
            'weighted_prediction': {
                'wind_speed': weighted_step['wind_speed'],
                'wind_direction': weighted_step['wind_direction'],
                'effective_speed': weighted_step['effective_speed'],
                'distance_moved': weighted_step['distance_moved'],
                'remaining_distance': weighted_step['remaining_distance'],
                'movement_status': weighted_step['movement_status']
            }
        }
        combined_travel_log.append(combined_step)
    
    # Build comprehensive results
    results = {
        'source_location': {
            'latitude': source_lat,
            'longitude': source_lon
        },
        'risk_location': {
            'latitude': risk_lat,
            'longitude': risk_lon
        },
        'wind_predictions': wind_results,
        'plume_travel': {
            'arrival_time_hours': weighted_arrival_time,  # Use weighted as primary
            'travel_log': combined_travel_log,
            'will_reach_destination': weighted_arrival_time is not None,
            'base_arrival_time_hours': base_arrival_time,
            'weighted_arrival_time_hours': weighted_arrival_time,
            'base_will_reach_destination': base_arrival_time is not None,
            'weighted_will_reach_destination': weighted_arrival_time is not None
        },
        'forecast_weight_used': wind_results.get('forecast_weight_used', 0.0),
        'forecast_confidence': wind_results.get('forecast_confidence', 0.0),
        'improvement': wind_results.get('improvement', 0.0)
    }
    
    # Add distance and bearing information
    total_distance = calculate_distance(source_lat, source_lon, risk_lat, risk_lon)
    bearing = calculate_bearing(source_lat, source_lon, risk_lat, risk_lon)
    
    results['plume_travel']['total_distance_km'] = total_distance
    results['plume_travel']['bearing_degrees'] = bearing
    
    print(f"\n=== Plume Travel Results ===")
    print(f"Total distance: {total_distance:.2f} km")
    print(f"Bearing: {bearing:.1f}°")
    
    print(f"\n=== Base Prediction Results ===")
    if base_arrival_time is not None:
        print(f"Base prediction: Gas plume will reach risk zone in {base_arrival_time:.1f} hours")
    else:
        print(f"Base prediction: Gas plume will not reach risk zone within {hours_ahead} hours")
    
    print(f"\n=== Weighted Prediction Results ===")
    if weighted_arrival_time is not None:
        print(f"Weighted prediction: Gas plume will reach risk zone in {weighted_arrival_time:.1f} hours")
    else:
        print(f"Weighted prediction: Gas plume will not reach risk zone within {hours_ahead} hours")
    
    return results

def main():
    """Main function to demonstrate forecast-weighted prediction."""
    print("=== PlumeTrackAI Forecast-Weighted Prediction ===")
    
    # Option 1: Fixed weight prediction
    print("\n1️⃣ Fixed Weight Prediction (30% forecast weight)")
    result1 = forecast_weighted_prediction(forecast_weight=0.3)
    
    # Option 2: Adaptive weight prediction
    print("\n2️⃣ Adaptive Weight Prediction")
    result2 = adaptive_forecast_weighting()
    
    # Option 3: Forecast-weighted plume calculation
    print("\n3️⃣ Forecast-Weighted Plume Calculation")
    # Example coordinates (you can modify these)
    source_lat, source_lon = 30.452, -91.188  # Example source
    risk_lat, risk_lon = 30.458, -91.182      # Example risk zone (slightly north and east)
    
    result3 = calculate_plume_forecast_weighted(
        source_lat=source_lat,
        source_lon=source_lon,
        risk_lat=risk_lat,
        risk_lon=risk_lon,
        forecast_weight=0.3
    )
    
    if result1:
        print("\n✅ Fixed weight prediction completed!")
    
    if result2:
        print("\n✅ Adaptive weight prediction completed!")
    
    if result3:
        print("\n✅ Forecast-weighted plume calculation completed!")

if __name__ == "__main__":
    main() 