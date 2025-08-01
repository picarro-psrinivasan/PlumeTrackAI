#!/usr/bin/env python3
"""
PlumeTrackAI - Gas Plume Travel Time Calculator
Calculates time for gas plume to reach risk destinations using LSTM wind predictions.
"""

import numpy as np
import math
import torch
from typing import Tuple, List, Optional
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prediction.wind_predictor import load_trained_model, get_recent_wind_data, prepare_input_sequence
from utils.geo_utils import calculate_bearing, calculate_distance, calculate_effective_wind_speed


def time_stepped_plume_travel(
    source_lat: float, 
    source_lon: float, 
    risk_lat: float, 
    risk_lon: float,
    wind_predictions: List[Tuple[float, float]],
    time_step_hours: float = 1.0
) -> Tuple[Optional[float], List[dict]]:
    """
    Calculate time for gas plume to reach risk destination using time-stepped algorithm.
    
    Args:
        source_lat, source_lon: Source coordinates
        risk_lat, risk_lon: Risk destination coordinates
        wind_predictions: List of (wind_speed_kmh, wind_direction_deg) tuples
        time_step_hours: Time step in hours (default: 1 hour)
    
    Returns:
        Tuple of (arrival_time_hours, travel_log)
        - arrival_time_hours: Time to reach destination (None if not reached)
        - travel_log: List of step-by-step travel details
    """
    # Calculate initial distance and bearing
    total_distance = calculate_distance(source_lat, source_lon, risk_lat, risk_lon)
    bearing = calculate_bearing(source_lat, source_lon, risk_lat, risk_lon)
    
    print(f"Initial distance to risk zone: {total_distance:.2f} km")
    print(f"Bearing to risk zone: {bearing:.1f}°")
    print(f"Number of wind predictions: {len(wind_predictions)}")
    
    # Initialize variables
    remaining_distance = total_distance
    current_time = 0.0
    travel_log = []
    
    for hour, (wind_speed, wind_direction) in enumerate(wind_predictions):
        # Calculate effective wind speed
        effective_speed = calculate_effective_wind_speed(wind_speed, wind_direction, bearing)
        
        # Calculate distance moved this step
        if effective_speed > 0:
            distance_moved = effective_speed * time_step_hours
            remaining_distance -= distance_moved
            movement_status = "advancing"
        else:
            distance_moved = 0
            movement_status = "no progress (wind opposes)"
        
        # Log this step
        step_info = {
            'hour': hour + 1,
            'time': current_time,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'effective_speed': effective_speed,
            'distance_moved': distance_moved,
            'remaining_distance': max(0, remaining_distance),
            'movement_status': movement_status
        }
        travel_log.append(step_info)
        
        print(f"Hour {hour + 1}: Wind {wind_speed:.1f} km/h @ {wind_direction:.1f}° → "
              f"Effective: {effective_speed:.1f} km/h → "
              f"Remaining: {max(0, remaining_distance):.2f} km")
        
        # Check if destination reached
        if remaining_distance <= 0:
            arrival_time = current_time + time_step_hours
            print(f"Gas plume reached risk zone at {arrival_time:.1f} hours!")
            return arrival_time, travel_log
        
        current_time += time_step_hours
    
    print(f" Gas plume did not reach risk zone within {len(wind_predictions)} hours")
    print(f"   Remaining distance: {remaining_distance:.2f} km")
    return None, travel_log


def predict_plume_travel_time(
    source_lat: float,
    source_lon: float,
    risk_lat: float,
    risk_lon: float,
    model_path: str = 'trained_models/wind_lstm_model.pth',
    data_file: str = 'data/15_min_avg_1site_1ms.csv',
    prediction_hours: int = 6
) -> Tuple[Optional[float], List[dict]]:
    """
    Predict gas plume travel time using trained LSTM model.
    
    Args:
        source_lat, source_lon: Source coordinates
        risk_lat, risk_lon: Risk destination coordinates
        model_path: Path to trained LSTM model
        data_file: Path to wind data file
        prediction_hours: Number of hours to predict ahead
    
    Returns:
        Tuple of (arrival_time_hours, travel_log)
    """
    print("=== PlumeTrackAI Gas Travel Time Prediction ===")
    
    # Load trained model
    model, scaler = load_trained_model(model_path)
    if model is None:
        print("Error: Could not load trained model!")
        return None, []
    
    # Get recent wind data for prediction
    recent_data = get_recent_wind_data(data_file, hours_back=6)
    if recent_data is None or len(recent_data) == 0:
        print("Error: Could not load recent wind data!")
        return None, []
    
    # Generate wind predictions for next 6 hours using multi-step model
    wind_predictions = []
    
    # Prepare input sequence for prediction
    input_sequence = prepare_input_sequence(recent_data, scaler, sequence_length=24)
    
    if input_sequence is None:
        print("Error: Could not prepare input sequence")
        return None, []
    
    # Make prediction for all 6 hours at once
    model.eval()
    with torch.no_grad():
        # Make prediction (input_sequence is already a tensor)
        prediction = model(input_sequence)  # Shape: (1, 6, 3)
        
        # Process each hour's prediction
        for hour in range(prediction_hours):
            # Get prediction for this hour
            hour_pred = prediction[0, hour, :]  # Shape: (3,)
            
            # Convert prediction back to original scale
            wind_speed_pred = scaler.inverse_transform(hour_pred[0].reshape(1, -1).numpy())[0, 0]
            
            # Convert sin/cos back to degrees
            wind_dir_sin = hour_pred[1].item()
            wind_dir_cos = hour_pred[2].item()
            wind_direction_pred = math.degrees(math.atan2(wind_dir_sin, wind_dir_cos))
            wind_direction_pred = (wind_direction_pred + 360) % 360
            
            # Convert wind speed from mph to km/h
            wind_speed_kmh = wind_speed_pred * 1.60934
            
            wind_predictions.append((wind_speed_kmh, wind_direction_pred))
    
    print(f"\nWind predictions for next {prediction_hours} hours:")
    for hour, (speed, direction) in enumerate(wind_predictions):
        print(f"Hour {hour + 1}: {speed:.1f} km/h @ {direction:.1f}°")
    
    # Calculate travel time
    arrival_time, travel_log = time_stepped_plume_travel(
        source_lat, source_lon, risk_lat, risk_lon, wind_predictions
    )
    
    return arrival_time, travel_log


def main():
    """
    Example usage of plume travel time calculation.
    """
    # Example coordinates (you can modify these)
    source_lat, source_lon = 40.7128, -74.0060  # New York City
    risk_lat, risk_lon = 40.7589, -73.9851      # Times Square (example risk zone)
    
    print("=== PlumeTrackAI Gas Travel Time Calculator ===")
    print(f"Source: ({source_lat}, {source_lon})")
    print(f"Risk Zone: ({risk_lat}, {risk_lon})")
    
    # Calculate travel time
    arrival_time, travel_log = predict_plume_travel_time(
        source_lat, source_lon, risk_lat, risk_lon
    )
    
    if arrival_time is not None:
        print(f"\nGas plume will reach risk zone in {arrival_time:.1f} hours")
    else:
        print(f"\nGas plume will not reach risk zone within prediction window")
    
    # Print detailed travel log
    print(f"\nDetailed Travel Log:")
    print("-" * 80)
    for step in travel_log:
        print(f"Hour {step['hour']:2d}: "
              f"Wind {step['wind_speed']:5.1f} km/h @ {step['wind_direction']:5.1f}° → "
              f"Effective: {step['effective_speed']:5.1f} km/h → "
              f"Remaining: {step['remaining_distance']:6.2f} km "
              f"({step['movement_status']})")


if __name__ == "__main__":
    import torch
    import numpy as np
    main() 