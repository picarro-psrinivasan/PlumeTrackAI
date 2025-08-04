#!/usr/bin/env python3
"""
Geographic utilities for PlumeTrackAI.
Contains functions for distance and bearing calculations.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)
        
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth's radius in kilometers
    earth_radius = 6371.0
    
    return earth_radius * c

def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the bearing from point 1 to point 2.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)
        
    Returns:
        Bearing in degrees (0-360)
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Calculate bearing
    dlon = lon2_rad - lon1_rad
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    bearing = math.degrees(math.atan2(y, x))
    
    # Normalize to 0-360
    bearing = (bearing + 360) % 360
    
    return bearing

def calculate_effective_wind_speed(wind_speed: float, wind_direction: float, bearing: float) -> float:
    """
    Calculate the effective wind speed in the direction of the bearing.
    
    Args:
        wind_speed: Wind speed in km/h
        wind_direction: Wind direction in degrees
        bearing: Direction to destination in degrees
        
    Returns:
        Effective wind speed in km/h (positive = favorable, negative = opposing)
    """
    # Calculate the angle between wind direction and bearing
    angle_diff = wind_direction - bearing
    
    # Normalize angle to -180 to 180 degrees
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360
    
    # Convert to radians
    angle_rad = math.radians(angle_diff)
    
    # Calculate effective wind speed
    effective_speed = wind_speed * math.cos(angle_rad)
    
    return effective_speed

def wind_predictions_to_geojson(
    source_lat: float,
    source_lon: float,
    wind_speeds: List[float],
    wind_directions: List[float],
    time_step_hours: float = 1.0,
    max_distance_km: float = 100.0
) -> Dict[str, Any]:
    """
    Convert wind predictions into a GeoJSON LineString showing the plume path.
    
    Args:
        source_lat, source_lon: Source coordinates
        wind_speeds: List of wind speeds in km/h for each hour
        wind_directions: List of wind directions in degrees for each hour
        time_step_hours: Time step in hours (default: 1 hour)
        max_distance_km: Maximum distance to plot (default: 100 km)
        
    Returns:
        GeoJSON LineString showing the plume path
    """
    
    # Initialize plume position
    current_lat = source_lat
    current_lon = source_lon
    
    # Create coordinates list for GeoJSON
    coordinates = [[source_lon, source_lat]]  # GeoJSON uses [lon, lat] order
    
    # Track plume movement for each hour
    for hour, (wind_speed, wind_direction) in enumerate(zip(wind_speeds, wind_directions)):
        # Calculate distance moved in this time step
        distance_moved_km = wind_speed * time_step_hours
        
        # Convert wind direction to radians
        wind_direction_rad = math.radians(wind_direction)
        
        # Calculate new position
        # Earth's radius in kilometers
        earth_radius = 6371.0
        
        # Calculate lat/lon change
        lat_change = (distance_moved_km / earth_radius) * math.cos(wind_direction_rad)
        lon_change = (distance_moved_km / earth_radius) * math.sin(wind_direction_rad) / math.cos(math.radians(current_lat))
        
        # Update position
        current_lat += math.degrees(lat_change)
        current_lon += math.degrees(lon_change)
        
        # Add to coordinates
        coordinates.append([current_lon, current_lat])
        
        # Check if we've exceeded max distance
        distance_from_source = calculate_distance(source_lat, source_lon, current_lat, current_lon)
        if distance_from_source > max_distance_km:
            break
    
    # Create GeoJSON LineString
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates
        },
        "properties": {
            "source": {
                "latitude": source_lat,
                "longitude": source_lon
            },
            "total_hours": len(wind_speeds),
            "max_distance_km": max_distance_km,
            "wind_predictions": [
                {
                    "hour": i + 1,
                    "wind_speed_kmh": wind_speeds[i],
                    "wind_direction_degrees": wind_directions[i]
                }
                for i in range(len(wind_speeds))
            ]
        }
    }
    
    return geojson

def create_plume_geojson_from_travel_log(
    source_lat: float,
    source_lon: float,
    travel_log: List[Dict[str, Any]],
    prediction_type: str = "weighted_prediction"
) -> Dict[str, Any]:
    """
    Create GeoJSON from travel log data.
    
    Args:
        source_lat, source_lon: Source coordinates
        travel_log: Travel log from plume calculation
        prediction_type: "base_prediction" or "weighted_prediction"
        
    Returns:
        GeoJSON LineString showing the plume path
    """
    
    # Extract wind speeds and directions from travel log
    wind_speeds = []
    wind_directions = []
    
    for step in travel_log:
        prediction = step.get(prediction_type, {})
        wind_speed = prediction.get('wind_speed', 0)
        wind_direction = prediction.get('wind_direction', 0)
        
        wind_speeds.append(wind_speed)
        wind_directions.append(wind_direction)
    
    # Create GeoJSON
    geojson = wind_predictions_to_geojson(
        source_lat=source_lat,
        source_lon=source_lon,
        wind_speeds=wind_speeds,
        wind_directions=wind_directions
    )
    
    # Add prediction type to properties
    geojson["properties"]["prediction_type"] = prediction_type
    
    return geojson

def create_comparison_geojson(
    source_lat: float,
    source_lon: float,
    travel_log: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create a GeoJSON FeatureCollection comparing base and weighted predictions.
    
    Args:
        source_lat, source_lon: Source coordinates
        travel_log: Travel log with both base and weighted predictions
        
    Returns:
        GeoJSON FeatureCollection with both paths
    """
    
    # Create base prediction path
    base_geojson = create_plume_geojson_from_travel_log(
        source_lat, source_lon, travel_log, "base_prediction"
    )
    base_geojson["properties"]["name"] = "Base Prediction"
    base_geojson["properties"]["color"] = "#ff0000"  # Red
    
    # Create weighted prediction path
    weighted_geojson = create_plume_geojson_from_travel_log(
        source_lat, source_lon, travel_log, "weighted_prediction"
    )
    weighted_geojson["properties"]["name"] = "Weighted Prediction"
    weighted_geojson["properties"]["color"] = "#0000ff"  # Blue
    
    # Create FeatureCollection
    feature_collection = {
        "type": "FeatureCollection",
        "features": [base_geojson, weighted_geojson]
    }
    
    return feature_collection 