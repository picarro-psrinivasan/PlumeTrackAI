#!/usr/bin/env python3
"""
Geographic utilities for PlumeTrackAI.
Contains functions for distance and bearing calculations.
"""

import math
from typing import Tuple


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate bearing from source (lat1, lon1) to destination (lat2, lon2).
    
    Args:
        lat1, lon1: Source coordinates (degrees)
        lat2, lon2: Destination coordinates (degrees)
    
    Returns:
        Bearing in degrees (0-360)
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Calculate bearing
    d_lon = lon2_rad - lon1_rad
    y = math.sin(d_lon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(d_lon)
    
    bearing = math.degrees(math.atan2(y, x))
    
    # Convert to 0-360 range
    bearing = (bearing + 360) % 360
    
    return bearing


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: Source coordinates (degrees)
        lat2, lon2: Destination coordinates (degrees)
    
    Returns:
        Distance in kilometers
    """
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    d_lat = lat2_rad - lat1_rad
    d_lon = lon2_rad - lon1_rad
    
    a = (math.sin(d_lat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    distance = R * c
    
    return distance


def calculate_effective_wind_speed(wind_speed: float, wind_direction: float, bearing: float) -> float:
    """
    Calculate effective wind speed in the direction of travel.
    
    Args:
        wind_speed: Wind speed in km/h
        wind_direction: Wind direction in degrees
        bearing: Direction to destination in degrees
    
    Returns:
        Effective wind speed in km/h (can be negative if wind opposes travel)
    """
    # Calculate angle difference
    angle_diff = abs(wind_direction - bearing)
    
    # Ensure angle difference is within 0-180 degrees
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    # Calculate effective wind speed
    effective_speed = wind_speed * math.cos(math.radians(angle_diff))
    
    return effective_speed 