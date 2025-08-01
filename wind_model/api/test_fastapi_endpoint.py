#!/usr/bin/env python3
"""
Test script for FastAPI Forecast-Weighted Prediction endpoint
Demonstrates how to use the API for making predictions.
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print("Make sure the server is running with: python fastapi_forecast_weighted.py")

def test_simple_prediction():
    """Test the simple prediction endpoint."""
    print("\n🎯 Testing simple prediction...")
    try:
        response = requests.get(f"{BASE_URL}/predict/simple", params={
            'latitude': 30.452,
            'longitude': -91.188,
            'hours_ahead': 6
        })
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Simple prediction successful")
            print(f"Base prediction: {result['base_prediction']['wind_speed_mph']:.2f} mph, {result['base_prediction']['wind_direction_degrees']:.1f}°")
            print(f"Weighted prediction: {result['weighted_prediction']['wind_speed_mph']:.2f} mph, {result['weighted_prediction']['wind_direction_degrees']:.1f}°")
            print(f"Forecast confidence: {result.get('forecast_confidence', 'N/A')}")
            print(f"Improvement: {result.get('improvement', 'N/A')}")
        else:
            print(f"❌ Simple prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")

def test_full_prediction():
    """Test the full prediction endpoint with custom parameters."""
    print("\n🎯 Testing full prediction with custom parameters...")
    
    # Prepare request data
    request_data = {
        "latitude": 30.452,
        "longitude": -91.188,
        "hours_ahead": 6,
        "forecast_weight": 0.4,
        "confidence_threshold": 0.7
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Full prediction successful")
            print(f"Location: {result['location']['latitude']}°N, {result['location']['longitude']}°E")
            print(f"Hours ahead: {result['hours_ahead']}")
            print(f"Base prediction: {result['base_prediction']['wind_speed_mph']:.2f} mph, {result['base_prediction']['wind_direction_degrees']:.1f}°")
            print(f"Weighted prediction: {result['weighted_prediction']['wind_speed_mph']:.2f} mph, {result['weighted_prediction']['wind_direction_degrees']:.1f}°")
            print(f"Forecast confidence: {result.get('forecast_confidence', 'N/A')}")
            print(f"Forecast weight used: {result.get('forecast_weight_used', 'N/A')}")
            print(f"Improvement: {result.get('improvement', 'N/A')}")
            
            # Show validation results if available
            if result.get('validation'):
                validation = result['validation']
                if 'validation_metrics' in validation:
                    metrics = validation['validation_metrics']
                    print(f"Overall accuracy: {metrics.get('overall_accuracy', 'N/A')}%")
        else:
            print(f"❌ Full prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")

def test_different_locations():
    """Test predictions for different locations."""
    print("\n🌍 Testing predictions for different locations...")
    
    locations = [
        {"name": "Baton Rouge, LA", "lat": 30.452, "lon": -91.188},
        {"name": "New Orleans, LA", "lat": 29.951, "lon": -90.071},
        {"name": "Houston, TX", "lat": 29.760, "lon": -95.369}
    ]
    
    for location in locations:
        print(f"\n📍 Testing {location['name']}...")
        try:
            response = requests.get(f"{BASE_URL}/predict/simple", params={
                'latitude': location['lat'],
                'longitude': location['lon'],
                'hours_ahead': 6
            })
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {location['name']}: {result['weighted_prediction']['wind_speed_mph']:.2f} mph, {result['weighted_prediction']['wind_direction_degrees']:.1f}°")
            else:
                print(f"❌ {location['name']}: Failed")
        except requests.exceptions.ConnectionError:
            print(f"❌ {location['name']}: Connection error")

def main():
    """Run all tests."""
    print("🧪 PlumeTrackAI FastAPI Endpoint Tests")
    print("=" * 50)
    
    # Test health check
    test_health_check()
    
    # Test simple prediction
    test_simple_prediction()
    
    # Test full prediction
    test_full_prediction()
    
    # Test different locations
    test_different_locations()
    
    print("\n" + "=" * 50)
    print("🏁 All tests completed!")

if __name__ == "__main__":
    main() 