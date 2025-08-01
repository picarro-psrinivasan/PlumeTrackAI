#!/usr/bin/env python3
"""
Test script for forecast weighted prediction
"""

import sys
import os

# Add src to path
sys.path.append('src')
sys.path.append('api')

from src.forecast_weighted_prediction import forecast_weighted_prediction, adaptive_forecast_weighting

def main():
    """Test the forecast weighted prediction system."""
    print("=== Testing Forecast Weighted Prediction ===")
    
    # Test fixed weight prediction
    print("\n1️⃣ Testing Fixed Weight Prediction (30% forecast weight)")
    result1 = forecast_weighted_prediction(forecast_weight=0.3)
    
    if result1:
        print("✅ Fixed weight prediction completed!")
    else:
        print("❌ Fixed weight prediction failed!")
    
    # Test adaptive weight prediction
    print("\n2️⃣ Testing Adaptive Weight Prediction")
    result2 = adaptive_forecast_weighting()
    
    if result2:
        print("✅ Adaptive weight prediction completed!")
    else:
        print("❌ Adaptive weight prediction failed!")

if __name__ == "__main__":
    main() 