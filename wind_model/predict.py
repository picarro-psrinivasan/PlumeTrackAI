#!/usr/bin/env python3
"""
PlumeTrackAI - Wind Prediction
Main script to make wind predictions using the trained LSTM model.
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

from src.predict_wind import main as predict_main

if __name__ == "__main__":
    print("=== PlumeTrackAI Wind Prediction ===")
    print("Starting prediction from wind_model directory...")
    
    try:
        predict_main()
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)
    
    print("Prediction completed!") 