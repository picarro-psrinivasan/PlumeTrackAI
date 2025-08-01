#!/usr/bin/env python3
"""
PlumeTrackAI - Wind Prediction Script
Makes wind predictions using the trained LSTM model.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prediction.wind_predictor import main as predict_main

if __name__ == "__main__":
    print("=== PlumeTrackAI Wind Prediction ===")
    print("Starting wind prediction...")
    
    try:
        predict_main()
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)
    
    print("Prediction completed!") 