#!/usr/bin/env python3
"""
PlumeTrackAI - Concentration Prediction Script
Makes concentration predictions using the trained LSTM model.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prediction.concentration_predictor import main as predict_main

if __name__ == "__main__":
    print("=== PlumeTrackAI Concentration Prediction ===")
    print("Starting concentration prediction...")
    
    try:
        predict_main()
    except Exception as e:
        print(f"Error during concentration prediction: {e}")
        sys.exit(1)
    
    print("Concentration prediction completed!") 