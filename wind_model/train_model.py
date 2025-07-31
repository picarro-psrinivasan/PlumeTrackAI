#!/usr/bin/env python3
"""
PlumeTrackAI - Wind Prediction Model Training
Main script to train the LSTM model for wind speed and direction prediction.
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

from src.lstm_model import main as train_main

if __name__ == "__main__":
    print("=== PlumeTrackAI Model Training ===")
    print("Starting training from wind_model directory...")
    
    try:
        train_main()
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
    
    print("Training completed!") 