#!/usr/bin/env python3
"""
PlumeTrackAI - Wind Prediction Model Training
Main script to train the LSTM model for wind speed and direction prediction.
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

from lstm_model import main as train_main

if __name__ == "__main__":
    print("=== PlumeTrackAI Model Training ===")
    print("Starting training from main directory...")
    
    # Change to src directory for relative paths to work
    original_dir = os.getcwd()
    os.chdir('src')
    
    try:
        train_main()
    finally:
        # Change back to original directory
        os.chdir(original_dir)
    
    print("Training completed!") 