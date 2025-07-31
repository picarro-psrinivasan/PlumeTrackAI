#!/usr/bin/env python3
"""
PlumeTrackAI - Wind Prediction
Main script to make wind predictions using the trained LSTM model.
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

from predict_wind import main as predict_main

if __name__ == "__main__":
    print("=== PlumeTrackAI Wind Prediction ===")
    print("Starting prediction from main directory...")
    
    # Change to src directory for relative paths to work
    original_dir = os.getcwd()
    os.chdir('src')
    
    try:
        predict_main()
    finally:
        # Change back to original directory
        os.chdir(original_dir)
    
    print("Prediction completed!") 