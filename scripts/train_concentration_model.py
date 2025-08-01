#!/usr/bin/env python3
"""
PlumeTrackAI - Concentration Model Training Script
Trains the LSTM model for concentration prediction.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_definitions.concentration_lstm_model import main as train_main

if __name__ == "__main__":
    print("=== PlumeTrackAI Concentration Model Training ===")
    print("Starting concentration model training...")
    
    try:
        train_main()
    except Exception as e:
        print(f"Error during concentration model training: {e}")
        sys.exit(1)
    
    print("Concentration model training completed!") 