#!/usr/bin/env python3
"""
PlumeTrackAI - Model Training Script
Trains the LSTM model for wind prediction.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_definitions.lstm_model import main as train_main

if __name__ == "__main__":
    print("=== PlumeTrackAI Model Training ===")
    print("Starting training...")
    
    try:
        train_main()
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
    
    print("Training completed!") 