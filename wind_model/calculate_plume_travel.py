#!/usr/bin/env python3
"""
PlumeTrackAI - Gas Plume Travel Time Calculator
Main script to calculate gas plume travel time to risk destinations.
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

from src.plume_travel import main as plume_main

if __name__ == "__main__":
    print("=== PlumeTrackAI Gas Travel Calculator ===")
    print("Starting plume travel calculation from wind_model directory...")
    
    try:
        plume_main()
    except Exception as e:
        print(f"Error during plume travel calculation: {e}")
        sys.exit(1)
    
    print("Plume travel calculation completed!") 