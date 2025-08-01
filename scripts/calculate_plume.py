#!/usr/bin/env python3
"""
PlumeTrackAI - Gas Plume Travel Calculator
Calculates gas plume travel time to risk destinations.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prediction.plume_calculator import main as plume_main

if __name__ == "__main__":
    print("PlumeTrackAI Gas Travel Calculator")
    print("Starting plume travel calculation...")
    
    try:
        plume_main()
    except Exception as e:
        print(f"Error during plume travel calculation: {e}")
        sys.exit(1)
    
    print("Plume travel calculation completed!") 