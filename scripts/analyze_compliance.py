#!/usr/bin/env python3
"""
PlumeTrackAI - Compliance Impact Analysis Script
Analyzes predicted concentrations for compliance violations and regulatory impact.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prediction.compliance_analyzer import main as analyze_main

if __name__ == "__main__":
    print("=== PlumeTrackAI Compliance Impact Analysis ===")
    print("Starting compliance analysis...")
    
    try:
        analyze_main()
    except Exception as e:
        print(f"Error during compliance analysis: {e}")
        sys.exit(1)
    
    print("Compliance analysis completed!") 