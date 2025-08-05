#!/usr/bin/env python3
"""
Test script for Druid integration.
This script tests the Druid connector and wind data fetching functionality.
"""

import sys
import os

# Add paths - fix the import issues
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'data_handling'))
sys.path.insert(0, os.path.join(project_root, 'prediction'))
sys.path.insert(0, os.path.join(project_root, 'utils'))

def test_druid_connection():
    """Test Druid connection and data fetching."""
    print("=== Testing Druid Integration ===")
    
    try:
        print("🔍 Importing DruidConnector...")
        from data_handling.druid_connector import DruidConnector
        print("✅ Successfully imported DruidConnector")
        
        # Test with default settings
        print("🔍 Creating Druid connector...")
        connector = DruidConnector(datasource="15_minutes_avg_data")
        print("✅ Successfully created Druid connector")
        
        print("🔍 Testing Druid connection...")
        if connector.test_connection():
            print("✅ Successfully connected to Druid")
            
            # Test data fetching
            print("🔍 Fetching recent wind data...")
            df = connector.get_latest_wind_data(hours_back=6)
            
            if df is not None:
                print(f"✅ Successfully fetched {len(df)} records")
                print(f"Data shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print(f"Sample data:")
                print(df.head())
                return True
            else:
                print("❌ Failed to fetch data from Druid")
                return False
        else:
            print("❌ Failed to connect to Druid")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"Current sys.path: {sys.path}")
        return False
    except Exception as e:
        print(f"❌ Error testing Druid integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wind_predictor_integration():
    """Test the updated wind predictor with Druid integration."""
    print("\n=== Testing Wind Predictor Integration ===")
    
    try:
        print("🔍 Importing wind_predictor...")
        from prediction.wind_predictor import get_recent_wind_data
        print("✅ Successfully imported wind_predictor")
        
        # Test with Druid as primary source
        print("🔍 Testing wind predictor with Druid data source...")
        recent_data = get_recent_wind_data(
            data_source='druid',
            druid_url='http://sc-fenceline-int-dev5.corp.picarro.com:8888',
            datasource='15_minutes_avg_data',
            monitoring_system_id='398ae5cb-7971-44b2-b153-c2898ab6fde8',
            hours_back=6
        )
        
        if recent_data is not None:
            print(f"✅ Successfully fetched data through wind predictor")
            print(f"Data shape: {recent_data.shape}")
            print(f"Columns: {list(recent_data.columns)}")
            return True
        else:
            print("❌ Failed to fetch data through wind predictor")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"Current sys.path: {sys.path}")
        return False
    except Exception as e:
        print(f"❌ Error testing wind predictor integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_to_csv():
    """Test fallback to CSV when Druid is unavailable."""
    print("\n=== Testing CSV Fallback ===")
    
    try:
        print("🔍 Importing wind_predictor for CSV test...")
        from prediction.wind_predictor import get_recent_wind_data
        print("✅ Successfully imported wind_predictor")
        
        # Test with CSV as fallback
        print("🔍 Testing CSV fallback...")
        recent_data = get_recent_wind_data(
            data_source='csv',
            data_file='../data/15_min_avg_1site_1ms.csv',
            hours_back=6
        )
        
        if recent_data is not None:
            print(f"✅ Successfully loaded data from CSV")
            print(f"Data shape: {recent_data.shape}")
            print(f"Columns: {list(recent_data.columns)}")
            return True
        else:
            print("❌ Failed to load data from CSV")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"Current sys.path: {sys.path}")
        return False
    except Exception as e:
        print(f"❌ Error testing CSV fallback: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Druid Integration Tests")
    
    # Debug: Show current working directory and Python path
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Test 1: Direct Druid connection
    druid_success = test_druid_connection()
    
    # Test 2: Wind predictor integration
    predictor_success = test_wind_predictor_integration()
    
    # Test 3: CSV fallback
    csv_success = test_fallback_to_csv()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Druid Connection: {'✅ PASS' if druid_success else '❌ FAIL'}")
    print(f"Wind Predictor Integration: {'✅ PASS' if predictor_success else '❌ FAIL'}")
    print(f"CSV Fallback: {'✅ PASS' if csv_success else '❌ FAIL'}")
    
    if druid_success and predictor_success:
        print("\n🎉 All critical tests passed! Druid integration is working.")
    elif csv_success:
        print("\n⚠️ Druid integration failed, but CSV fallback is working.")
    else:
        print("\n❌ Multiple tests failed. Please check your configuration.")

if __name__ == "__main__":
    main() 