#!/usr/bin/env python3
"""
Druid Database Connector for PlumeTrackAI
Provides functions to fetch wind data from Apache Druid database.
"""

import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DruidConnector:
    """
    Connector for Apache Druid database to fetch wind data.
    """
    
    def __init__(self, druid_url: str = "http://sc-fenceline-int-dev5.corp.picarro.com:8888", datasource: str = "15_minutes_avg_data", username: str = "druid", password: str = "FoolishPassword"):
        """
        Initialize Druid connector.
        
        Args:
            druid_url: Druid broker URL (default: sc-fenceline-int-dev5.corp.picarro.com:8888)
            datasource: Druid datasource name (default: 15_minutes_avg_data)
            username: Druid username (default: druid)
            password: Druid password (default: FoolishPassword)
        """
        self.druid_url = druid_url.rstrip('/')
        self.datasource = datasource
        self.username = username
        self.password = password
        self.query_url = f"{self.druid_url}/druid/v2/sql"
        
    def test_connection(self) -> bool:
        """
        Test connection to Druid database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Simple query to test connection - use the same query structure as the main query
            query = f"""
            SELECT *
            FROM "{self.datasource}"
            WHERE "monitoring_system_id" = '398ae5cb-7971-44b2-b153-c2898ab6fde8'
              AND "__time" >= CURRENT_TIMESTAMP - INTERVAL '1' HOUR
            ORDER BY "__time" DESC
            LIMIT 1
            """
            response = requests.post(
                self.query_url,
                headers={"Content-Type": "application/json"},
                json={"query": query},
                auth=(self.username, self.password),
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Successfully connected to Druid datasource: {self.datasource}")
                return True
            else:
                logger.error(f"Failed to connect to Druid: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing Druid connection: {e}")
            return False
    
    def get_wind_data_from_druid(
        self,
        hours_back: int = 6,
        monitoring_system_id: str = '398ae5cb-7971-44b2-b153-c2898ab6fde8',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch wind data from Druid database using the specific query structure.
        
        Args:
            hours_back: Number of hours of data to fetch
            monitoring_system_id: Monitoring system ID to filter by
            start_time: Start time for data range (optional)
            end_time: End time for data range (optional)
            
        Returns:
            pd.DataFrame: Wind data with columns ['timestamp', 'wind_speed', 'wind_direction_deg']
        """
        try:
            # Build the specific SQL query as requested - fetch extra hours to ensure we have enough data
            extra_hours = hours_back + 2  # Add 2 extra hours to ensure we have enough data
            query = f"""
            SELECT *
            FROM "{self.datasource}"
            WHERE "monitoring_system_id" = '{monitoring_system_id}'
              AND "__time" >= CURRENT_TIMESTAMP - INTERVAL '{extra_hours}' HOUR
            ORDER BY "__time" DESC
            """
            
            logger.info(f"Executing Druid query: {query}")
            
            logger.info(f"Executing Druid query: {query}")
            
            # Execute query
            response = requests.post(
                self.query_url,
                headers={"Content-Type": "application/json"},
                json={"query": query},
                auth=(self.username, self.password),
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Druid query failed: {response.status_code} - {response.text}")
                return None
            
            # Parse response
            result = response.json()
            
            if not result:
                logger.warning("No data returned from Druid query")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(result)
            
            # Debug: Print column names to see what we got
            logger.info(f"DataFrame columns: {list(df.columns)}")
            logger.info(f"DataFrame shape: {df.shape}")
            if not df.empty:
                logger.info(f"Sample data: {df.head()}")
            
            # Check if we have the expected columns
            if df.empty:
                logger.warning("No data returned from Druid query")
                return None
            
            # Handle timestamp column - Druid uses __time
            if '__time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['__time'])
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                logger.error(f"Expected timestamp column not found. Available columns: {list(df.columns)}")
                return None
            
            # Sort by timestamp (oldest first for LSTM)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Map columns to expected names
            # First, let's see what columns we actually have
            logger.info(f"Available columns: {list(df.columns)}")
            
            # Extract wind data from JSON wind_metrics column (same as CSV processing)
            if 'wind_metrics' in df.columns:
                logger.info("Found wind_metrics column, extracting wind data from JSON...")
                
                wind_speeds = []
                wind_directions = []
                
                for idx, row in df.iterrows():
                    try:
                        # Check if wind_metrics is NaN or not a string
                        if pd.isna(row['wind_metrics']) or not isinstance(row['wind_metrics'], str):
                            logger.warning(f"Skipping row {idx}: Missing or invalid wind data")
                            continue
                            
                        # Parse the JSON wind_metrics
                        wind_metrics = json.loads(row['wind_metrics'])
                        
                        # Extract wind speed (convert from m/s to mph)
                        wind_speed_mps = wind_metrics.get('avg_wind_speed_meters_per_sec', 0)
                        wind_speed_mph = wind_speed_mps * 2.23694  # Convert m/s to mph
                        
                        # Extract wind direction
                        wind_direction = wind_metrics.get('avg_wind_direction_deg', 0)
                        
                        # Only append if we have valid data
                        if wind_speed_mph > 0 and wind_direction >= 0:
                            wind_speeds.append(wind_speed_mph)
                            wind_directions.append(wind_direction)
                        else:
                            logger.warning(f"Skipping row {idx}: Invalid wind values (speed: {wind_speed_mph}, direction: {wind_direction})")
                        
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.warning(f"Skipping row {idx}: Error parsing JSON - {e}")
                        continue
                
                # Create new DataFrame with extracted wind data
                if wind_speeds and wind_directions:
                    df = pd.DataFrame({
                        'wind_speed': wind_speeds,
                        'wind_direction_deg': wind_directions
                    })
                    logger.info(f"Successfully extracted wind data from {len(wind_speeds)} rows")
                else:
                    logger.error("No valid wind data found in wind_metrics JSON")
                    return None
            else:
                # Try to find wind speed and direction columns directly
                wind_speed_col = None
                wind_direction_col = None
                
                # Common column name variations
                speed_variations = ['wind_speed', 'windSpeed', 'wind_speed_mph', 'wind_speed_ms', 'speed']
                direction_variations = ['wind_direction', 'windDirection', 'wind_direction_deg', 'wind_direction_degrees', 'direction']
                
                for col in df.columns:
                    if col.lower() in [v.lower() for v in speed_variations]:
                        wind_speed_col = col
                    elif col.lower() in [v.lower() for v in direction_variations]:
                        wind_direction_col = col
                
                # If we found the columns, rename them to standard names
                if wind_speed_col:
                    df['wind_speed'] = df[wind_speed_col]
                if wind_direction_col:
                    df['wind_direction_deg'] = df[wind_direction_col]
            
            logger.info(f"Successfully fetched {len(df)} records from Druid")
            logger.info(f"Final columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Druid: {e}")
            return None
    
    def get_latest_wind_data(
        self,
        hours_back: int = 6,
        monitoring_system_id: str = '398ae5cb-7971-44b2-b153-c2898ab6fde8'
    ) -> Optional[pd.DataFrame]:
        """
        Get the most recent wind data from Druid.
        
        Args:
            hours_back: Number of hours of data to fetch
            monitoring_system_id: Monitoring system ID to filter by
            
        Returns:
            pd.DataFrame: Recent wind data with required columns
        """
        df = self.get_wind_data_from_druid(hours_back=hours_back, monitoring_system_id=monitoring_system_id)
        
        if df is None or df.empty:
            return None
        
        # Ensure we have the required columns
        required_columns = ['wind_speed', 'wind_direction_deg']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns. Available columns: {list(df.columns)}")
            return None
        
        # Keep only required columns and drop NaN values
        df = df[required_columns].dropna()
        
        if df.empty:
            logger.warning("No valid wind data found after cleaning")
            return None
        
        # Get the most recent data (assuming 15-minute intervals)
        time_steps_needed = hours_back * 4  # 4 time steps per hour
        min_required = time_steps_needed + 24  # Extra for sequence length
        
        # Ensure we have enough data
        if len(df) < min_required:
            logger.warning(f"Only have {len(df)} records, need at least {min_required}")
            # If we don't have enough, use what we have but warn
            recent_data = df
        else:
            recent_data = df.tail(min_required)
        
        logger.info(f"Returning {len(recent_data)} time steps of recent data")
        
        return recent_data

def get_recent_wind_data_from_druid(
    druid_url: str = "http://sc-fenceline-int-dev5.corp.picarro.com:8888",
    datasource: str = "15_minutes_avg_data",
    monitoring_system_id: str = '398ae5cb-7971-44b2-b153-c2898ab6fde8',
    hours_back: int = 6
) -> Optional[pd.DataFrame]:
    """
    Convenience function to get recent wind data from Druid.
    
    Args:
        druid_url: Druid broker URL
        datasource: Druid datasource name (default: 15_minutes_avg_data)
        monitoring_system_id: Monitoring system ID to filter by
        hours_back: Number of hours of data to fetch
        
    Returns:
        pd.DataFrame: Recent wind data
    """
    connector = DruidConnector(druid_url=druid_url, datasource=datasource)
    
    # Test connection first
    if not connector.test_connection():
        logger.error("Failed to connect to Druid database")
        return None
    
    return connector.get_latest_wind_data(hours_back=hours_back, monitoring_system_id=monitoring_system_id)

# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Druid Connector ===")
    
    # Test connection
    connector = DruidConnector()
    if connector.test_connection():
        print("✅ Successfully connected to Druid")
        
        # Fetch recent data
        df = connector.get_latest_wind_data(hours_back=6)
        if df is not None:
            print(f"✅ Successfully fetched {len(df)} records")
            print(f"Data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Sample data:")
            print(df.head())
        else:
            print("❌ Failed to fetch data")
    else:
        print("❌ Failed to connect to Druid") 