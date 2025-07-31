
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from typing import Dict, Any, Optional

def setup_openmeteo_client() -> openmeteo_requests.Client:
    """
    Setup the Open-Meteo API client with cache and retry on error.
    
    Returns:
        openmeteo_requests.Client: Configured client with caching and retry logic
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)

def load_wind_data(
    latitude: float = 30.452,
    longitude: float = -91.188,
    client: Optional[openmeteo_requests.Client] = None
) -> pd.DataFrame:
    """
    Load wind and temperature data from Open-Meteo API.
    
    Args:
        latitude (float): Latitude coordinate (default: 30.452)
        longitude (float): Longitude coordinate (default: -91.188)
        client (openmeteo_requests.Client, optional): Pre-configured client. 
                                                    If None, creates a new one.
    
    Returns:
        pd.DataFrame: DataFrame containing hourly weather data with columns:
                     - date: datetime index
                     - temperature_2m: Temperature at 2m height (°C)
                     - wind_speed_10m, wind_speed_80m, wind_speed_120m, wind_speed_180m: Wind speeds at different heights (km/h)
                     - wind_direction_10m, wind_direction_80m, wind_direction_120m, wind_direction_180m: Wind directions at different heights (°)
                     - wind_gusts_10m: Wind gusts at 10m height (km/h)
                     - temperature_80m, temperature_120m, temperature_180m: Temperatures at different heights (°C)
    
    Raises:
        Exception: If API request fails or data processing fails
    """
    # Setup client if not provided
    if client is None:
        client = setup_openmeteo_client()
    
    # Define API parameters
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m", "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
            "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m",
            "wind_gusts_10m", "temperature_80m", "temperature_120m", "temperature_180m"
        ],
    }
    
    # Make API request
    responses = client.weather_api(url, params=params)
    response = responses[0]
    
    # Print location information
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")
    
    # Process hourly data
    hourly = response.Hourly()
    
    # Extract all variables in the same order as requested
    hourly_variables = [
        hourly.Variables(0).ValuesAsNumpy(),  # temperature_2m
        hourly.Variables(1).ValuesAsNumpy(),  # wind_speed_10m
        hourly.Variables(2).ValuesAsNumpy(),  # wind_speed_80m
        hourly.Variables(3).ValuesAsNumpy(),  # wind_speed_120m
        hourly.Variables(4).ValuesAsNumpy(),  # wind_speed_180m
        hourly.Variables(5).ValuesAsNumpy(),  # wind_direction_10m
        hourly.Variables(6).ValuesAsNumpy(),  # wind_direction_80m
        hourly.Variables(7).ValuesAsNumpy(),  # wind_direction_120m
        hourly.Variables(8).ValuesAsNumpy(),  # wind_direction_180m
        hourly.Variables(9).ValuesAsNumpy(),  # wind_gusts_10m
        hourly.Variables(10).ValuesAsNumpy(), # temperature_80m
        hourly.Variables(11).ValuesAsNumpy(), # temperature_120m
        hourly.Variables(12).ValuesAsNumpy(), # temperature_180m
    ]
    
    # Create datetime index
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }
    
    # Add all variables to the data dictionary
    variable_names = [
        "temperature_2m", "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
        "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m",
        "wind_gusts_10m", "temperature_80m", "temperature_120m", "temperature_180m"
    ]
    
    for name, values in zip(variable_names, hourly_variables):
        hourly_data[name] = values
    
    # Create and return DataFrame
    hourly_dataframe = pd.DataFrame(data=hourly_data)
    return hourly_dataframe

def main():
    """
    Main function to demonstrate the data loader usage.
    """
    try:
        # Load wind data for default location (Baton Rouge area)
        print("Loading wind data for Baton Rouge area...")
        df = load_wind_data()
        
        print("\nHourly data preview:")
        print(df.head())
        print(f"\nData shape: {df.shape}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading wind data: {e}")
        return None

if __name__ == "__main__":
    main()