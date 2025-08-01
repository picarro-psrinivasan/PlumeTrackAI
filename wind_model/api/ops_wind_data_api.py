
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

def extract_forecast_wind_data(
    latitude: float = 30.452,
    longitude: float = -91.188,
    hours_ahead: int = 6,
    client: Optional[openmeteo_requests.Client] = None
) -> Dict[str, Any]:
    """
    Extract wind speed and wind direction from forecast API for validation.
    
    Args:
        latitude (float): Latitude coordinate (default: 30.452)
        longitude (float): Longitude coordinate (default: -91.188)
        hours_ahead (int): Number of hours ahead to get forecast for (default: 6)
        client (openmeteo_requests.Client, optional): Pre-configured client. 
                                                    If None, creates a new one.
    
    Returns:
        Dict[str, Any]: Dictionary containing forecast wind data:
                       - 'wind_speed_forecast': List of wind speeds (mph)
                       - 'wind_direction_forecast': List of wind directions (degrees)
                       - 'forecast_times': List of forecast timestamps
                       - 'location': Dict with lat/lon coordinates
                       - 'hours_ahead': Number of hours forecasted
    """
    try:
        # Load wind data from API
        df = load_wind_data(latitude, longitude, client)
        
        if df is None or df.empty:
            raise Exception("No data received from API")
        
        # Get current time and filter for future forecasts
        current_time = pd.Timestamp.now()
        # Convert timezone-aware timestamps to timezone-naive for comparison
        df['date_naive'] = df['date'].dt.tz_localize(None)
        future_data = df[df['date_naive'] > current_time].copy()
        df = df.drop('date_naive', axis=1)  # Clean up temporary column
        
        if future_data.empty:
            raise Exception("No future forecast data available")
        
        # Take the first 'hours_ahead' hours of forecast
        forecast_data = future_data.head(hours_ahead)
        
        # Extract wind speed and direction (using 10m height as standard)
        wind_speeds = forecast_data['wind_speed_10m'].tolist()
        wind_directions = forecast_data['wind_direction_10m'].tolist()
        forecast_times = forecast_data['date'].tolist()
        
        # Convert wind speed from km/h to mph for consistency with our model
        wind_speeds_mph = [speed * 0.621371 for speed in wind_speeds]
        
        result = {
            'wind_speed_forecast': wind_speeds_mph,
            'wind_direction_forecast': wind_directions,
            'forecast_times': forecast_times,
            'location': {
                'latitude': latitude,
                'longitude': longitude
            },
            'hours_ahead': hours_ahead,
            'data_source': 'Open-Meteo API',
            'units': {
                'wind_speed': 'mph',
                'wind_direction': 'degrees'
            }
        }
        
        print(f"Extracted forecast data for {hours_ahead} hours ahead:")
        print(f"Location: {latitude}°N, {longitude}°E")
        print(f"Wind speeds (mph): {wind_speeds_mph}")
        print(f"Wind directions (°): {wind_directions}")
        
        return result
        
    except Exception as e:
        print(f"Error extracting forecast wind data: {e}")
        return None

def validate_prediction_with_forecast(
    predicted_wind_speed: float,
    predicted_wind_direction: float,
    latitude: float = 30.452,
    longitude: float = -91.188,
    hours_ahead: int = 6
) -> Dict[str, Any]:
    """
    Validate model prediction against forecast API data.
    
    Args:
        predicted_wind_speed (float): Model predicted wind speed (mph)
        predicted_wind_direction (float): Model predicted wind direction (degrees)
        latitude (float): Latitude coordinate for forecast validation
        longitude (float): Longitude coordinate for forecast validation
        hours_ahead (int): Number of hours ahead for validation
    
    Returns:
        Dict[str, Any]: Validation results with metrics and comparison
    """
    try:
        # Get forecast data for validation
        forecast_data = extract_forecast_wind_data(latitude, longitude, hours_ahead)
        
        if forecast_data is None:
            return {
                'error': 'Could not retrieve forecast data for validation',
                'predicted': {
                    'wind_speed': predicted_wind_speed,
                    'wind_direction': predicted_wind_direction
                }
            }
        
        # Get the forecast value for the target time (hours_ahead)
        forecast_wind_speed = forecast_data['wind_speed_forecast'][hours_ahead - 1]
        forecast_wind_direction = forecast_data['wind_direction_forecast'][hours_ahead - 1]
        
        # Calculate validation metrics
        wind_speed_error = abs(predicted_wind_speed - forecast_wind_speed)
        wind_direction_error = abs(predicted_wind_direction - forecast_wind_direction)
        
        # Normalize wind direction error (account for circular nature)
        if wind_direction_error > 180:
            wind_direction_error = 360 - wind_direction_error
        
        # Calculate percentage errors
        wind_speed_percentage_error = (wind_speed_error / forecast_wind_speed) * 100 if forecast_wind_speed > 0 else 0
        wind_direction_percentage_error = (wind_direction_error / 360) * 100
        
        validation_results = {
            'predicted': {
                'wind_speed': predicted_wind_speed,
                'wind_direction': predicted_wind_direction
            },
            'forecast': {
                'wind_speed': forecast_wind_speed,
                'wind_direction': forecast_wind_direction
            },
            'errors': {
                'wind_speed_error': wind_speed_error,
                'wind_direction_error': wind_direction_error,
                'wind_speed_percentage_error': wind_speed_percentage_error,
                'wind_direction_percentage_error': wind_direction_percentage_error
            },
            'validation_metrics': {
                'wind_speed_accuracy': max(0, 100 - wind_speed_percentage_error),
                'wind_direction_accuracy': max(0, 100 - wind_direction_percentage_error),
                'overall_accuracy': max(0, 100 - (wind_speed_percentage_error + wind_direction_percentage_error) / 2)
            },
            'location': forecast_data['location'],
            'hours_ahead': hours_ahead,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        print(f"\n=== Validation Results ===")
        print(f"Predicted: {predicted_wind_speed:.2f} mph, {predicted_wind_direction:.1f}°")
        print(f"Forecast:  {forecast_wind_speed:.2f} mph, {forecast_wind_direction:.1f}°")
        print(f"Wind Speed Error: {wind_speed_error:.2f} mph ({wind_speed_percentage_error:.1f}%)")
        print(f"Wind Direction Error: {wind_direction_error:.1f}° ({wind_direction_percentage_error:.1f}%)")
        print(f"Overall Accuracy: {validation_results['validation_metrics']['overall_accuracy']:.1f}%")
        
        return validation_results
        
    except Exception as e:
        print(f"Error in validation: {e}")
        return {
            'error': f'Validation failed: {str(e)}',
            'predicted': {
                'wind_speed': predicted_wind_speed,
                'wind_direction': predicted_wind_direction
            }
        }

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
        
        # Test forecast extraction
        print("\n" + "="*50)
        print("Testing forecast extraction...")
        forecast_data = extract_forecast_wind_data()
        
        if forecast_data:
            print(f"Successfully extracted {forecast_data['hours_ahead']} hours of forecast data")
        
        return df
        
    except Exception as e:
        print(f"Error loading wind data: {e}")
        return None

if __name__ == "__main__":
    main()