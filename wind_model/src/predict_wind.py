import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add the src directory to the path so we can import load_data
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data import load_and_preprocess_data
from lstm_model import WindLSTM

def load_trained_model(model_path='models/wind_lstm_model.pth'):
    """
    Load the trained LSTM model and scaler.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        tuple: (model, scaler)
    """
    try:
        # Load the saved model with weights_only=False for compatibility
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Create model instance
        model = WindLSTM(
            input_size=3,
            hidden_size=64,
            num_layers=2,
            output_size=3,
            dropout=0.2
        )
        
        # Load the trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load the scaler
        scaler = checkpoint['scaler']
        
        # Set model to evaluation mode
        model.eval()
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Model performance: {checkpoint['metrics']}")
        
        return model, scaler
        
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found!")
        print("Please train the model first using: python lstm_model.py")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def prepare_input_sequence(recent_data, scaler, sequence_length=24):
    """
    Prepare input sequence for prediction.
    
    Args:
        recent_data (pd.DataFrame): Recent wind data with columns ['wind_speed', 'wind_direction_deg']
        scaler: Fitted scaler for wind speed
        sequence_length (int): Number of time steps needed for prediction
        
    Returns:
        torch.Tensor: Prepared input sequence
    """
    
    # Ensure we have enough data
    if len(recent_data) < sequence_length:
        print(f"Error: Need at least {sequence_length} time steps, but only have {len(recent_data)}")
        return None
    
    # Take the most recent sequence_length time steps
    recent_data = recent_data.tail(sequence_length).copy()
    
    # Convert wind direction to sin/cos
    recent_data['wind_dir_sin'] = np.sin(np.deg2rad(recent_data['wind_direction_deg']))
    recent_data['wind_dir_cos'] = np.cos(np.deg2rad(recent_data['wind_direction_deg']))
    
    # Scale wind speed
    wind_speed_scaled = scaler.transform(recent_data[['wind_speed']])
    
    # Create feature array
    features = np.column_stack([
        wind_speed_scaled.flatten(),
        recent_data['wind_dir_sin'].values,
        recent_data['wind_dir_cos'].values
    ])
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.FloatTensor(features).unsqueeze(0)  # Shape: (1, sequence_length, 3)
    
    return input_tensor

def predict_wind_6hours_ahead(model, input_sequence, scaler):
    """
    Predict wind speed and direction 6 hours ahead.
    
    Args:
        model: Trained LSTM model
        input_sequence (torch.Tensor): Input sequence
        scaler: Fitted scaler for inverse transformation
        
    Returns:
        dict: Predicted wind speed and direction
    """
    
    model.eval()
    
    with torch.no_grad():
        # Make prediction
        prediction = model(input_sequence)
        
        # Convert to numpy
        prediction_np = prediction.cpu().numpy()
        
        # Inverse transform wind speed to original scale
        wind_speed_pred = scaler.inverse_transform(prediction_np[:, 0:1])[0, 0]
        
        # Get wind direction from sin/cos
        wind_dir_sin = prediction_np[0, 1]
        wind_dir_cos = prediction_np[0, 2]
        wind_direction_pred = np.degrees(np.arctan2(wind_dir_sin, wind_dir_cos))
        
        # Ensure direction is between 0 and 360 degrees
        if wind_direction_pred < 0:
            wind_direction_pred += 360
        
        return {
            'wind_speed_mph': round(wind_speed_pred, 2),
            'wind_direction_degrees': round(wind_direction_pred, 1)
        }

def get_recent_wind_data(data_file='../data/15_min_avg_1site_1ms.csv', hours_back=6):
    """
    Get recent wind data for prediction.
    
    Args:
        data_file (str): Path to the CSV file
        hours_back (int): How many hours of data to get
        
    Returns:
        pd.DataFrame: Recent wind data
    """
    
    try:
        # Load the data
        df = pd.read_csv(data_file)
        
        # Extract wind data from JSON wind_metrics column
        from load_data import extract_wind_data
        df = extract_wind_data(df)
        
        # Keep only wind speed and direction columns
        if 'wind_speed' in df.columns and 'wind_direction_deg' in df.columns:
            df = df[['wind_speed', 'wind_direction_deg']].dropna()
        else:
            print("Error: Required columns not found in data file")
            return None
        
        # Get the most recent data (assuming 15-minute intervals)
        time_steps_needed = hours_back * 4  # 4 time steps per hour
        recent_data = df.tail(time_steps_needed + 24)  # Extra for sequence length
        
        print(f"Loaded {len(recent_data)} time steps of recent data")
        print(f"Data range: {recent_data.index[0]} to {recent_data.index[-1]}")
        
        return recent_data
        
    except Exception as e:
        print(f"Error loading recent data: {e}")
        return None

def main():
    """
    Main function to demonstrate wind prediction.
    """
    print("=== PlumeTrackAI Wind Prediction ===")
    
    # Load the trained model
    print("\nLoading trained model...")
    model, scaler = load_trained_model()
    
    if model is None:
        return
    
    # Get recent wind data
    print("\nLoading recent wind data...")
    recent_data = get_recent_wind_data()
    
    if recent_data is None:
        return
    
    # Prepare input sequence
    print("\nPreparing input sequence...")
    input_sequence = prepare_input_sequence(recent_data, scaler, sequence_length=24)
    
    if input_sequence is None:
        return
    
    print(f"Input sequence shape: {input_sequence.shape}")
    
    # Make prediction
    print("\nMaking prediction for 6 hours ahead...")
    prediction = predict_wind_6hours_ahead(model, input_sequence, scaler)
    
    # Display results
    print("\n=== Wind Prediction (6 hours ahead) ===")
    print(f"Predicted Wind Speed: {prediction['wind_speed_mph']} mph")
    print(f"Predicted Wind Direction: {prediction['wind_direction_degrees']}°")
    
    # Show recent data for context
    print("\n=== Recent Wind Data (Last 6 hours) ===")
    recent_6hours = recent_data.tail(24)  # Last 6 hours (24 * 15 min)
    print(f"Current Wind Speed: {recent_6hours['wind_speed'].iloc[-1]:.1f} mph")
    print(f"Current Wind Direction: {recent_6hours['wind_direction_deg'].iloc[-1]:.1f}°")
    
    # Show trend
    avg_speed_6h = recent_6hours['wind_speed'].mean()
    print(f"Average Wind Speed (6h): {avg_speed_6h:.1f} mph")
    
    if prediction['wind_speed_mph'] > avg_speed_6h:
        trend = "increasing"
    elif prediction['wind_speed_mph'] < avg_speed_6h:
        trend = "decreasing"
    else:
        trend = "stable"
    
    print(f"Wind Speed Trend: {trend}")
    
    return prediction

def predict_from_custom_data(wind_speeds, wind_directions, model_path='models/wind_lstm_model.pth'):
    """
    Make prediction using custom wind data.
    
    Args:
        wind_speeds (list): List of wind speeds (mph)
        wind_directions (list): List of wind directions (degrees)
        model_path (str): Path to the trained model
        
    Returns:
        dict: Predicted wind speed and direction
    """
    
    # Load model
    model, scaler = load_trained_model(model_path)
    if model is None:
        return None
    
    # Create DataFrame
    data = pd.DataFrame({
        'wind_speed': wind_speeds,
        'wind_direction_deg': wind_directions
    })
    
    # Prepare input sequence
    input_sequence = prepare_input_sequence(data, scaler, sequence_length=24)
    if input_sequence is None:
        return None
    
    # Make prediction
    prediction = predict_wind_6hours_ahead(model, input_sequence, scaler)
    
    return prediction

if __name__ == "__main__":
    # Run the main prediction
    prediction = main()
    
    # Example of using custom data
    print("\n" + "="*50)
    print("Example: Prediction from custom data")
    print("="*50)
    
    # Example: 24 time steps of wind data (6 hours)
    custom_wind_speeds = [10, 12, 15, 18, 20, 22, 25, 23, 21, 19, 17, 15, 
                         13, 11, 9, 8, 7, 6, 5, 4, 3, 4, 5, 6]
    custom_wind_directions = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                             105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160]
    
    custom_prediction = predict_from_custom_data(custom_wind_speeds, custom_wind_directions)
    
    if custom_prediction:
        print(f"Custom Data Prediction:")
        print(f"Wind Speed: {custom_prediction['wind_speed_mph']} mph")
        print(f"Wind Direction: {custom_prediction['wind_direction_degrees']}°") 