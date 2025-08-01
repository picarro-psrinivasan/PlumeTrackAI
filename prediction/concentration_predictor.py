import torch
import numpy as np
import pandas as pd
import json
import sys
import os

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_definitions.concentration_lstm_model import ConcentrationLSTM
from data_handling.concentration_loader import extract_concentration_data

def load_trained_concentration_model(model_path='trained_models/concentration_lstm_model.pth'):
    """
    Load the trained concentration LSTM model.
    
    Args:
        model_path (str): Path to the trained model file
        
    Returns:
        tuple: (model, scaler, metrics, input_size) or (None, None, None, None) if error
    """
    try:
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Extract model parameters
        model_state_dict = checkpoint['model_state_dict']
        scaler = checkpoint['scaler']
        metrics = checkpoint['metrics']
        input_size = checkpoint['input_size']
        
        # Create model instance
        model = ConcentrationLSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            output_steps=6,
            dropout=0.2
        )
        
        # Load the trained weights
        model.load_state_dict(model_state_dict)
        model.eval()
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Model performance: {metrics}")
        
        return model, scaler, metrics, input_size
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None

def prepare_concentration_input_sequence(recent_data, scaler, sequence_length=24):
    """
    Prepare input sequence for concentration prediction.
    
    Args:
        recent_data (pd.DataFrame): Recent concentration data
        scaler: MinMaxScaler used for normalization
        sequence_length (int): Number of time steps to use as input
        
    Returns:
        torch.Tensor: Input sequence tensor
    """
    try:
        # Convert DataFrame to numpy array
        data_array = recent_data.values
        
        # Normalize the data
        data_scaled = scaler.transform(data_array)
        
        # Take the last sequence_length time steps
        if len(data_scaled) >= sequence_length:
            input_sequence = data_scaled[-sequence_length:]
        else:
            # Pad with zeros if not enough data
            padding = np.zeros((sequence_length - len(data_scaled), data_scaled.shape[1]))
            input_sequence = np.vstack([padding, data_scaled])
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0)
        
        return input_tensor
        
    except Exception as e:
        print(f"Error preparing input sequence: {e}")
        return None

def predict_concentration_6hours_ahead(model, input_sequence, scaler):
    """
    Predict concentration values for the next 6 hours.
    
    Args:
        model: Trained concentration LSTM model
        input_sequence (torch.Tensor): Input sequence tensor
        scaler: MinMaxScaler used for normalization
        
    Returns:
        list: List of dictionaries containing predictions for each hour
    """
    try:
        with torch.no_grad():
            # Make prediction
            predictions = model(input_sequence)
            
            # Convert to numpy array
            predictions_np = predictions.squeeze(0).numpy()  # Shape: (6, num_compounds)
            
            # Inverse transform to get original scale
            predictions_original = scaler.inverse_transform(predictions_np)
            
            # Create prediction results
            predictions_list = []
            for hour in range(6):
                hour_pred = predictions_original[hour]
                
                # Create dictionary for this hour
                hour_dict = {
                    'hour': hour + 1,
                    'compounds': {}
                }
                
                # Add each compound's concentration
                for i, concentration in enumerate(hour_pred):
                    compound_name = f"compound_{i+1}"  # You might want to map this to actual compound names
                    hour_dict['compounds'][compound_name] = round(concentration, 3)
                
                predictions_list.append(hour_dict)
            
            return predictions_list
            
    except Exception as e:
        print(f"Error making concentration prediction: {e}")
        return None

def get_recent_concentration_data(data_file='data/15_min_avg_1site_1ms.csv', hours_back=6):
    """
    Get recent concentration data for prediction.
    
    Args:
        data_file (str): Path to the CSV file
        hours_back (int): How many hours of data to get
        
    Returns:
        pd.DataFrame: Recent concentration data
    """
    try:
        # Load the data
        df = pd.read_csv(data_file)
        
        # Extract concentration data from JSON concentrations column
        concentration_df = extract_concentration_data(df)
        
        if concentration_df is None:
            print("Error: Could not extract concentration data")
            return None
        
        # Get the most recent data (assuming 15-minute intervals)
        time_steps_needed = hours_back * 4  # 4 time steps per hour
        recent_data = concentration_df.tail(time_steps_needed + 24)  # Extra for sequence length
        
        print(f"Loaded {len(recent_data)} time steps of recent concentration data")
        print(f"Data range: {recent_data.index[0]} to {recent_data.index[-1]}")
        print(f"Compounds: {list(recent_data.columns)}")
        
        return recent_data
        
    except Exception as e:
        print(f"Error loading recent concentration data: {e}")
        return None

def main():
    """
    Main function to demonstrate concentration prediction.
    """
    print("=== PlumeTrackAI Concentration Prediction ===")
    
    # Load the trained model
    print("\nLoading trained concentration model...")
    model, scaler, metrics, input_size = load_trained_concentration_model()
    
    if model is None:
        return
    
    # Get recent concentration data
    print("\nLoading recent concentration data...")
    recent_data = get_recent_concentration_data()
    
    if recent_data is None:
        return
    
    # Prepare input sequence
    print("\nPreparing input sequence...")
    input_sequence = prepare_concentration_input_sequence(recent_data, scaler, sequence_length=24)
    
    if input_sequence is None:
        return
    
    print(f"Input sequence shape: {input_sequence.shape}")
    
    # Make prediction
    print("\nMaking concentration prediction for 6 hours ahead...")
    predictions = predict_concentration_6hours_ahead(model, input_sequence, scaler)
    
    if predictions is None:
        return
    
    # Display results
    print("\n=== Concentration Predictions (6 hours ahead) ===")
    for pred in predictions:
        print(f"\nHour {pred['hour']}:")
        for compound, concentration in pred['compounds'].items():
            print(f"  {compound}: {concentration} PPB")
    
    # Show recent data for context
    print("\n=== Recent Concentration Data (Last 6 hours) ===")
    recent_6hours = recent_data.tail(24)  # Last 6 hours (24 * 15 min)
    
    print("Current concentrations:")
    for compound in recent_6hours.columns:
        current_value = recent_6hours[compound].iloc[-1]
        avg_value = recent_6hours[compound].mean()
        print(f"  {compound}: {current_value:.3f} PPB (avg: {avg_value:.3f} PPB)")
    
    print("\nConcentration prediction completed!")

if __name__ == "__main__":
    main() 