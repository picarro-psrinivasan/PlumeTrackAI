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
            output_steps=24,  # 24 time steps (6 hours * 4 intervals per hour)
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
        dict: Dictionary containing:
            - 'predictions': List of dictionaries for each time step
            - 'dataframe': Pandas DataFrame with predictions
            - 'summary': Summary statistics
    """
    # Compound mapping from model output to actual names
    compound_mapping = {
        'compound_1': 'C2H4Cl2',
        'compound_2': 'C2H4O',
        'compound_3': 'C2H3Cl', 
        'compound_4': 'C4H6',
        'compound_5': 'C4H5Cl',
        'compound_6': 'C6H6'
    }
    
    try:
        with torch.no_grad():
            # Make prediction
            predictions = model(input_sequence)
            
            # Convert to numpy array
            predictions_np = predictions.squeeze(0).numpy()  # Shape: (24, num_compounds)
            
            # Inverse transform to get original scale
            predictions_original = scaler.inverse_transform(predictions_np)
            
            # Create prediction results
            predictions_list = []
            for step in range(24):  # 24 time steps (6 hours * 4 intervals per hour)
                step_pred = predictions_original[step]
                
                # Create dictionary for this time step
                step_dict = {
                    'step': step + 1,
                    'time_minutes': (step + 1) * 15,  # 15-minute intervals
                    'compounds': {}
                }
                
                # Add each compound's concentration
                for i, concentration in enumerate(step_pred):
                    compound_key = f"compound_{i+1}"
                    compound_name = compound_mapping.get(compound_key, compound_key)
                    step_dict['compounds'][compound_name] = round(concentration, 3)
                
                predictions_list.append(step_dict)
            
            # Create DataFrame for easier analysis
            df_data = []
            for pred in predictions_list:
                row = {
                    'time_minutes': pred['time_minutes'],
                    'step': pred['step']
                }
                row.update(pred['compounds'])
                df_data.append(row)
            
            predictions_df = pd.DataFrame(df_data)
            
            # Calculate summary statistics
            summary = {}
            for compound in ['C2H4Cl2', 'C2H4O', 'C2H3Cl', 'C4H6', 'C4H5Cl', 'C6H6']:
                if compound in predictions_df.columns:
                    values = predictions_df[compound].values
                    summary[compound] = {
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'mean': float(values.mean()),
                        'std': float(values.std())
                    }
            
            return {
                'predictions': predictions_list,
                'dataframe': predictions_df,
                'summary': summary
            }
            
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

def get_concentration_predictions(model_path='trained_models/concentration_lstm_model.pth', data_file='data/15_min_avg_1site_1ms.csv'):
    """
    Get concentration predictions for the next 6 hours.
    
    Args:
        model_path (str): Path to the trained model
        data_file (str): Path to the data file
        
    Returns:
        dict: Prediction results containing predictions, dataframe, and summary
    """
    # Load the trained model
    model, scaler, metrics, input_size = load_trained_concentration_model(model_path)
    
    if model is None:
        print("Error: Could not load concentration model.")
        return None
    
    # Get recent concentration data
    recent_data = get_recent_concentration_data(data_file)
    
    if recent_data is None:
        print("Error: Could not load recent concentration data.")
        return None
    
    # Prepare input sequence and make predictions
    input_sequence = prepare_concentration_input_sequence(recent_data, scaler, sequence_length=24)
    
    if input_sequence is None:
        print("Error: Could not prepare input sequence.")
        return None
    
    prediction_results = predict_concentration_6hours_ahead(model, input_sequence, scaler)
    
    if prediction_results is None:
        print("Error: Could not make predictions.")
        return None
    
    return prediction_results

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
    prediction_results = predict_concentration_6hours_ahead(model, input_sequence, scaler)
    
    if prediction_results is None:
        return
    
    predictions = prediction_results['predictions']
    predictions_df = prediction_results['dataframe']
    summary = prediction_results['summary']
    
    # Display results
    print("\n=== Concentration Predictions (6 hours ahead - 15-min intervals) ===")
    for pred in predictions:
        minutes = pred['time_minutes']
        hours = minutes // 60
        mins = minutes % 60
        time_str = f"{hours:02d}:{mins:02d}"
        print(f"\nTime {time_str} (Step {pred['step']}):")
        for compound, concentration in pred['compounds'].items():
            print(f"  {compound}: {concentration} PPB")
    
    # Show recent data for context
    recent_6hours = recent_data.tail(24)  # Last 6 hours (24 * 15 min)
    
    print("Current concentrations:")
    for compound in recent_6hours.columns:
        current_value = recent_6hours[compound].iloc[-1]
        avg_value = recent_6hours[compound].mean()
        print(f"  {compound}: {current_value:.3f} PPB (avg: {avg_value:.3f} PPB)")
    
    print("\nConcentration prediction completed!")

if __name__ == "__main__":
    main() 