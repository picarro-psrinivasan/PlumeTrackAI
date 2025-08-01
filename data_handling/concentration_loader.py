import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os
import json

def extract_concentration_data(df):
    """
    Extract concentration data from the concentrations JSON column.
    
    Args:
        df (pd.DataFrame): DataFrame with concentrations column
        
    Returns:
        pd.DataFrame: DataFrame with extracted concentration columns for each compound
    """
    concentration_data = []
    
    for idx, row in df.iterrows():
        try:
            # Check if concentrations is NaN or not a string
            if pd.isna(row['concentrations']) or not isinstance(row['concentrations'], str):
                print(f"Skipping row {idx}: Missing or invalid concentration data")
                continue
                
            # Parse the JSON concentrations
            concentrations = json.loads(row['concentrations'])
            
            # Extract concentration values for each compound
            row_concentrations = {}
            for compound_id, compound_data in concentrations.items():
                compound_code = compound_data.get('compound_code', f'compound_{compound_id}')
                concentration = compound_data.get('concentration', 0)
                unit = compound_data.get('unit', 'PPB')
                
                # Only include compounds with valid concentration values
                if concentration > 0:
                    row_concentrations[compound_code] = concentration
            
            if row_concentrations:
                concentration_data.append(row_concentrations)
            else:
                print(f"Skipping row {idx}: No valid concentration values")
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Skipping row {idx}: Error parsing JSON - {e}")
            continue
    
    # Convert to DataFrame
    if concentration_data:
        concentration_df = pd.DataFrame(concentration_data)
        
        # Fill NaN values with 0 (no detection)
        concentration_df = concentration_df.fillna(0)
        
        # Count how many rows we processed
        total_rows_processed = len(df)
        valid_rows = len(concentration_data)
        print(f"Successfully extracted concentration data from {valid_rows}/{total_rows_processed} rows ({valid_rows/total_rows_processed*100:.1f}% valid data)")
        print(f"Compounds found: {list(concentration_df.columns)}")
        print(f"Concentration data shape: {concentration_df.shape}")
        
        return concentration_df
    else:
        print("No valid concentration data found!")
        return None

def load_and_preprocess_concentration_data(file_path="data/15_min_avg_1site_1ms.csv", sequence_length=24, target_steps=24):
    """
    Load and preprocess concentration data for LSTM training.
    
    Args:
        file_path (str): Path to the CSV file
        sequence_length (int): Number of time steps to use as input
        target_steps (int): Number of time steps to predict ahead (15-min intervals)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, scaler)
    """
    
    # Construct full path relative to current working directory
    current_dir = os.getcwd()
    full_path = os.path.join(current_dir, file_path)
    
    print(f"Loading concentration data from: {full_path}")
    
    try:
        # Load CSV
        df = pd.read_csv(full_path)
        print(f"Data loaded successfully! Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
    except FileNotFoundError:
        print(f"Error: File {full_path} not found!")
        print("Please place your CSV file in the data/ directory.")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None
    
    # Extract concentration data from JSON concentrations column
    print("\nExtracting concentration data from JSON concentrations column...")
    concentration_df = extract_concentration_data(df)
    
    if concentration_df is None:
        print("Error: Could not extract concentration data.")
        return None, None, None, None
    
    # Display concentration statistics
    print("\nConcentration Statistics:")
    for column in concentration_df.columns:
        values = concentration_df[column].values
        print(f"{column}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")
    
    # Convert to numpy array
    data = concentration_df.values
    print(f"\nData after cleaning: {data.shape}")
    
    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    print(f"\nCreating sequences with sequence_length={sequence_length}, target_steps={target_steps}")
    X, y = create_sequences(data_scaled, sequence_length, target_steps)
    
    if X is None or y is None:
        print("Error: Could not create sequences.")
        return None, None, None, None
    
    print(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
    
    # Split the data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler

def create_sequences(data, sequence_length, target_steps):
    """
    Create sequences for LSTM training.
    
    Args:
        data (np.array): Input data array
        sequence_length (int): Number of time steps to use as input
        target_steps (int): Number of time steps to predict ahead (15-min intervals)
    
    Returns:
        tuple: (X, y) where X is input sequences and y is target sequences
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length - target_steps + 1):
        X.append(data[i:(i + sequence_length)])
        targets = []
        for step in range(target_steps):
            target_idx = i + sequence_length + step
            if target_idx < len(data):
                targets.append(data[target_idx])
        if len(targets) == target_steps:
            y.append(np.array(targets))
    
    if len(X) == 0 or len(y) == 0:
        print("Error: Could not create sequences. Check sequence_length and target_hours.")
        return None, None
    
    return np.array(X), np.array(y)

def explore_concentration_data(file_path="data/15_min_avg_1site_1ms.csv"):
    """
    Explore the concentration data to understand its structure.
    
    Args:
        file_path (str): Path to the CSV file
    """
    print("=== Concentration Data Exploration ===")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Look at a sample of concentration data
        sample_concentrations = df['concentrations'].iloc[0]
        print(f"\nSample concentration data:")
        print(sample_concentrations[:500] + "..." if len(sample_concentrations) > 500 else sample_concentrations)
        
        # Parse and show structure
        try:
            concentrations = json.loads(sample_concentrations)
            print(f"\nParsed concentration structure:")
            for compound_id, compound_data in concentrations.items():
                print(f"  Compound {compound_id}: {compound_data.get('compound_code', 'Unknown')} - {compound_data.get('concentration', 0)} {compound_data.get('unit', 'Unknown')}")
        except:
            print("Could not parse sample concentration data")
            
    except Exception as e:
        print(f"Error exploring data: {e}")

if __name__ == "__main__":
    # Test the concentration data extraction
    explore_concentration_data() 