import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os

def load_and_preprocess_data(file_path="15_min_avg_1site_1ms.csv", sequence_length=24, target_hours=6):
    """
    Load and preprocess wind data for LSTM training.
    
    Args:
        file_path (str): Path to the CSV file (relative to data/ directory)
        sequence_length (int): Number of time steps to use as input
        target_hours (int): Number of hours to predict ahead
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, scaler)
    """
    
    # Construct full path relative to data directory
    data_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(data_dir, file_path)
    
    print(f"Loading data from: {full_path}")
    
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
    
    # Keep only necessary columns
    if 'wind_speed' in df.columns and 'wind_direction_deg' in df.columns:
        df = df[['wind_speed', 'wind_direction_deg']].dropna()
    else:
        print("Error: Required columns 'wind_speed' and 'wind_direction_deg' not found!")
        print(f"Available columns: {list(df.columns)}")
        return None, None, None, None
    
    print(f"Data after cleaning: {df.shape}")
    
    # Convert wind direction (degrees) to sine and cosine
    df['wind_dir_sin'] = np.sin(np.deg2rad(df['wind_direction_deg']))
    df['wind_dir_cos'] = np.cos(np.deg2rad(df['wind_direction_deg']))
    
    # Scale wind speed
    scaler = MinMaxScaler()
    df['wind_speed_scaled'] = scaler.fit_transform(df[['wind_speed']])
    
    # Final features: wind_speed_scaled, wind_dir_sin, wind_dir_cos
    features = df[['wind_speed_scaled', 'wind_dir_sin', 'wind_dir_cos']].values
    
    # Create sequences for LSTM
    X, y = create_sequences(features, sequence_length, target_hours)
    
    print(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
    
    # Split data into train, validation, and test sets
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
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
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler

def create_sequences(data, sequence_length, target_hours):
    """
    Create input sequences and target values for LSTM.
    
    Args:
        data (np.array): Input data
        sequence_length (int): Number of time steps for input
        target_hours (int): Number of hours to predict ahead
    
    Returns:
        tuple: (X, y) sequences
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length - target_hours + 1):
        # Input sequence
        X.append(data[i:(i + sequence_length)])
        # Target (predicting target_hours ahead)
        y.append(data[i + sequence_length + target_hours - 1])
    
    return np.array(X), np.array(y)

def explore_data(file_path="15_min_avg_1site_1ms.csv"):
    """
    Explore the data to understand its structure.
    """
    data_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(data_dir, file_path)
    
    try:
        df = pd.read_csv(full_path)
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First 5 rows:")
        print(df.head())
        print(f"\nData types:")
        print(df.dtypes)
        print(f"\nMissing values:")
        print(df.isnull().sum())
        print(f"\nBasic statistics:")
        print(df.describe())
        
        return df
    except Exception as e:
        print(f"Error exploring data: {e}")
        return None

if __name__ == "__main__":
    # Explore the data first
    df = explore_data()
    
    if df is not None:
        print("\n=== Loading and Preprocessing Data ===")
        train_loader, val_loader, test_loader, scaler = load_and_preprocess_data()
        
        if train_loader is not None:
            print("\n=== Data Loaders Created Successfully ===")
            print(f"Train batches: {len(train_loader)}")
            print(f"Validation batches: {len(val_loader)}")
            print(f"Test batches: {len(test_loader)}")
            
            # Show example batch
            for batch_X, batch_y in train_loader:
                print(f"Example batch X shape: {batch_X.shape}")
                print(f"Example batch y shape: {batch_y.shape}")
                break