#!/usr/bin/env python3
"""
Forecast-Aware LSTM Training
This module fine-tunes the LSTM model using forecast data from ops_wind_data_api.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add paths
sys.path.append('..')
sys.path.append('../api')

# Handle imports based on how the script is run
try:
    from lstm_model import WindLSTM
    from load_data import load_and_preprocess_data
    from api.ops_wind_data_api import extract_forecast_wind_data, validate_prediction_with_forecast
except ImportError:
    # If running directly, adjust paths
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lstm_model import WindLSTM
    from load_data import load_and_preprocess_data
    from api.ops_wind_data_api import extract_forecast_wind_data, validate_prediction_with_forecast

class ForecastAwareLoss(nn.Module):
    """
    Custom loss function that incorporates forecast data for better training.
    """
    
    def __init__(self, forecast_weight=0.3, alpha=0.7):
        """
        Initialize forecast-aware loss.
        
        Args:
            forecast_weight (float): Weight for forecast-based loss component
            alpha (float): Balance between MSE and forecast alignment
        """
        super(ForecastAwareLoss, self).__init__()
        self.forecast_weight = forecast_weight
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions, targets, forecast_data=None):
        """
        Compute forecast-aware loss.
        
        Args:
            predictions: Model predictions
            targets: Actual targets
            forecast_data: Forecast data for alignment
            
        Returns:
            torch.Tensor: Combined loss
        """
        # Standard MSE loss
        mse_loss = self.mse_loss(predictions, targets)
        
        if forecast_data is not None:
            # Forecast alignment loss
            forecast_loss = self.compute_forecast_alignment_loss(predictions, forecast_data)
            # Combine losses
            total_loss = self.alpha * mse_loss + (1 - self.alpha) * forecast_loss
        else:
            total_loss = mse_loss
            
        return total_loss
    
    def compute_forecast_alignment_loss(self, predictions, forecast_data):
        """
        Compute loss that encourages predictions to align with forecast data.
        """
        # Extract wind speed predictions (first column)
        pred_wind_speed = predictions[:, 0]
        
        # Get forecast wind speeds
        forecast_wind_speeds = torch.tensor(forecast_data['wind_speed_forecast'], 
                                          dtype=torch.float32, device=predictions.device)
        
        # Compute alignment loss
        alignment_loss = self.mse_loss(pred_wind_speed, forecast_wind_speeds)
        
        return alignment_loss

def train_with_forecast_awareness(
    model,
    train_loader,
    val_loader,
    forecast_data,
    num_epochs=20,
    learning_rate=0.001,
    device='cpu',
    forecast_weight=0.3
):
    """
    Train model with forecast awareness.
    
    Args:
        model: LSTM model
        train_loader: Training data loader
        val_loader: Validation data loader
        forecast_data: Forecast data for alignment
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        forecast_weight: Weight for forecast alignment
        
    Returns:
        tuple: (train_losses, val_losses, best_model)
    """
    
    model = model.to(device)
    
    # Use forecast-aware loss
    criterion = ForecastAwareLoss(forecast_weight=forecast_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    
    print(f"Training with forecast awareness for {num_epochs} epochs...")
    print(f"Forecast weight: {forecast_weight}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Compute forecast-aware loss
            loss = criterion(output, target, forecast_data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target, forecast_data)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}")
    
    return train_losses, val_losses, best_model

def fine_tune_with_forecast_data(
    model_path='models/wind_lstm_model.pth',
    latitude=30.452,
    longitude=-91.188,
    hours_ahead=6,
    num_epochs=10,
    learning_rate=0.0001
):
    """
    Fine-tune existing model with forecast data.
    
    Args:
        model_path: Path to existing model
        latitude: Latitude for forecast data
        longitude: Longitude for forecast data
        hours_ahead: Hours ahead for forecast
        num_epochs: Number of fine-tuning epochs
        learning_rate: Learning rate for fine-tuning
        
    Returns:
        dict: Fine-tuning results
    """
    
    print("=== Forecast-Aware Model Fine-tuning ===")
    
    # Load existing model
    print("Loading existing model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = WindLSTM(
        input_size=3,
        hidden_size=64,
        num_layers=2,
        output_size=3,
        dropout=0.2
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = checkpoint['scaler']
    
    print("✅ Model loaded successfully")
    
    # Get forecast data
    print("Retrieving forecast data...")
    forecast_data = extract_forecast_wind_data(latitude, longitude, hours_ahead)
    
    if forecast_data is None:
        print("❌ Could not retrieve forecast data")
        return None
    
    print("✅ Forecast data retrieved")
    
    # Load and preprocess data
    print("Loading training data...")
    train_loader, val_loader, test_loader, _ = load_and_preprocess_data()
    
    if train_loader is None:
        print("❌ Could not load training data")
        return None
    
    print("✅ Training data loaded")
    
    # Fine-tune with forecast awareness
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_losses, val_losses, best_model = train_with_forecast_awareness(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        forecast_data=forecast_data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        forecast_weight=0.3
    )
    
    # Save fine-tuned model
    fine_tuned_path = 'models/wind_lstm_model_fine_tuned.pth'
    torch.save({
        'model_state_dict': best_model,
        'scaler': scaler,
        'forecast_data': forecast_data,
        'fine_tuning_metrics': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': min(val_losses)
        }
    }, fine_tuned_path)
    
    print(f"✅ Fine-tuned model saved as: {fine_tuned_path}")
    
    # Test fine-tuned model
    print("\nTesting fine-tuned model...")
    model.load_state_dict(best_model)
    model.eval()
    
    # Make a test prediction
    from predict_wind import get_recent_wind_data, prepare_input_sequence, predict_wind_6hours_ahead
    
    recent_data = get_recent_wind_data()
    if recent_data is not None:
        input_sequence = prepare_input_sequence(recent_data, scaler)
        if input_sequence is not None:
            prediction = predict_wind_6hours_ahead(model, input_sequence, scaler)
            
            if prediction:
                print(f"Fine-tuned prediction: {prediction['wind_speed_mph']:.2f} mph, {prediction['wind_direction_degrees']:.1f}°")
                
                # Validate against forecast
                validation = validate_prediction_with_forecast(
                    predicted_wind_speed=prediction['wind_speed_mph'],
                    predicted_wind_direction=prediction['wind_direction_degrees'],
                    latitude=latitude,
                    longitude=longitude,
                    hours_ahead=hours_ahead
                )
                
                if 'validation_metrics' in validation:
                    accuracy = validation['validation_metrics']['overall_accuracy']
                    print(f"Fine-tuned model accuracy: {accuracy:.1f}%")
    
    return {
        'model_path': fine_tuned_path,
        'forecast_data': forecast_data,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': min(val_losses)
    }

def main():
    """Main function to run forecast-aware fine-tuning."""
    print("=== PlumeTrackAI Forecast-Aware Fine-tuning ===")
    
    results = fine_tune_with_forecast_data(
        model_path='models/wind_lstm_model.pth',
        latitude=30.452,
        longitude=-91.188,
        hours_ahead=6,
        num_epochs=10,
        learning_rate=0.0001
    )
    
    if results:
        print("\n✅ Fine-tuning completed successfully!")
        print(f"Best validation loss: {results['best_val_loss']:.6f}")
    else:
        print("\n❌ Fine-tuning failed!")

if __name__ == "__main__":
    main() 