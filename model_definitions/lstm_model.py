import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
import os

# Add the src directory to the path so we can import load_data
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_handling.loader import load_and_preprocess_data

class WindLSTM(nn.Module):
    """
    LSTM model for wind speed and direction prediction with multi-step output.
    """
    
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_steps=6, dropout=0.2):
        """
        Initialize the LSTM model.
        
        Args:
            input_size (int): Number of input features (wind_speed_scaled, wind_dir_sin, wind_dir_cos)
            hidden_size (int): Number of LSTM hidden units
            num_layers (int): Number of LSTM layers
            output_steps (int): Number of time steps to predict ahead (default: 6 hours)
            dropout (float): Dropout rate for regularization
        """
        super(WindLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer for each time step prediction
        self.fc = nn.Linear(hidden_size, input_size * output_steps)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_steps * input_size)
                          Reshaped to (batch_size, output_steps, input_size) for convenience
        """
        # Initialize hidden state and cell state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last time step output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Final prediction for all time steps
        output = self.fc(last_output)
        
        # Reshape to (batch_size, output_steps, input_size)
        output = output.view(batch_size, self.output_steps, -1)
        
        return output

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cpu'):
    """
    Train the LSTM model.
    
    Args:
        model: LSTM model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        device (str): Device to train on ('cpu' or 'cuda')
        
    Returns:
        tuple: (train_losses, val_losses, best_model)
    """
    
    # Move model to device
    model = model.to(device)
    
    # Simple MSE loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Lists to store losses
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_model = None
    
    print(f"Training on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            # Move data to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Move data to device
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}')
    
    print(f"Training completed! Best validation loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses, best_model

def evaluate_model(model, test_loader, scaler, device='cpu'):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained LSTM model
        test_loader: Test data loader
        scaler: Fitted scaler for inverse transformation
        device (str): Device to evaluate on
        
    Returns:
        dict: Evaluation metrics
    """
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            # Move data to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Get predictions
            predictions = model(batch_X)
            
            # Move back to CPU for numpy operations
            predictions = predictions.cpu().numpy()
            targets = batch_y.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Reshape predictions and targets for evaluation
    # all_predictions shape: (batch_size, 6, 3) -> (batch_size * 6, 3)
    # all_targets shape: (batch_size, 6, 3) -> (batch_size * 6, 3)
    batch_size = all_predictions.shape[0]
    all_predictions_flat = all_predictions.reshape(-1, 3)
    all_targets_flat = all_targets.reshape(-1, 3)
    
    # Inverse transform ALL features to get original scale
    predictions_original = scaler.inverse_transform(all_predictions_flat)
    targets_original = scaler.inverse_transform(all_targets_flat)
    
    # Extract wind speed (first column)
    wind_speed_pred = predictions_original[:, 0]
    wind_speed_true = targets_original[:, 0]
    
    # Extract wind direction components (sin/cos)
    wind_dir_sin_pred = predictions_original[:, 1]
    wind_dir_cos_pred = predictions_original[:, 2]
    wind_dir_sin_true = targets_original[:, 1]
    wind_dir_cos_true = targets_original[:, 2]
    
    # Convert sin/cos back to degrees for direction evaluation
    wind_dir_pred_deg = np.degrees(np.arctan2(wind_dir_sin_pred, wind_dir_cos_pred))
    wind_dir_true_deg = np.degrees(np.arctan2(wind_dir_sin_true, wind_dir_cos_true))
    
    # Normalize angles to 0-360 degrees
    wind_dir_pred_deg = (wind_dir_pred_deg + 360) % 360
    wind_dir_true_deg = (wind_dir_true_deg + 360) % 360
    
    # Calculate wind speed metrics
    wind_speed_mse = mean_squared_error(wind_speed_true, wind_speed_pred)
    wind_speed_mae = mean_absolute_error(wind_speed_true, wind_speed_pred)
    wind_speed_rmse = np.sqrt(wind_speed_mse)
    
    # Calculate wind direction metrics (handle circular nature)
    wind_dir_diff = np.abs(wind_dir_pred_deg - wind_dir_true_deg)
    wind_dir_diff = np.minimum(wind_dir_diff, 360 - wind_dir_diff)  # Handle circular distance
    wind_dir_mae = np.mean(wind_dir_diff)
    wind_dir_rmse = np.sqrt(np.mean(wind_dir_diff**2))
    
    # Calculate R-squared for wind speed
    ss_res = np.sum((wind_speed_true - wind_speed_pred) ** 2)
    ss_tot = np.sum((wind_speed_true - np.mean(wind_speed_true)) ** 2)
    wind_speed_r2 = 1 - (ss_res / ss_tot)
    
    # Calculate overall metrics (average of wind speed and direction)
    overall_mse = (wind_speed_mse + wind_dir_rmse**2) / 2
    overall_mae = (wind_speed_mae + wind_dir_mae) / 2
    overall_rmse = np.sqrt(overall_mse)
    
    metrics = {
        'Wind_Speed_MSE': wind_speed_mse,
        'Wind_Speed_MAE': wind_speed_mae,
        'Wind_Speed_RMSE': wind_speed_rmse,
        'Wind_Speed_R2': wind_speed_r2,
        'Wind_Direction_MAE': wind_dir_mae,
        'Wind_Direction_RMSE': wind_dir_rmse,
        'Overall_MSE': overall_mse,
        'Overall_MAE': overall_mae,
        'Overall_RMSE': overall_rmse
    }
    
    print("\n=== Model Evaluation ===")
    print("Wind Speed Metrics:")
    print(f"  MSE: {wind_speed_mse:.4f} mph²")
    print(f"  MAE: {wind_speed_mae:.4f} mph")
    print(f"  RMSE: {wind_speed_rmse:.4f} mph")
    print(f"  R-squared: {wind_speed_r2:.4f}")
    print("\nWind Direction Metrics:")
    print(f"  MAE: {wind_dir_mae:.2f} degrees")
    print(f"  RMSE: {wind_dir_rmse:.2f} degrees")
    print("\nOverall Model Metrics:")
    print(f"  Overall MSE: {overall_mse:.4f}")
    print(f"  Overall MAE: {overall_mae:.4f}")
    print(f"  Overall RMSE: {overall_rmse:.4f}")
    
    # Print sample predictions
    print(f"\nSample Predictions (first 5):")
    print("Time | Pred Speed | True Speed | Pred Dir | True Dir")
    print("-" * 50)
    for i in range(min(5, len(wind_speed_pred))):
        print(f"{i+1:4d} | {wind_speed_pred[i]:10.2f} | {wind_speed_true[i]:10.2f} | {wind_dir_pred_deg[i]:8.1f}° | {wind_dir_true_deg[i]:8.1f}°")
    
    return metrics, predictions_original, targets_original

# def plot_training_history(train_losses, val_losses):
#     """
#     Plot training and validation loss history.
#     """
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label='Training Loss', color='blue')
#     plt.plot(val_losses, label='Validation Loss', color='red')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def plot_predictions(wind_speed_pred, wind_speed_true, num_points=100):
#     """
#     Plot actual vs predicted wind speeds.
#     """
#     plt.figure(figsize=(12, 6))
    
#     # Plot first 100 points for clarity
#     x = range(min(num_points, len(wind_speed_pred)))
#     plt.plot(x, wind_speed_true[:num_points], label='Actual', color='blue', alpha=0.7)
#     plt.plot(x, wind_speed_pred[:num_points], label='Predicted', color='red', alpha=0.7)
    
#     plt.xlabel('Time Steps')
#     plt.ylabel('Wind Speed (mph)')
#     plt.title('Actual vs Predicted Wind Speed')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def main():
    """
    Main function to train and evaluate the LSTM model.
    """
    print("=== PlumeTrackAI LSTM Wind Prediction Model ===")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    train_loader, val_loader, test_loader, scaler = load_and_preprocess_data(
        sequence_length=24,  # 6 hours of lookback
        target_hours=6       # Predict 6 hours ahead
    )
    
    if train_loader is None:
        print("Error: Could not load data. Please check your CSV file.")
        return
    
    # Create model
    print("\nCreating LSTM model...")
    model = WindLSTM(
        input_size=3,      # wind_speed_scaled, wind_dir_sin, wind_dir_cos
        hidden_size=64,    # Back to 64 - simpler is better
        num_layers=2,      # Back to 2 layers - avoid overfitting
        output_steps=6,    # Predict 6 hours ahead (one prediction per hour)
        dropout=0.2        # Back to 0.2 - balanced regularization
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nStarting model training...")
    train_losses, val_losses, best_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=200,  # More epochs with simpler model
        learning_rate=0.001,  # Back to standard learning rate
        device=device
    )
    
    # Load best model
    model.load_state_dict(best_model)
    
    # Plot training history
    # plot_training_history(train_losses, val_losses)
    
    # Evaluate model
    metrics, wind_speed_pred, wind_speed_true = evaluate_model(
        model=model,
        test_loader=test_loader,
        scaler=scaler,
        device=device
    )
    
    # Plot predictions
    # plot_predictions(wind_speed_pred, wind_speed_true)
    
    # Save the trained model
    torch.save({
        'model_state_dict': best_model,
        'scaler': scaler,
        'metrics': metrics
    }, 'trained_models/wind_lstm_model.pth', _use_new_zipfile_serialization=False)
    
    print("\nModel saved as 'trained_models/wind_lstm_model.pth'")
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 