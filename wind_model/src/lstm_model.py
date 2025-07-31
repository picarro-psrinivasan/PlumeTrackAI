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
from load_data import load_and_preprocess_data

class WindLSTM(nn.Module):
    """
    LSTM model for wind speed and direction prediction.
    """
    
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=3, dropout=0.2):
        """
        Initialize the LSTM model.
        
        Args:
            input_size (int): Number of input features (wind_speed_scaled, wind_dir_sin, wind_dir_cos)
            hidden_size (int): Number of LSTM hidden units
            num_layers (int): Number of LSTM layers
            output_size (int): Number of output features (same as input for wind prediction)
            dropout (float): Dropout rate for regularization
        """
        super(WindLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer for final prediction
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
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
        
        # Final prediction
        output = self.fc(last_output)
        
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
    
    # Loss function and optimizer
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
    
    # Inverse transform to get original scale (for wind speed only)
    # Note: We only inverse transform the wind speed (first column)
    wind_speed_pred = scaler.inverse_transform(all_predictions[:, 0:1]).flatten()
    wind_speed_true = scaler.inverse_transform(all_targets[:, 0:1]).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(wind_speed_true, wind_speed_pred)
    mae = mean_absolute_error(wind_speed_true, wind_speed_pred)
    rmse = np.sqrt(mse)
    
    # Calculate R-squared
    ss_res = np.sum((wind_speed_true - wind_speed_pred) ** 2)
    ss_tot = np.sum((wind_speed_true - np.mean(wind_speed_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    print("\n=== Model Evaluation ===")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R-squared: {r2:.4f}")
    
    return metrics, wind_speed_pred, wind_speed_true

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
        hidden_size=64,    # Number of LSTM hidden units
        num_layers=2,      # Number of LSTM layers
        output_size=3,     # Same as input (predict wind speed and direction)
        dropout=0.2        # Dropout for regularization
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nStarting model training...")
    train_losses, val_losses, best_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        learning_rate=0.001,
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
    }, 'models/wind_lstm_model.pth', _use_new_zipfile_serialization=False)
    
    print("\nModel saved as 'models/wind_lstm_model.pth'")
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 