import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
import os

# Add the current directory to the path so we can import concentration_loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_handling.concentration_loader import load_and_preprocess_concentration_data

class ConcentrationLSTM(nn.Module):
    """
    LSTM model for concentration prediction with multi-step output.
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_steps=6, dropout=0.2):
        """
        Initialize the LSTM model for concentration prediction.
        
        Args:
            input_size (int): Number of input features (number of compounds)
            hidden_size (int): Number of LSTM hidden units
            num_layers (int): Number of LSTM layers
            output_steps (int): Number of time steps to predict ahead (default: 6 hours)
            dropout (float): Dropout rate for regularization
        """
        super(ConcentrationLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        self.input_size = input_size
        
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
            torch.Tensor: Output tensor of shape (batch_size, output_steps, input_size)
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
        output = output.view(batch_size, self.output_steps, self.input_size)
        
        return output

def train_concentration_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cpu'):
    """
    Train the concentration LSTM model.
    
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
    print(f"Training on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    print(f"Training completed! Best validation loss: {best_val_loss:.6f}")
    return train_losses, val_losses, best_model

def evaluate_concentration_model(model, test_loader, scaler, device='cpu'):
    """
    Evaluate the concentration LSTM model.
    
    Args:
        model: LSTM model
        test_loader: Test data loader
        scaler: MinMaxScaler used for normalization
        device (str): Device to evaluate on ('cpu' or 'cuda')
        
    Returns:
        tuple: (metrics, predictions, actual_values)
    """
    print("\n=== Model Evaluation ===")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            
            # Reshape outputs and targets for evaluation
            batch_predictions = outputs.reshape(-1, outputs.size(-1))
            batch_targets = batch_y.reshape(-1, batch_y.size(-1))
            
            all_predictions.append(batch_predictions.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Inverse transform to get original scale
    all_predictions_original = scaler.inverse_transform(all_predictions)
    all_targets_original = scaler.inverse_transform(all_targets)
    
    # Calculate metrics
    mse = mean_squared_error(all_targets_original, all_predictions_original)
    mae = mean_absolute_error(all_targets_original, all_predictions_original)
    rmse = np.sqrt(mse)
    
    # Calculate R-squared for each compound
    r2_scores = []
    for i in range(all_targets_original.shape[1]):
        if np.var(all_targets_original[:, i]) > 0:
            r2 = 1 - np.sum((all_targets_original[:, i] - all_predictions_original[:, i])**2) / np.sum((all_targets_original[:, i] - np.mean(all_targets_original[:, i]))**2)
            r2_scores.append(r2)
        else:
            r2_scores.append(0)
    
    avg_r2 = np.mean(r2_scores)
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': avg_r2,
        'R2_per_compound': r2_scores
    }
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R-squared: {avg_r2:.4f}")
    
    return metrics, all_predictions_original, all_targets_original

def main():
    """
    Main function to train and evaluate the concentration LSTM model.
    """
    print("=== PlumeTrackAI Concentration LSTM Model ===")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess concentration data
    print("\nLoading and preprocessing concentration data...")
    train_loader, val_loader, test_loader, scaler = load_and_preprocess_concentration_data(
        sequence_length=24,  # 6 hours of lookback
        target_hours=6       # Predict 6 hours ahead
    )
    
    if train_loader is None:
        print("Error: Could not load concentration data. Please check your CSV file.")
        return
    
    # Get input size from the data
    sample_batch = next(iter(train_loader))
    input_size = sample_batch[0].shape[-1]  # Number of compounds
    print(f"Number of compounds (input features): {input_size}")
    
    # Create model
    print("\nCreating concentration LSTM model...")
    model = ConcentrationLSTM(
        input_size=input_size,  # Number of compounds
        hidden_size=64,         # Number of LSTM hidden units
        num_layers=2,           # Number of LSTM layers
        output_steps=6,         # Predict 6 hours ahead (one prediction per hour)
        dropout=0.2             # Dropout for regularization
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nStarting model training...")
    train_losses, val_losses, best_model = train_concentration_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        learning_rate=0.001,
        device=device
    )
    
    # Load best model
    model.load_state_dict(best_model)
    
    # Evaluate model
    metrics, predictions, actual_values = evaluate_concentration_model(
        model=model,
        test_loader=test_loader,
        scaler=scaler,
        device=device
    )
    
    # Save the trained model
    torch.save({
        'model_state_dict': best_model,
        'scaler': scaler,
        'metrics': metrics,
        'input_size': input_size
    }, 'trained_models/concentration_lstm_model.pth', _use_new_zipfile_serialization=False)
    
    print("\nModel saved as 'trained_models/concentration_lstm_model.pth'")
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 