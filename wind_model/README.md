# Wind Model Package

This package contains all wind model related functionality for PlumeTrackAI.

## Directory Structure

```
wind_model/
├── __init__.py                 # Package initialization
├── README.md                   # This file
├── train_model.py              # Main training script
├── predict.py                  # Main prediction script
├── ops_wind_data_api.py        # Wind data API operations
├── src/                        # Core model implementation
│   ├── __init__.py
│   ├── lstm_model.py          # LSTM model implementation
│   ├── predict_wind.py        # Wind prediction functionality
│   └── load_data.py           # Data loading utilities
└── models/                     # Trained model files
    └── wind_lstm_model.pth    # Trained LSTM model
```

## Usage

### Training the Model
```bash
cd wind_model
python train_model.py
```

### Making Predictions
```bash
cd wind_model
python predict.py
```

### Using the API
```python
from wind_model.src.lstm_model import WindLSTM
from wind_model.src.predict_wind import predict_wind_6hours_ahead
from wind_model.src.load_data import load_and_preprocess_data
```

## Files Description

- **train_model.py**: Main script to train the LSTM model for wind speed and direction prediction
- **predict.py**: Main script to make wind predictions using the trained model
- **ops_wind_data_api.py**: Operations for wind data API integration
- **src/lstm_model.py**: LSTM neural network implementation for wind prediction
- **src/predict_wind.py**: Wind prediction functionality and utilities
- **src/load_data.py**: Data loading, preprocessing, and feature engineering utilities
- **models/wind_lstm_model.pth**: Trained LSTM model weights and scaler

## Dependencies

All dependencies are listed in the root `requirements.txt` file. 