# PlumeTrackAI

A machine learning system for predicting wind speed and direction using LSTM neural networks, with gas plume travel time calculation capabilities.

## Project Structure

```
PlumeTrackAI/
├── src/                    # Source code
│   ├── models/            # Model definitions
│   │   ├── __init__.py
│   │   └── lstm_model.py  # LSTM model architecture
│   ├── data/              # Data handling
│   │   ├── __init__.py
│   │   └── loader.py      # Data loading and preprocessing
│   ├── prediction/        # Prediction logic
│   │   ├── __init__.py
│   │   ├── wind_predictor.py    # Wind prediction
│   │   └── plume_calculator.py  # Gas plume calculations
│   └── utils/             # Utilities
│       ├── __init__.py
│       ├── geo_utils.py   # Geographic calculations
│       └── metrics.py     # Evaluation metrics
├── scripts/               # Execution scripts
│   ├── train_model.py     # Train the LSTM model
│   ├── predict_wind.py    # Make wind predictions
│   └── calculate_plume.py # Calculate gas plume travel time
├── data/                  # Data files
│   └── 15_min_avg_1site_1ms.csv
├── models/                # Trained models
│   └── wind_lstm_model.pth
├── tests/                 # Unit tests
├── config/                # Configuration files
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Place your wind data CSV file in the `data/` directory**

## Usage

### Training the Model

```bash
python scripts/train_model.py
```

This will:
- Load wind data from `data/15_min_avg_1site_1ms.csv`
- Preprocess the data for LSTM training
- Train the model for 10 epochs
- Save the trained model to `models/wind_lstm_model.pth`

### Making Wind Predictions

```bash
python scripts/predict_wind.py
```

This will:
- Load the trained model from `models/wind_lstm_model.pth`
- Load recent wind data
- Predict wind speed and direction for next 6 hours

### Calculating Gas Plume Travel Time

```bash
python scripts/calculate_plume.py
```

This will:
- Load the trained model from `models/wind_lstm_model.pth`
- Load recent wind data
- Predict wind for next 6 hours
- Calculate gas plume travel time to risk destination

## Model Details

- **Architecture**: LSTM with 2 layers, 64 hidden units
- **Input**: 24 time steps (6 hours) of wind data
- **Output**: Wind speed and direction for next 6 hours (multi-step prediction)
- **Features**: Wind speed (scaled), wind direction (sin/cos components)

## Data Format

The system expects a CSV file with a `wind_metrics` column containing JSON data:
```json
{
  "avg_wind_speed_meters_per_sec": 2.21,
  "avg_wind_direction_deg": 193.38
}
```

## Gas Plume Calculation

The system uses a time-stepped algorithm to calculate gas plume travel time:
1. **Wind Prediction**: LSTM predicts wind for next 6 hours
2. **Geographic Analysis**: Calculates distance and bearing to risk zone
3. **Effective Wind Speed**: Computes wind component toward destination
4. **Travel Time**: Iteratively calculates progress toward risk zone

## Performance

The model typically achieves:
- **RMSE**: ~0.2-0.3 mph for wind speed
- **R²**: ~0.8-0.9 for wind speed prediction
- **Training time**: 3-6 minutes (10 epochs on CPU)

## Development

### Project Structure Benefits

- **Modularity**: Clear separation of concerns
- **Maintainability**: Easy to find and modify specific functionality
- **Reusability**: Utilities can be used across different components
- **Testability**: Each module can be tested independently

### Adding New Features

1. **New Models**: Add to `src/models/`
2. **New Data Sources**: Add to `src/data/`
3. **New Predictions**: Add to `src/prediction/`
4. **New Utilities**: Add to `src/utils/`
5. **New Scripts**: Add to `scripts/`
