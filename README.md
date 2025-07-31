# PlumeTrackAI

A machine learning system for predicting wind speed and direction using LSTM neural networks.

## Project Structure

```
PlumeTrackAI/
├── src/                    # Source code
│   ├── load_data.py       # Data loading and preprocessing
│   ├── lstm_model.py      # LSTM model definition and training
│   └── predict_wind.py    # Wind prediction functionality
├── data/                   # Data files
│   ├── 15_min_avg_1site_1ms.csv  # Wind data CSV file
│   └── .gitkeep
├── models/                 # Trained models
│   └── wind_lstm_model.pth # Saved LSTM model (after training)
├── train_model.py         # Main training script
├── predict.py             # Main prediction script
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
python train_model.py
```

This will:
- Load wind data from `data/15_min_avg_1site_1ms.csv`
- Preprocess the data for LSTM training
- Train the model for 10 epochs
- Save the trained model to `models/wind_lstm_model.pth`

### Making Predictions

```bash
python predict.py
```

This will:
- Load the trained model from `models/wind_lstm_model.pth`
- Load recent wind data
- Predict wind speed and direction 6 hours ahead

## Model Details

- **Architecture**: LSTM with 2 layers, 64 hidden units
- **Input**: 24 time steps (6 hours) of wind data
- **Output**: Wind speed and direction 6 hours ahead
- **Features**: Wind speed (scaled), wind direction (sin/cos components)

## Data Format

The system expects a CSV file with a `wind_metrics` column containing JSON data:
```json
{
  "avg_wind_speed_meters_per_sec": 2.21,
  "avg_wind_direction_deg": 193.38
}
```

## Performance

The model typically achieves:
- **RMSE**: ~0.2-0.3 mph for wind speed
- **R²**: ~0.8-0.9 for wind speed prediction
- **Training time**: 3-6 minutes (10 epochs on CPU)
