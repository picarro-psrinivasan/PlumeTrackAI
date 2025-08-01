# PlumeTrackAI

A machine learning system for predicting wind speed and direction using LSTM neural networks with forecast-weighted predictions.

## Features

- **LSTM Wind Prediction**: Deep learning model for wind speed and direction forecasting
- **Forecast-Weighted Predictions**: Combines LSTM predictions with weather forecast data
- **REST API**: FastAPI-based web service for easy integration
- **Confidence-Based Weighting**: Automatically adjusts forecast weight based on confidence
- **Validation**: Compares predictions against actual forecast data
- **Multiple Endpoints**: Simple and full prediction options

## Project Structure

```
PlumeTrackAI/
├── data/                   # Data files
│   ├── 15_min_avg_1site_1ms.csv  # Wind data CSV file
│   └── anemometer_1s_1ms.csv     # Additional wind data
├── wind_model/            # Core ML and API code
│   ├── src/               # Source code
│   │   ├── load_data.py       # Data loading and preprocessing
│   │   ├── lstm_model.py      # LSTM model definition and training
│   │   ├── predict_wind.py    # Wind prediction functionality
│   │   └── forecast_weighted_prediction.py  # Forecast-weighted predictions
│   ├── api/               # API endpoints
│   │   ├── fastapi_forecast_weighted.py  # Main API server
│   │   └── ops_wind_data_api.py  # Weather data integration
│   ├── models/            # Trained models
│   │   └── wind_lstm_model.pth  # Saved LSTM model
│   ├── validation/        # Validation scripts
│   └── utils/            # Utility functions
├── train_model.py         # Main training script
├── predict.py             # Main prediction script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd PlumeTrackAI

# Install dependencies
pip install -r requirements.txt

# Activate virtual environment (if using)
source venv/bin/activate
```

### 2. Training the Model

```bash
python train_model.py
```

This will:
- Load wind data from `data/15_min_avg_1site_1ms.csv`
- Preprocess the data for LSTM training
- Train the model for 10 epochs
- Save the trained model to `wind_model/models/wind_lstm_model.pth`

### 3. Making Predictions

```bash
python predict.py
```

This will:
- Load the trained model
- Load recent wind data
- Predict wind speed and direction 6 hours ahead

## API Usage

### Starting the API Server

```bash
cd wind_model/api
python fastapi_forecast_weighted.py
```

The server will start on `http://localhost:8000`

### API Documentation
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### API Endpoints

#### 1. Health Check
```http
GET /health
```

#### 2. Simple Prediction
```http
GET /predict/simple?latitude=30.452&longitude=-91.188&hours_ahead=6
```

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-01-15T10:30:00",
  "base_prediction": {
    "wind_speed_mph": 12.5,
    "wind_direction_degrees": 270.0
  },
  "weighted_prediction": {
    "wind_speed_mph": 14.2,
    "wind_direction_degrees": 275.0
  },
  "forecast_confidence": 0.85,
  "improvement": 2.3
}
```

#### 3. Full Prediction (POST)
```http
POST /predict
Content-Type: application/json

{
  "latitude": 30.452,
  "longitude": -91.188,
  "hours_ahead": 6,
  "forecast_weight": 0.3,
  "confidence_threshold": 0.8
}
```

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-01-15T10:30:00",
  "location": {
    "latitude": 30.452,
    "longitude": -91.188
  },
  "hours_ahead": 6,
  "base_prediction": {
    "wind_speed_mph": 12.5,
    "wind_direction_degrees": 270.0
  },
  "weighted_prediction": {
    "wind_speed_mph": 14.2,
    "wind_direction_degrees": 275.0,
    "base_weight": 0.7,
    "forecast_weight": 0.3
  },
  "forecast_data": {
    "wind_speed_forecast": [15.0, 16.2, 14.8, 13.5, 12.9, 11.8],
    "wind_direction_forecast": [275, 280, 285, 290, 295, 300]
  },
  "forecast_confidence": 0.85,
  "forecast_weight_used": 0.3,
  "validation": {
    "validation_metrics": {
      "overall_accuracy": 87.5,
      "wind_speed_accuracy": 90.2,
      "wind_direction_accuracy": 84.8
    }
  },
  "improvement": 5.4
}
```

### Usage Examples

#### Python Requests
```python
import requests

# Simple prediction
response = requests.get("http://localhost:8000/predict/simple", params={
    'latitude': 30.452,
    'longitude': -91.188,
    'hours_ahead': 6
})
result = response.json()
print(f"Wind: {result['weighted_prediction']['wind_speed_mph']} mph")

# Full prediction
response = requests.post("http://localhost:8000/predict", json={
    "latitude": 30.452,
    "longitude": -91.188,
    "hours_ahead": 6,
    "forecast_weight": 0.4,
    "confidence_threshold": 0.7
})
result = response.json()
print(f"Improvement: {result['improvement']}%")
```

#### cURL
```bash
# Simple prediction
curl "http://localhost:8000/predict/simple?latitude=30.452&longitude=-91.188&hours_ahead=6"

# Full prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 30.452,
    "longitude": -91.188,
    "hours_ahead": 6,
    "forecast_weight": 0.3,
    "confidence_threshold": 0.8
  }'
```

## Model Details

- **Architecture**: LSTM with 2 layers, 64 hidden units
- **Input**: 24 time steps (6 hours) of wind data
- **Output**: Wind speed and direction 6 hours ahead
- **Features**: Wind speed (scaled), wind direction (sin/cos components)
- **Forecast Integration**: Combines LSTM predictions with weather forecast data
- **Confidence Weighting**: Automatically adjusts forecast weight based on data confidence

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
- **API response time**: 2-5 seconds
- **Forecast improvement**: 3-8% over base LSTM predictions

## Understanding the Results

### Base Prediction
- Pure LSTM model output using historical data patterns
- No forecast data influence

### Weighted Prediction
- Combines LSTM prediction with weather forecast data
- Uses confidence-based weighting
- Formula: `weighted = (1-forecast_weight) × base + forecast_weight × forecast`

### Forecast Confidence
- Measures consistency of forecast data (0-1)
- Higher confidence = more reliable forecast data
- Affects the weight given to forecast data

### Improvement
- Percentage improvement of weighted prediction over base prediction
- Based on validation against actual forecast data
- Positive values indicate forecast weighting helped

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid parameters)
- `500`: Internal server error (prediction failed)

## Dependencies

### Core ML
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `torch>=2.0.0` - Deep learning
- `scikit-learn>=1.3.0` - Machine learning utilities

### API
- `fastapi>=0.104.1` - Web framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `pydantic>=2.5.0` - Data validation
- `requests>=2.31.0` - HTTP client
- `openmeteo-requests>=0.1.0` - Weather data API

### Development
- `pytest>=7.4.0` - Testing
- `matplotlib>=3.7.0` - Visualization
- `jupyter>=1.0.0` - Development notebooks

## Troubleshooting

1. **Server won't start**: Check if port 8000 is available
2. **Import errors**: Ensure all dependencies are installed
3. **Model loading fails**: Check if model file exists in `wind_model/models/` directory
4. **Forecast data unavailable**: Check internet connection
5. **Numpy serialization errors**: Fixed in latest version - update dependencies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]
