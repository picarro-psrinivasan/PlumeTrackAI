# PlumeTrackAI FastAPI Forecast-Weighted Prediction API

This FastAPI endpoint provides REST API access to forecast-weighted wind predictions, combining LSTM model predictions with weather forecast data.

## Features

- **Base LSTM Prediction**: Pure LSTM model predictions
- **Forecast-Weighted Prediction**: LSTM predictions enhanced with weather forecast data
- **Confidence-Based Weighting**: Automatically adjusts forecast weight based on confidence
- **Validation**: Compares predictions against actual forecast data
- **Multiple Endpoints**: Simple and full prediction options

## Installation

1. Install FastAPI dependencies:
```bash
pip install -r requirements_fastapi.txt
```

2. Ensure the main PlumeTrackAI dependencies are installed:
```bash
pip install -r ../requirements.txt
```

## Running the API Server

### Start the server:
```bash
cd wind_model/api
python fastapi_forecast_weighted.py
```

The server will start on `http://localhost:8000`

### API Documentation:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "service": "forecast-weighted-prediction"
}
```

### 2. Simple Prediction
```http
GET /predict/simple?latitude=30.452&longitude=-91.188&hours_ahead=6
```

**Parameters:**
- `latitude` (float): Latitude coordinate (-90 to 90)
- `longitude` (float): Longitude coordinate (-180 to 180)
- `hours_ahead` (int): Hours ahead for prediction (1-24)

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

### 3. Full Prediction (POST)
```http
POST /predict
Content-Type: application/json

{
  "latitude": 30.452,
  "longitude": -91.188,
  "hours_ahead": 6,
  "forecast_weight": 0.3,
  "confidence_threshold": 0.8,
  "model_path": null
}
```

**Request Body:**
- `latitude` (float): Latitude coordinate
- `longitude` (float): Longitude coordinate
- `hours_ahead` (int): Hours ahead for prediction
- `forecast_weight` (float): Weight for forecast data (0-1)
- `confidence_threshold` (float): Minimum confidence threshold (0-1)
- `model_path` (string, optional): Path to custom model

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
  "base_validation": {
    "validation_metrics": {
      "overall_accuracy": 82.1,
      "wind_speed_accuracy": 85.3,
      "wind_direction_accuracy": 78.9
    }
  },
  "improvement": 5.4
}
```

## Usage Examples

### Python Requests
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

### cURL
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

## Testing

Run the test script to verify the API:
```bash
python test_fastapi_endpoint.py
```

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

Error responses include detailed error messages:
```json
{
  "detail": "Prediction failed: Could not load model"
}
```

## Configuration

### Environment Variables
- `MODEL_PATH`: Custom model path (optional)
- `DEFAULT_LATITUDE`: Default latitude (default: 30.452)
- `DEFAULT_LONGITUDE`: Default longitude (default: -91.188)

### Model Parameters
- `forecast_weight`: How much to weight forecast data (0-1)
- `confidence_threshold`: Minimum confidence for full forecast weight
- `hours_ahead`: Prediction horizon (1-24 hours)

## Troubleshooting

1. **Server won't start**: Check if port 8000 is available
2. **Import errors**: Ensure all dependencies are installed
3. **Model loading fails**: Check if model file exists in `models/` directory
4. **Forecast data unavailable**: Check internet connection and API keys

## Performance

- **Response time**: Typically 2-5 seconds
- **Concurrent requests**: Supports multiple simultaneous requests
- **Memory usage**: ~500MB for model and data
- **CPU usage**: Moderate during prediction

## Security

- Input validation on all parameters
- Error handling prevents information leakage
- No sensitive data in responses
- CORS enabled for web applications 