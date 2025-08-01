# Wind Model Package Structure

## 📁 Organized Project Structure

The wind model package has been reorganized for better maintainability and separation of concerns:

```
wind_model/
├── 📄 Core Files
│   ├── __init__.py              # Package initialization with imports
│   ├── README.md                # Comprehensive documentation
│   ├── main.py                  # Interactive menu entry point
│   ├── train_model.py           # Training script
│   └── predict.py               # Basic prediction script
│
├── 🧠 src/                      # Core Model Implementation
│   ├── __init__.py
│   ├── lstm_model.py           # LSTM neural network
│   ├── predict_wind.py         # Wind prediction functionality
│   └── load_data.py            # Data loading utilities
│
├── 📦 api/                      # API and External Integrations
│   ├── __init__.py
│   ├── api_predict.py          # Simple prediction API
│   └── ops_wind_data_api.py    # Open-Meteo API integration
│
├── ✅ validation/               # Validation and Testing
│   ├── __init__.py
│   ├── predict_with_validation.py    # Comprehensive prediction with validation
│   └── test_forecast_validation.py  # Forecast validation tests
│
├── 🛠️ utils/                   # Utility Functions
│   └── __init__.py
│
└── 📁 models/                   # Trained Model Files
    └── wind_lstm_model.pth     # Trained LSTM model
```

## 🔧 Package Organization

### 📦 `api/` - API and External Integrations
**Purpose:** Handle external API interactions and provide simple prediction interfaces

**Files:**
- **`api_predict.py`**: Simple prediction API with validation
  - `predict_wind_with_validation()` - Main prediction function
  - `get_prediction_confidence()` - Confidence assessment
  - `format_prediction_results()` - Formatted output

- **`ops_wind_data_api.py`**: Open-Meteo API integration
  - `extract_forecast_wind_data()` - Extract forecast data
  - `validate_prediction_with_forecast()` - Validate predictions
  - `load_wind_data()` - Load wind data from API

### ✅ `validation/` - Validation and Testing
**Purpose:** Comprehensive validation workflows and testing functionality

**Files:**
- **`predict_with_validation.py`**: Comprehensive prediction with validation
  - `predict_and_validate()` - Full prediction workflow
  - `predict_with_confidence_threshold()` - Threshold-based prediction

- **`test_forecast_validation.py`**: Forecast validation test suite
  - `test_forecast_extraction()` - Test forecast data extraction
  - `test_prediction_validation()` - Test prediction validation
  - `test_real_prediction_validation()` - Test real predictions

### 🧠 `src/` - Core Model Implementation
**Purpose:** Core LSTM model and prediction functionality

**Files:**
- **`lstm_model.py`**: LSTM neural network implementation
  - `WindLSTM` class - LSTM model architecture
  - `train_model()` - Training function
  - `evaluate_model()` - Evaluation function

- **`predict_wind.py`**: Wind prediction functionality
  - `predict_wind_6hours_ahead()` - Main prediction function
  - `load_trained_model()` - Model loading
  - `prepare_input_sequence()` - Input preparation

- **`load_data.py`**: Data loading and preprocessing
  - `load_and_preprocess_data()` - Data loading
  - `extract_wind_data()` - Wind data extraction
  - `create_sequences()` - Sequence creation

### 🛠️ `utils/` - Utility Functions
**Purpose:** Common helper functions and utilities

**Files:**
- **`__init__.py`**: Package initialization (ready for future utilities)

## 🚀 Usage Examples

### Interactive Menu
```bash
cd wind_model
python main.py
```

### Direct API Usage
```python
from wind_model.api import predict_wind_with_validation

# Make prediction with automatic validation
results = predict_wind_with_validation()
```

### Comprehensive Validation
```python
from wind_model.validation import predict_and_validate

# Get detailed prediction with full validation
results = predict_and_validate()
```

### Core Model Usage
```python
from wind_model.src.lstm_model import WindLSTM
from wind_model.src.predict_wind import predict_wind_6hours_ahead

# Use core model functionality
```

## 📊 Import Structure

### Main Package Imports
```python
from wind_model import (
    predict_wind_with_validation,
    predict_and_validate,
    WindLSTM,
    predict_wind_6hours_ahead
)
```

### API Package Imports
```python
from wind_model.api import (
    predict_wind_with_validation,
    get_prediction_confidence,
    extract_forecast_wind_data,
    validate_prediction_with_forecast
)
```

### Validation Package Imports
```python
from wind_model.validation import (
    predict_and_validate,
    predict_with_confidence_threshold
)
```

## 🎯 Benefits of This Organization

1. **✅ Separation of Concerns**: Each directory has a specific purpose
2. **✅ Easy Navigation**: Clear file organization and naming
3. **✅ Modular Design**: Independent packages that can be imported separately
4. **✅ Scalable Structure**: Easy to add new functionality
5. **✅ Clear Documentation**: Each package has its own purpose and documentation
6. **✅ Interactive Interface**: Main menu for easy access to all functionality

## 🔧 Maintenance

### Adding New API Functions
- Add to `api/` directory
- Update `api/__init__.py` with new imports
- Update main package `__init__.py` if needed

### Adding New Validation Tests
- Add to `validation/` directory
- Update `validation/__init__.py` with new imports
- Update main package `__init__.py` if needed

### Adding New Model Features
- Add to `src/` directory
- Update `src/__init__.py` with new imports
- Update main package `__init__.py` if needed

This organized structure makes the codebase more maintainable, scalable, and user-friendly! 🚀 