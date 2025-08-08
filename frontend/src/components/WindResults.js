import React from 'react';
import { Wind, Clock, TrendingUp, CheckCircle, AlertCircle } from 'lucide-react';

const WindResults = ({ results }) => {
  if (!results) return null;

  const { 
    weighted_prediction, 
    hourly_predictions, 
    forecast_confidence, 
    forecast_weight_used,
    validation 
  } = results;

  const getWindDirectionName = (degrees) => {
    const directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
    const index = Math.round(degrees / 22.5) % 16;
    return directions[index];
  };

  const getWindSpeedCategory = (speed) => {
    if (speed < 5) return { name: 'Light', color: 'text-green-600', bg: 'bg-green-100' };
    if (speed < 10) return { name: 'Moderate', color: 'text-yellow-600', bg: 'bg-yellow-100' };
    if (speed < 20) return { name: 'Strong', color: 'text-orange-600', bg: 'bg-orange-100' };
    return { name: 'Very Strong', color: 'text-red-600', bg: 'bg-red-100' };
  };

  return (
    <div className="space-y-4">
      {/* Summary Card */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4 border border-blue-200">
        <div className="flex items-center space-x-2 mb-3">
          <Wind className="text-blue-600" size={20} />
          <h4 className="text-lg font-semibold text-gray-900">Final Prediction</h4>
        </div>
        
        {weighted_prediction && (
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {weighted_prediction.wind_speed_mph?.toFixed(1)} mph
              </div>
              <div className="text-sm text-gray-600">Wind Speed</div>
              <div className={`inline-block px-2 py-1 rounded-full text-xs font-medium mt-1 ${
                getWindSpeedCategory(weighted_prediction.wind_speed_mph).bg
              } ${getWindSpeedCategory(weighted_prediction.wind_speed_mph).color}`}>
                {getWindSpeedCategory(weighted_prediction.wind_speed_mph).name}
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {weighted_prediction.wind_direction_degrees?.toFixed(0)}°
              </div>
              <div className="text-sm text-gray-600">Direction</div>
              <div className="text-xs text-gray-500 mt-1">
                {getWindDirectionName(weighted_prediction.wind_direction_degrees)}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Confidence & Weight */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-50 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-2">
            <CheckCircle className="text-green-600" size={16} />
            <span className="text-sm font-medium text-gray-700">Confidence</span>
          </div>
          <div className="text-lg font-bold text-green-600">
            {(forecast_confidence * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-gray-50 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-2">
            <TrendingUp className="text-blue-600" size={16} />
            <span className="text-sm font-medium text-gray-700">Forecast Weight</span>
          </div>
          <div className="text-lg font-bold text-blue-600">
            {(forecast_weight_used * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Hourly Predictions */}
      {hourly_predictions && hourly_predictions.length > 0 && (
        <div>
          <div className="flex items-center space-x-2 mb-3">
            <Clock className="text-gray-600" size={16} />
            <h4 className="font-medium text-gray-900">Hourly Predictions</h4>
          </div>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {hourly_predictions.map((prediction, index) => (
              <div key={index} className="flex items-center justify-between bg-gray-50 rounded-lg p-3">
                <div className="flex items-center space-x-3">
                  <div className="text-sm font-medium text-gray-700">
                    Hour {prediction.hour}
                  </div>
                  <div className="text-xs text-gray-500">
                    {prediction.wind_speed_mph.toFixed(1)} mph
                  </div>
                  <div className="text-xs text-gray-500">
                    {prediction.wind_direction_degrees.toFixed(0)}° {getWindDirectionName(prediction.wind_direction_degrees)}
                  </div>
                </div>
                <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                  getWindSpeedCategory(prediction.wind_speed_mph).bg
                } ${getWindSpeedCategory(prediction.wind_speed_mph).color}`}>
                  {getWindSpeedCategory(prediction.wind_speed_mph).name}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Validation Results */}
      {validation && (
        <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
          <div className="flex items-center space-x-2 mb-3">
            <AlertCircle className="text-yellow-600" size={16} />
            <h4 className="font-medium text-gray-900">Validation Results</h4>
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Wind Speed Accuracy:</span>
              <div className="font-semibold text-green-600">
                {validation.validation_metrics?.wind_speed_accuracy?.toFixed(1)}%
              </div>
            </div>
            <div>
              <span className="text-gray-600">Direction Accuracy:</span>
              <div className="font-semibold text-green-600">
                {validation.validation_metrics?.wind_direction_accuracy?.toFixed(1)}%
              </div>
            </div>
            <div>
              <span className="text-gray-600">Overall Accuracy:</span>
              <div className="font-semibold text-green-600">
                {validation.validation_metrics?.overall_accuracy?.toFixed(1)}%
              </div>
            </div>
            <div>
              <span className="text-gray-600">Speed Error:</span>
              <div className="font-semibold text-red-600">
                {validation.errors?.wind_speed_error?.toFixed(2)} mph
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Timestamp */}
      <div className="text-xs text-gray-500 text-center pt-2 border-t">
        Prediction generated at: {new Date(results.timestamp).toLocaleString()}
      </div>
    </div>
  );
};

export default WindResults; 