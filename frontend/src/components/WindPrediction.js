import React, { useState } from 'react';
import { Search, Wind, Clock, MapPin, AlertCircle } from 'lucide-react';
import WindMap from './WindMap';
import WindResults from './WindResults';
import { predictWind } from '../services/api';

const WindPrediction = () => {
  const [formData, setFormData] = useState({
    latitude: 30.452,
    longitude: -91.188,
    hours_ahead: 6,
    forecast_weight: 0.5,
    confidence_threshold: 0.8
  });
  
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'latitude' || name === 'longitude' || name === 'forecast_weight' || name === 'confidence_threshold' 
        ? parseFloat(value) 
        : parseInt(value)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const data = await predictWind(formData);
      setResults(data);
    } catch (err) {
      setError(err.message || 'Failed to predict wind conditions');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Form Section */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Wind className="text-primary-500" size={20} />
          <h2 className="text-xl font-semibold text-gray-900">Wind Prediction</h2>
        </div>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Latitude */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Latitude
              </label>
              <input
                type="number"
                name="latitude"
                value={formData.latitude}
                onChange={handleInputChange}
                step="0.001"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="30.452"
              />
            </div>

            {/* Longitude */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Longitude
              </label>
              <input
                type="number"
                name="longitude"
                value={formData.longitude}
                onChange={handleInputChange}
                step="0.001"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="-91.188"
              />
            </div>

            {/* Hours Ahead */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Hours Ahead
              </label>
              <input
                type="number"
                name="hours_ahead"
                value={formData.hours_ahead}
                onChange={handleInputChange}
                min="1"
                max="24"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>

            {/* Forecast Weight */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Forecast Weight (0-1)
              </label>
              <input
                type="number"
                name="forecast_weight"
                value={formData.forecast_weight}
                onChange={handleInputChange}
                min="0"
                max="1"
                step="0.1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>

            {/* Confidence Threshold */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Confidence Threshold (0-1)
              </label>
              <input
                type="number"
                name="confidence_threshold"
                value={formData.confidence_threshold}
                onChange={handleInputChange}
                min="0"
                max="1"
                step="0.1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full md:w-auto px-6 py-3 bg-primary-500 text-white rounded-md hover:bg-primary-600 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            {loading ? (
              <>
                <div className="loading-spinner"></div>
                <span>Predicting...</span>
              </>
            ) : (
              <>
                <Search size={20} />
                <span>Predict Wind Conditions</span>
              </>
            )}
          </button>
        </form>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertCircle className="text-red-500" size={20} />
            <span className="text-red-700 font-medium">Error: {error}</span>
          </div>
        </div>
      )}

      {/* Map Section - Always show */}
      <div className="bg-white rounded-lg shadow-sm border p-4">
        <div className="flex items-center space-x-2 mb-4">
          <MapPin className="text-primary-500" size={20} />
          <h3 className="text-lg font-semibold text-gray-900">Wind Map</h3>
        </div>
        <div className="h-96">
          <WindMap 
            latitude={formData.latitude}
            longitude={formData.longitude}
            windData={results}
          />
        </div>
      </div>

      {/* Results Section - Only show when results are available */}
      {results && (
        <div className="bg-white rounded-lg shadow-sm border p-4">
          <div className="flex items-center space-x-2 mb-4">
            <Clock className="text-primary-500" size={20} />
            <h3 className="text-lg font-semibold text-gray-900">Prediction Results</h3>
          </div>
          <WindResults results={results} />
        </div>
      )}
    </div>
  );
};

export default WindPrediction; 