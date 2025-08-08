import React, { useState } from 'react';
import { Search, MapPin, Clock, AlertCircle, Navigation } from 'lucide-react';
import PlumeMap from './PlumeMap';
import PlumeResults from './PlumeResults';
import { predictPlume } from '../services/api';

const PlumeTravel = () => {
  const [formData, setFormData] = useState({
    source_latitude: 30.452,
    source_longitude: -91.188,
    risk_latitude: 30.458,
    risk_longitude: -91.182,
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
      [name]: name.includes('latitude') || name.includes('longitude') || name === 'forecast_weight' || name === 'confidence_threshold' 
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
      const data = await predictPlume(formData);
      setResults(data);
    } catch (err) {
      setError(err.message || 'Failed to calculate plume travel');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Form Section */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Navigation className="text-plume-500" size={20} />
          <h2 className="text-xl font-semibold text-gray-900">Plume Travel Prediction</h2>
        </div>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Source Location */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-3">Source Location</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Source Latitude
                </label>
                <input
                  type="number"
                  name="source_latitude"
                  value={formData.source_latitude}
                  onChange={handleInputChange}
                  step="0.001"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-plume-500 focus:border-transparent"
                  placeholder="30.452"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Source Longitude
                </label>
                <input
                  type="number"
                  name="source_longitude"
                  value={formData.source_longitude}
                  onChange={handleInputChange}
                  step="0.001"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-plume-500 focus:border-transparent"
                  placeholder="-91.188"
                />
              </div>
            </div>
          </div>

          {/* Risk Location */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-3">Risk Location</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Risk Latitude
                </label>
                <input
                  type="number"
                  name="risk_latitude"
                  value={formData.risk_latitude}
                  onChange={handleInputChange}
                  step="0.001"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-plume-500 focus:border-transparent"
                  placeholder="30.458"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Risk Longitude
                </label>
                <input
                  type="number"
                  name="risk_longitude"
                  value={formData.risk_longitude}
                  onChange={handleInputChange}
                  step="0.001"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-plume-500 focus:border-transparent"
                  placeholder="-91.182"
                />
              </div>
            </div>
          </div>

          {/* Prediction Parameters */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-3">Prediction Parameters</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-plume-500 focus:border-transparent"
                />
              </div>
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
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-plume-500 focus:border-transparent"
                />
              </div>
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
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-plume-500 focus:border-transparent"
                />
              </div>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full md:w-auto px-6 py-3 bg-plume-500 text-white rounded-md hover:bg-plume-600 focus:outline-none focus:ring-2 focus:ring-plume-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            {loading ? (
              <>
                <div className="loading-spinner"></div>
                <span>Calculating...</span>
              </>
            ) : (
              <>
                <Search size={20} />
                <span>Calculate Plume Travel</span>
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
          <MapPin className="text-plume-500" size={20} />
          <h3 className="text-lg font-semibold text-gray-900">Plume Travel Map</h3>
        </div>
        <div className="h-96">
          <PlumeMap 
            sourceLat={formData.source_latitude}
            sourceLon={formData.source_longitude}
            riskLat={formData.risk_latitude}
            riskLon={formData.risk_longitude}
            plumeData={results}
          />
        </div>
      </div>

      {/* Results Section - Only show when results are available */}
      {results && (
        <div className="bg-white rounded-lg shadow-sm border p-4">
          <div className="flex items-center space-x-2 mb-4">
            <Clock className="text-plume-500" size={20} />
            <h3 className="text-lg font-semibold text-gray-900">Travel Results</h3>
          </div>
          <PlumeResults results={results} />
        </div>
      )}
    </div>
  );
};

export default PlumeTravel; 