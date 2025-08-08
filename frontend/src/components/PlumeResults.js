import React from 'react';
import { Navigation, Clock, MapPin, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

const PlumeResults = ({ results }) => {
  if (!results) return null;

  const { plume_travel, travel_log } = results;

  const getStatusColor = (willReach) => {
    return willReach ? 'text-green-600' : 'text-red-600';
  };

  const getStatusIcon = (willReach) => {
    return willReach ? CheckCircle : XCircle;
  };

  const getMovementStatusColor = (status) => {
    if (status.includes('progress')) return 'text-green-600';
    if (status.includes('opposes')) return 'text-red-600';
    if (status.includes('limited')) return 'text-yellow-600';
    return 'text-gray-600';
  };

  return (
    <div className="space-y-4">
      {/* Summary Card */}
      <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-4 border border-green-200">
        <div className="flex items-center space-x-2 mb-3">
          <Navigation className="text-green-600" size={20} />
          <h4 className="text-lg font-semibold text-gray-900">Travel Summary</h4>
        </div>
        
        {plume_travel && (
          <div className="space-y-3">
            {/* Status */}
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Status:</span>
              <div className="flex items-center space-x-2">
                {(() => {
                  const StatusIcon = getStatusIcon(plume_travel.will_reach_destination);
                  return (
                    <>
                      <StatusIcon className={getStatusColor(plume_travel.will_reach_destination)} size={16} />
                      <span className={`font-semibold ${getStatusColor(plume_travel.will_reach_destination)}`}>
                        {plume_travel.will_reach_destination ? 'Will Reach Destination' : 'Will Not Reach Destination'}
                      </span>
                    </>
                  );
                })()}
              </div>
            </div>

            {/* Distance */}
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Total Distance:</span>
              <span className="font-semibold text-gray-900">
                {plume_travel.total_distance_km?.toFixed(2)} km
              </span>
            </div>

            {/* Bearing */}
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Bearing:</span>
              <span className="font-semibold text-gray-900">
                {plume_travel.bearing_degrees?.toFixed(1)}°
              </span>
            </div>

            {/* Arrival Time */}
            {plume_travel.arrival_time_hours && (
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">Arrival Time:</span>
                <span className="font-semibold text-green-600">
                  {plume_travel.arrival_time_hours} hours
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Travel Log */}
      {travel_log && travel_log.length > 0 && (
        <div>
          <div className="flex items-center space-x-2 mb-3">
            <Clock className="text-gray-600" size={16} />
            <h4 className="font-medium text-gray-900">Hour-by-Hour Travel</h4>
          </div>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {travel_log.map((log, index) => (
              <div key={index} className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="text-sm font-medium text-gray-700">
                      Hour {log.hour}
                    </span>
                    <span className="text-xs text-gray-500">
                      {log.time.toFixed(1)}h elapsed
                    </span>
                  </div>
                  <div className="text-xs text-gray-500">
                    {log.remaining_distance?.toFixed(3)} km remaining
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-gray-600">Wind:</span>
                    <div className="font-medium">
                      {log.wind_speed?.toFixed(1)} km/h @ {log.wind_direction?.toFixed(0)}°
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-600">Effective Speed:</span>
                    <div className={`font-medium ${
                      log.effective_speed > 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {log.effective_speed?.toFixed(1)} km/h
                    </div>
                  </div>
                </div>
                
                <div className="mt-2">
                  <span className="text-gray-600 text-xs">Movement:</span>
                  <div className={`text-xs font-medium ${getMovementStatusColor(log.movement_status)}`}>
                    {log.movement_status}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Key Metrics */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-blue-50 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-2">
            <MapPin className="text-blue-600" size={16} />
            <span className="text-sm font-medium text-gray-700">Source</span>
          </div>
          <div className="text-sm text-gray-900">
            {results.source_location?.latitude?.toFixed(3)}°N, {results.source_location?.longitude?.toFixed(3)}°W
          </div>
        </div>
        <div className="bg-orange-50 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-2">
            <AlertTriangle className="text-orange-600" size={16} />
            <span className="text-sm font-medium text-gray-700">Risk Zone</span>
          </div>
          <div className="text-sm text-gray-900">
            {results.risk_location?.latitude?.toFixed(3)}°N, {results.risk_location?.longitude?.toFixed(3)}°W
          </div>
        </div>
      </div>

      {/* Timestamp */}
      <div className="text-xs text-gray-500 text-center pt-2 border-t">
        Calculation completed at: {new Date(results.timestamp).toLocaleString()}
      </div>
    </div>
  );
};

export default PlumeResults; 