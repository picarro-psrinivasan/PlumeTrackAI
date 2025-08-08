import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import { MapPin, Navigation, AlertTriangle } from 'lucide-react';

// Function to calculate destination point given starting point, bearing, and distance
const calculateDestinationPoint = (lat, lng, bearing, distanceMeters) => {
  const earthRadius = 6371000; // Earth's radius in meters
  const latRad = lat * Math.PI / 180;
  const lngRad = lng * Math.PI / 180;
  const bearingRad = bearing * Math.PI / 180;
  
  const angularDistance = distanceMeters / earthRadius;
  
  const newLatRad = Math.asin(
    Math.sin(latRad) * Math.cos(angularDistance) +
    Math.cos(latRad) * Math.sin(angularDistance) * Math.cos(bearingRad)
  );
  
  const newLngRad = lngRad + Math.atan2(
    Math.sin(bearingRad) * Math.sin(angularDistance) * Math.cos(latRad),
    Math.cos(angularDistance) - Math.sin(latRad) * Math.sin(newLatRad)
  );
  
  return {
    lat: newLatRad * 180 / Math.PI,
    lng: newLngRad * 180 / Math.PI
  };
};

// Fix for default markers in Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const PlumeMap = ({ sourceLat, sourceLon, riskLat, riskLon, plumeData }) => {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);

  useEffect(() => {
    if (!mapRef.current) {
      console.log('PlumeMap: mapRef.current is null');
      return;
    }

    // Calculate center point between source and risk
    const centerLat = (sourceLat + riskLat) / 2;
    const centerLon = (sourceLon + riskLon) / 2;

    console.log('PlumeMap: Initializing map with center:', centerLat, centerLon);
    console.log('PlumeMap: plumeData structure:', plumeData);
    console.log('PlumeMap: GeoJSON data:', plumeData?.geojson);
    
    // Initialize map
    const map = L.map(mapRef.current).setView([centerLat, centerLon], 13);
    mapInstanceRef.current = map;

    // Add tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    // Add source marker
    const sourceIcon = L.divIcon({
      className: 'source-marker',
      html: `
        <div style="
          background-color: #ef4444;
          color: white;
          border-radius: 50%;
          width: 30px;
          height: 30px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
          border: 3px solid white;
          box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        ">
          S
        </div>
      `,
      iconSize: [30, 30],
      iconAnchor: [15, 15]
    });

    const sourceMarker = L.marker([sourceLat, sourceLon], { icon: sourceIcon })
      .addTo(map)
      .bindPopup(`
        <div class="text-center">
          <strong>Source Location</strong><br>
          ${sourceLat.toFixed(3)}°N, ${sourceLon.toFixed(3)}°W
        </div>
      `);

    // Add risk marker
    const riskIcon = L.divIcon({
      className: 'risk-marker',
      html: `
        <div style="
          background-color: #f59e0b;
          color: white;
          border-radius: 50%;
          width: 30px;
          height: 30px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
          border: 3px solid white;
          box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        ">
          R
        </div>
      `,
      iconSize: [30, 30],
      iconAnchor: [15, 15]
    });

    const riskMarker = L.marker([riskLat, riskLon], { icon: riskIcon })
      .addTo(map)
      .bindPopup(`
        <div class="text-center">
          <strong>Risk Location</strong><br>
          ${riskLat.toFixed(3)}°N, ${riskLon.toFixed(3)}°W
        </div>
      `);

    // Add GeoJSON plume path if available
    if (plumeData && plumeData.geojson) {
      console.log('PlumeMap: Creating GeoJSON layer with data:', plumeData.geojson);
      
      // Comment out the GeoJSON plume path - using trajectory instead
      // const plumeLayer = L.geoJSON(plumeData.geojson, {
      //   style: {
      //     color: '#22c55e',
      //     weight: 4,
      //     opacity: 0.8
      //   },
      //   onEachFeature: (feature, layer) => {
      //     if (feature.properties) {
      //       layer.bindPopup(`
      //         <div class="text-center">
      //           <strong>Plume Travel Path</strong><br>
      //           Distance: ${plumeData.plume_travel?.total_distance_km?.toFixed(2) || 'N/A'} km<br>
      //           Bearing: ${plumeData.plume_travel?.bearing_degrees?.toFixed(1) || 'N/A'}°
      //         </div>
      //       `);
      //     }
      //   }
      // }).addTo(map);
      
      console.log('PlumeMap: GeoJSON plume path hidden - using trajectory path instead');
      
      console.log('PlumeMap: GeoJSON layer created and added to map');
      
      // Add wind direction arrows along the plume path
      // Check for wind data in different possible formats
      let windPredictions = null;
      if (plumeData.wind_data && plumeData.wind_data.hourly_predictions) {
        windPredictions = plumeData.wind_data.hourly_predictions;
      } else if (plumeData.hourly_wind_predictions) {
        windPredictions = plumeData.hourly_wind_predictions;
      } else if (plumeData.wind_data && plumeData.wind_data.forecast_data) {
        // Create hourly predictions from forecast data
        const forecastData = plumeData.wind_data.forecast_data;
        const windSpeedForecast = forecastData.wind_speed_forecast || [];
        const windDirectionForecast = forecastData.wind_direction_forecast || [];
        windPredictions = [];
        
        for (let hour = 0; hour < Math.min(windSpeedForecast.length, 6); hour++) {
          windPredictions.push({
            hour: hour + 1,
            wind_speed_mph: windSpeedForecast[hour] || 0,
            wind_direction_degrees: windDirectionForecast[hour] || 0
          });
        }
      }

      if (windPredictions && windPredictions.length > 0) {
        // Debug: Log wind direction changes
        console.log('PlumeMap: Wind direction changes:');
        windPredictions.forEach((prediction, index) => {
          if (index > 0) {
            const prevDirection = windPredictions[index - 1].wind_direction_degrees;
            const currDirection = prediction.wind_direction_degrees;
            const change = Math.abs(currDirection - prevDirection);
            console.log(`Hour ${index} -> ${index + 1}: ${prevDirection}° -> ${currDirection}° (change: ${change}°)`);
          }
        });
        
        // Note: No longer using GeoJSON path coordinates - using trajectory instead
        console.log('PlumeMap: Using trajectory path instead of GeoJSON path');



        // Add plume trajectory path visualization (similar to wind trajectory)
        const plumeTrajectoryPoints = [];
        let currentLat = sourceLat;
        let currentLon = sourceLon;
        
        // Always start from source
        plumeTrajectoryPoints.push([currentLat, currentLon]);
        
        // Debug: Log the wind predictions being used for trajectory
        console.log('PlumeMap: Wind predictions for trajectory:', windPredictions);
        console.log('PlumeMap: Full plumeData structure:', plumeData);
        console.log('PlumeMap: wind_data structure:', plumeData.wind_data);
        
        // Use model's base prediction wind directions for sharp trajectory turns
        console.log('PlumeMap: Using model base predictions for sharp trajectory turns');
        
        // Use weighted predictions to show hourly variations and deviations
        let modelWindDirections = null;
        let modelWindSpeeds = null;
        
        if (windPredictions && windPredictions.length > 0) {
          // Use weighted predictions for hourly variations
          modelWindDirections = windPredictions.map(p => p.wind_direction_degrees);
          modelWindSpeeds = windPredictions.map(p => p.wind_speed_mph);
          console.log('PlumeMap: Using weighted predictions for hourly variations');
        } else if (plumeData.wind_data && plumeData.wind_data.forecast_data) {
          // Fallback to raw forecast data
          const forecastData = plumeData.wind_data.forecast_data;
          modelWindDirections = forecastData.wind_direction_forecast || [];
          modelWindSpeeds = forecastData.wind_speed_forecast || [];
          console.log('PlumeMap: Using raw forecast data as fallback');
        } else {
          // Last fallback to base prediction
          const basePrediction = plumeData.wind_data?.base_prediction;
          if (basePrediction) {
            const baseDirection = basePrediction.wind_direction_degrees;
            const baseSpeed = basePrediction.wind_speed_mph;
            modelWindDirections = Array(6).fill(baseDirection);
            modelWindSpeeds = Array(6).fill(baseSpeed);
            console.log('PlumeMap: Using base prediction as last fallback');
          }
        }
        
        console.log('PlumeMap: Model wind directions:', modelWindDirections);
        console.log('PlumeMap: Model wind speeds:', modelWindSpeeds);
        
        // Use actual wind speeds (no amplification)
        const actualSpeeds = modelWindSpeeds;
        
        // Use model predictions for trajectory
        modelWindDirections.forEach((direction, index) => {
          const speed = actualSpeeds[index];
          console.log(`Model Hour ${index + 1}: Wind direction ${direction}°, actual speed ${speed} mph`);
          
          // Convert wind speed from mph to meters per hour
          const distanceMeters = speed * 1609.34; // mph to m/h
          
          // Calculate new position using proper geographic calculations
          const newPosition = calculateDestinationPoint(currentLat, currentLon, direction, distanceMeters);
          
          console.log(`  Moving from (${currentLat.toFixed(6)}, ${currentLon.toFixed(6)}) to (${newPosition.lat.toFixed(6)}, ${newPosition.lng.toFixed(6)})`);
          
          plumeTrajectoryPoints.push([newPosition.lat, newPosition.lng]);
          currentLat = newPosition.lat;
          currentLon = newPosition.lng;
        });
        
        // Debug: Log trajectory points and calculations
        console.log('PlumeMap: Trajectory points:', plumeTrajectoryPoints);
        console.log('PlumeMap: Detailed trajectory calculation:');
        plumeTrajectoryPoints.forEach((point, index) => {
          if (index > 0) {
            const prevPoint = plumeTrajectoryPoints[index - 1];
            const bearing = Math.atan2(
              point[1] - prevPoint[1], 
              point[0] - prevPoint[0]
            ) * 180 / Math.PI;
            const normalizedBearing = (bearing + 360) % 360;
            console.log(`Segment ${index}: ${prevPoint[0].toFixed(6)}, ${prevPoint[1].toFixed(6)} -> ${point[0].toFixed(6)}, ${point[1].toFixed(6)} (bearing: ${normalizedBearing.toFixed(1)}°)`);
          }
        });
        

        
        // Draw the plume trajectory path
        if (plumeTrajectoryPoints.length > 1) {
          const plumeTrajectory = L.polyline(plumeTrajectoryPoints, {
            color: '#16a34a',
            weight: 4,
            opacity: 0.6,
            dashArray: '8, 4'
          }).addTo(map);
          
          // Add wind direction arrows hourly along the plume trajectory
          const numArrows = Math.min(windPredictions.length, 6); // Max 6 arrows
          
          // Calculate positions along the trajectory for arrow placement
          const trajectoryPositions = [];
          if (plumeTrajectoryPoints.length > 1) {
            // Place arrows evenly along the trajectory
            for (let i = 0; i < numArrows; i++) {
              const index = Math.floor((i / (numArrows - 1)) * (plumeTrajectoryPoints.length - 1));
              trajectoryPositions.push(plumeTrajectoryPoints[index]);
            }
          } else {
            // If no trajectory, place arrows in a circle around source
            const radius = 0.01; // Small radius around source
            for (let i = 0; i < numArrows; i++) {
              const angle = (i / numArrows) * 2 * Math.PI;
              const lat = sourceLat + radius * Math.cos(angle);
              const lng = sourceLon + radius * Math.sin(angle);
              trajectoryPositions.push([lat, lng]);
            }
          }
          
          for (let i = 0; i < numArrows; i++) {
            const prediction = windPredictions[i];
            const position = trajectoryPositions[i];
            
            if (position && prediction) {
              const { wind_speed_mph, wind_direction_degrees } = prediction;
              
              // Use model wind direction if available, otherwise fallback to weighted prediction
              let windDirection;
              if (modelWindDirections && modelWindDirections[i] !== undefined) {
                windDirection = modelWindDirections[i];
              } else {
                windDirection = wind_direction_degrees;
              }
              
              console.log(`Arrow ${i + 1}: Model wind direction ${windDirection}° at position (${position[0].toFixed(6)}, ${position[1].toFixed(6)})`);
              
              // Create wind direction arrow
              const windIcon = L.divIcon({
                className: 'plume-travel-arrow',
                html: `
                  <div style="
                    transform: rotate(${windDirection}deg);
                    width: 35px;
                    height: 35px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                  ">
                    <svg width="35" height="35" viewBox="0 0 35 35">
                      <defs>
                        <filter id="plume-wind-shadow-${i}" x="-50%" y="-50%" width="200%" height="200%">
                          <feDropShadow dx="1" dy="1" stdDeviation="1" flood-color="rgba(0,0,0,0.3)"/>
                        </filter>
                      </defs>
                      <path 
                        d="M17.5 2 L17.5 33 M13 12 L17.5 2 L22 12" 
                        stroke="#22c55e" 
                        stroke-width="3" 
                        fill="none" 
                        stroke-linecap="round" 
                        stroke-linejoin="round"
                        filter="url(#plume-wind-shadow-${i})"
                      />
                      <circle cx="17.5" cy="17.5" r="15" stroke="#22c55e" stroke-width="2" fill="rgba(34, 197, 94, 0.15)"/>
                      <text x="17.5" y="20" text-anchor="middle" font-size="10" font-weight="bold" fill="#22c55e">${i + 1}</text>
                    </svg>
                  </div>
                `,
                iconSize: [35, 35],
                iconAnchor: [17.5, 17.5]
              });

              L.marker([position[0], position[1]], { icon: windIcon })
                .addTo(map)
                .bindPopup(`
                  <div class="text-center">
                    <strong>Hour ${prediction.hour}</strong><br>
                    <strong>Wind Direction: ${windDirection.toFixed(1)}°</strong><br>
                    Wind Speed: ${wind_speed_mph} mph<br>
                    <em>This arrow shows the wind direction for this hour</em>
                  </div>
                `);
            }
          }
          
          // Add small direction markers along the trajectory
          plumeTrajectoryPoints.forEach((point, index) => {
            if (index < windPredictions.length) {
              const prediction = windPredictions[index];
              
              // Use model wind direction if available, otherwise fallback to weighted prediction
              let windDirection;
              if (modelWindDirections && modelWindDirections[index] !== undefined) {
                windDirection = modelWindDirections[index];
              } else {
                windDirection = prediction.wind_direction_degrees;
              }
              
              const arrowIcon = L.divIcon({
                className: 'trajectory-arrow',
                html: `
                  <div style="
                    transform: rotate(${windDirection}deg);
                    width: 15px;
                    height: 15px;
                  ">
                    <svg width="15" height="15" viewBox="0 0 15 15">
                      <path 
                        d="M7.5 1 L7.5 14 M5 6 L7.5 1 L10 6" 
                        stroke="#16a34a" 
                        stroke-width="1.5" 
                        fill="none" 
                        stroke-linecap="round" 
                        stroke-linejoin="round"
                      />
                    </svg>
                  </div>
                `,
                iconSize: [15, 15],
                iconAnchor: [7.5, 7.5]
              });
              
              L.marker(point, { icon: arrowIcon })
                .addTo(map)
                .bindPopup(`
                  <div class="text-center">
                    <strong>Hour ${prediction.hour}</strong><br>
                    Wind Speed: ${prediction.wind_speed_mph} mph<br>
                    Wind Direction: ${prediction.wind_direction_degrees}°
                  </div>
                `);
            }
          });
        }
      }
    } else {
      console.log('PlumeMap: No GeoJSON data available');
    }

    // Add direct line between source and risk
    const directLine = L.polyline([[sourceLat, sourceLon], [riskLat, riskLon]], {
      color: '#6b7280',
      weight: 2,
      opacity: 0.6,
      dashArray: '5, 5'
    }).addTo(map);

    // Add legend
    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = function() {
      const div = L.DomUtil.create('div', 'info legend');
      div.style.backgroundColor = 'white';
      div.style.padding = '10px';
      div.style.borderRadius = '5px';
      div.style.border = '2px solid #ccc';
      div.style.fontSize = '12px';
      
      const willReach = plumeData?.plume_travel?.will_reach_destination;
      const arrivalTime = plumeData?.plume_travel?.arrival_time_hours;
      
      div.innerHTML = `
        <h4 style="margin: 0 0 10px 0;">Plume Travel</h4>
        <div style="margin-bottom: 8px;">
          <span style="display: inline-block; width: 12px; height: 12px; background-color: #ef4444; border-radius: 50%; margin-right: 5px;"></span>
          Source
        </div>
        <div style="margin-bottom: 8px;">
          <span style="display: inline-block; width: 12px; height: 12px; background-color: #f59e0b; border-radius: 50%; margin-right: 5px;"></span>
          Risk Zone
        </div>
        <div style="margin-bottom: 8px;">
          <span style="display: inline-block; width: 12px; height: 3px; background-color: #16a34a; margin-right: 5px; border-top: 2px dashed #16a34a;"></span>
          Plume Path (Wind-Driven)
        </div>
        <div style="margin-bottom: 8px;">
          <span style="display: inline-block; width: 12px; height: 3px; background-color: #16a34a; margin-right: 5px; border-top: 2px dashed #16a34a;"></span>
          Wind-Driven Trajectory
        </div>
        <div style="margin-bottom: 8px;">
          <span style="display: inline-block; width: 12px; height: 2px; background-color: #6b7280; margin-right: 5px; border-top: 1px dashed #6b7280;"></span>
          Direct Line
        </div>
        <div style="margin-bottom: 8px;">
          <svg width="16" height="16" viewBox="0 0 16 16" style="display: inline-block; margin-right: 5px;">
            <path d="M8 2 L8 14 M6 6 L8 2 L10 6" stroke="#22c55e" stroke-width="2" fill="none"/>
            <circle cx="8" cy="8" r="6" stroke="#22c55e" stroke-width="1" fill="rgba(34, 197, 94, 0.15)"/>
            <text x="8" y="10" text-anchor="middle" font-size="6" font-weight="bold" fill="#22c55e">1</text>
          </svg>
          Wind Direction (Hourly)
        </div>
        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee;">
          <strong>Status:</strong> ${willReach ? 'Will reach' : 'Will not reach'}<br>
          ${arrivalTime ? `Arrival: ${arrivalTime} hours` : ''}
        </div>
      `;
      return div;
    };
    legend.addTo(map);

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
      }
    };
  }, [sourceLat, sourceLon, riskLat, riskLon, plumeData]);

  return (
    <div className="relative map-container">
      <div ref={mapRef} style={{ height: '100%', width: '100%' }} />
      
      {/* Info overlay when no plume data */}
      {!plumeData && (
        <div className="absolute top-4 left-4 bg-white bg-opacity-90 rounded-lg p-3 shadow-md">
          <div className="text-center">
            <Navigation className="mx-auto mb-2 text-gray-400" size={20} />
            <p className="text-gray-600 text-sm">Submit form to see plume travel</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default PlumeMap; 