import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import { Wind, Navigation } from 'lucide-react';

// Fix for default markers in Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const WindMap = ({ latitude, longitude, windData }) => {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);

  useEffect(() => {
    if (!mapRef.current) {
      console.log('WindMap: mapRef.current is null');
      return;
    }

    console.log('WindMap: Initializing map with coordinates:', latitude, longitude);
    console.log('WindMap: Full windData structure:', windData);
    
    // Initialize map
    const map = L.map(mapRef.current).setView([latitude, longitude], 12);
    mapInstanceRef.current = map;

    // Add tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    // Add location marker
    const locationMarker = L.marker([latitude, longitude])
      .addTo(map)
      .bindPopup(`
        <div class="text-center">
          <strong>Prediction Location</strong><br>
          ${latitude.toFixed(3)}°N, ${longitude.toFixed(3)}°W
        </div>
      `);

    // Add wind direction indicators if we have hourly predictions
    if (windData && windData.hourly_predictions) {
      // Use weighted predictions to show hourly variations and deviations
      let windDirections = [];
      let windSpeeds = [];
      
      if (windData.hourly_predictions && windData.hourly_predictions.length > 0) {
        // Use weighted predictions for hourly variations
        windDirections = windData.hourly_predictions.map(p => p.wind_direction_degrees);
        windSpeeds = windData.hourly_predictions.map(p => p.wind_speed_mph);
        console.log('WindMap: Using weighted predictions for hourly variations');
      } else if (windData.base_prediction) {
        // Fallback to base prediction if no hourly predictions
        const baseDirection = windData.base_prediction.wind_direction_degrees;
        const baseSpeed = windData.base_prediction.wind_speed_mph;
        windDirections = Array(6).fill(baseDirection);
        windSpeeds = Array(6).fill(baseSpeed);
        console.log('WindMap: Using base prediction as fallback');
      }
      
      windData.hourly_predictions.forEach((prediction, index) => {
        const wind_direction_degrees = windDirections[index];
        const wind_speed_mph = windSpeeds[index];
        
        // Create wind direction arrow with SVG for better visibility
        const windIcon = L.divIcon({
          className: 'wind-arrow',
          html: `
            <div style="
              transform: rotate(${wind_direction_degrees}deg);
              width: 40px;
              height: 40px;
              display: flex;
              align-items: center;
              justify-content: center;
            ">
              <svg width="40" height="40" viewBox="0 0 40 40">
                <defs>
                  <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="2" dy="2" stdDeviation="2" flood-color="rgba(0,0,0,0.3)"/>
                  </filter>
                </defs>
                <path 
                  d="M20 5 L20 35 M15 15 L20 5 L25 15" 
                  stroke="#3b82f6" 
                  stroke-width="3" 
                  fill="none" 
                  stroke-linecap="round" 
                  stroke-linejoin="round"
                  filter="url(#shadow)"
                />
                <circle cx="20" cy="20" r="18" stroke="#3b82f6" stroke-width="2" fill="rgba(59, 130, 246, 0.1)"/>
              </svg>
            </div>
          `,
          iconSize: [40, 40],
          iconAnchor: [20, 20]
        });

        // Position the wind arrow in a circle around the main location
        const radius = 0.02; // Larger radius for better visibility
        const angle = (index * 2 * Math.PI) / windData.hourly_predictions.length;
        const windLat = latitude + radius * Math.cos(angle);
        const windLon = longitude + radius * Math.sin(angle);

        const windMarker = L.marker([windLat, windLon], { icon: windIcon })
          .addTo(map)
          .bindPopup(`
            <div class="text-center">
              <strong>Hour ${prediction.hour}</strong><br>
              Speed: ${wind_speed_mph} mph<br>
              Direction: ${wind_direction_degrees}°
            </div>
          `);
      });

      // Add wind path visualization
      const pathPoints = [];
      let currentLat = latitude;
      let currentLon = longitude;
      
      windData.hourly_predictions.forEach((prediction, index) => {
        const wind_direction_degrees = windDirections[index];
        const wind_speed_mph = windSpeeds[index];
        
        // Convert wind speed from mph to degrees (approximate conversion)
        // 1 mph ≈ 0.001 degrees at this latitude
        const distance = wind_speed_mph * 0.001;
        
        // Calculate new position based on wind direction and speed
        const directionRad = (wind_direction_degrees * Math.PI) / 180;
        const newLat = currentLat + distance * Math.cos(directionRad);
        const newLon = currentLon + distance * Math.sin(directionRad);
        
        pathPoints.push([currentLat, currentLon]);
        currentLat = newLat;
        currentLon = newLon;
      });
      
      // Add the final point
      pathPoints.push([currentLat, currentLon]);
      
      // Draw the wind path
      if (pathPoints.length > 1) {
        const windPath = L.polyline(pathPoints, {
          color: '#3b82f6',
          weight: 3,
          opacity: 0.7,
          dashArray: '10, 5'
        }).addTo(map);
        
        // Add arrow markers along the path
        pathPoints.forEach((point, index) => {
          if (index < windData.hourly_predictions.length) {
            const prediction = windData.hourly_predictions[index];
            const wind_direction_degrees = windDirections[index];
            const wind_speed_mph = windSpeeds[index];
            
            const arrowIcon = L.divIcon({
              className: 'path-arrow',
              html: `
                <div style="
                  transform: rotate(${wind_direction_degrees}deg);
                  width: 20px;
                  height: 20px;
                ">
                  <svg width="20" height="20" viewBox="0 0 20 20">
                    <path 
                      d="M10 2 L10 18 M7 8 L10 2 L13 8" 
                      stroke="#1e40af" 
                      stroke-width="2" 
                      fill="none" 
                      stroke-linecap="round" 
                      stroke-linejoin="round"
                    />
                  </svg>
                </div>
              `,
              iconSize: [20, 20],
              iconAnchor: [10, 10]
            });
            
            L.marker(point, { icon: arrowIcon })
              .addTo(map)
              .bindPopup(`
                <div class="text-center">
                  <strong>Hour ${prediction.hour}</strong><br>
                  Speed: ${wind_speed_mph} mph<br>
                  Direction: ${wind_direction_degrees}°
                </div>
              `);
          }
        });
      }
    }

    // Add wind speed legend
    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = function() {
      const div = L.DomUtil.create('div', 'info legend');
      div.style.backgroundColor = 'white';
      div.style.padding = '10px';
      div.style.borderRadius = '5px';
      div.style.border = '2px solid #ccc';
      div.innerHTML = `
        <h4 style="margin: 0 0 10px 0;">Wind Prediction</h4>
        <div style="font-size: 12px;">
          <div>📍 Location: ${latitude.toFixed(3)}°N, ${longitude.toFixed(3)}°W</div>
          <div style="margin: 8px 0;">
            <svg width="16" height="16" viewBox="0 0 16 16" style="display: inline-block; margin-right: 4px;">
              <path d="M8 2 L8 14 M6 6 L8 2 L10 6" stroke="#3b82f6" stroke-width="2" fill="none"/>
              <circle cx="8" cy="8" r="6" stroke="#3b82f6" stroke-width="1" fill="rgba(59, 130, 246, 0.1)"/>
            </svg>
            Hourly wind direction
          </div>
          <div style="margin: 8px 0;">
            <svg width="16" height="3" viewBox="0 0 16 3" style="display: inline-block; margin-right: 4px;">
              <line x1="0" y1="1.5" x2="16" y2="1.5" stroke="#3b82f6" stroke-width="2" stroke-dasharray="3,2"/>
            </svg>
            Wind trajectory path
          </div>
          <div style="margin: 8px 0;">
            <svg width="16" height="16" viewBox="0 0 16 16" style="display: inline-block; margin-right: 4px;">
              <path d="M8 2 L8 14 M6 6 L8 2 L10 6" stroke="#1e40af" stroke-width="1.5" fill="none"/>
            </svg>
            Path direction markers
          </div>
          <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee;">
            Hours: ${windData?.hours_ahead || 6}
          </div>
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
  }, [latitude, longitude, windData]);

  return (
    <div className="relative map-container">
      <div ref={mapRef} style={{ height: '100%', width: '100%' }} />
      
      {/* Info overlay when no wind data */}
      {!windData && (
        <div className="absolute top-4 left-4 bg-white bg-opacity-90 rounded-lg p-3 shadow-md">
          <div className="text-center">
            <Wind className="mx-auto mb-2 text-gray-400" size={20} />
            <p className="text-gray-600 text-sm">Submit form to see wind predictions</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default WindMap; 