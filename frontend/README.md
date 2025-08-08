# PlumeTrackAI Frontend

A modern React web application for visualizing wind predictions and plume travel calculations from the PlumeTrackAI API.

## Features

- **Wind Prediction Visualization**: Interactive maps showing wind conditions and hourly predictions
- **Plume Travel Analysis**: Visualize gas plume travel paths from source to risk zones
- **Real-time API Integration**: Connect to your PlumeTrackAI backend API
- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Maps**: Powered by Leaflet with custom markers and overlays

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- PlumeTrackAI backend API running on `http://localhost:8000`

## Installation

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```

4. **Open your browser:**
   Navigate to `http://localhost:3000`

## Usage

### Wind Prediction Tab

1. **Enter Location**: Input latitude and longitude coordinates
2. **Set Parameters**: Configure hours ahead, forecast weight, and confidence threshold
3. **Predict**: Click "Predict Wind Conditions" to get results
4. **View Results**: 
   - Interactive map with wind direction indicators
   - Detailed hourly predictions
   - Validation metrics and confidence scores

### Plume Travel Tab

1. **Source Location**: Enter the gas plume source coordinates
2. **Risk Location**: Enter the destination/risk zone coordinates
3. **Set Parameters**: Configure prediction parameters
4. **Calculate**: Click "Calculate Plume Travel" to analyze the path
5. **View Results**:
   - Interactive map showing plume travel path
   - Travel summary with arrival predictions
   - Hour-by-hour travel log

## API Configuration

The frontend connects to your PlumeTrackAI API at `http://localhost:8000`. To change the API URL:

1. Edit `src/services/api.js`
2. Update the `API_BASE_URL` constant
3. Restart the development server

## Project Structure

```
frontend/
├── public/
│   └── index.html          # Main HTML file
├── src/
│   ├── components/         # React components
│   │   ├── Header.js       # Application header
│   │   ├── WindPrediction.js # Wind prediction form and results
│   │   ├── PlumeTravel.js  # Plume travel form and results
│   │   ├── WindMap.js      # Interactive wind map
│   │   ├── PlumeMap.js     # Interactive plume map
│   │   ├── WindResults.js  # Wind prediction results display
│   │   └── PlumeResults.js # Plume travel results display
│   ├── services/
│   │   └── api.js          # API service functions
│   ├── App.js              # Main application component
│   ├── index.js            # React entry point
│   └── index.css           # Global styles
├── package.json            # Dependencies and scripts
├── tailwind.config.js      # Tailwind CSS configuration
└── README.md               # This file
```

## Technologies Used

- **React 18**: Modern React with hooks and functional components
- **Leaflet**: Interactive mapping library
- **Tailwind CSS**: Utility-first CSS framework
- **Lucide React**: Beautiful icon library
- **Axios**: HTTP client for API requests

## Available Scripts

- `npm start`: Start development server
- `npm build`: Build for production
- `npm test`: Run tests
- `npm eject`: Eject from Create React App

## Customization

### Styling
- Modify `tailwind.config.js` for theme customization
- Edit `src/index.css` for global styles
- Component-specific styles are in Tailwind classes

### Map Configuration
- Update map tiles in `WindMap.js` and `PlumeMap.js`
- Customize markers and overlays
- Modify legend content and styling

### API Integration
- Add new API endpoints in `src/services/api.js`
- Create new components for additional features
- Extend result displays as needed

## Troubleshooting

### Common Issues

1. **API Connection Error**:
   - Ensure your PlumeTrackAI backend is running on port 8000
   - Check CORS settings in your API
   - Verify network connectivity

2. **Map Not Loading**:
   - Check internet connection (required for map tiles)
   - Verify Leaflet CSS is loaded
   - Check browser console for errors

3. **Build Errors**:
   - Clear `node_modules` and reinstall dependencies
   - Check Node.js version compatibility
   - Verify all required dependencies are installed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the PlumeTrackAI system. See the main project license for details. 