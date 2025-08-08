import React, { useState } from 'react';
import { Wind, MapPin, Zap, Globe } from 'lucide-react';
import WindPrediction from './components/WindPrediction';
import PlumeTravel from './components/PlumeTravel';
import Header from './components/Header';

function App() {
  const [activeTab, setActiveTab] = useState('wind');

  const tabs = [
    {
      id: 'wind',
      name: 'Wind Prediction',
      icon: Wind,
      description: 'Predict wind conditions for any location'
    },
    {
      id: 'plume',
      name: 'Plume Travel',
      icon: MapPin,
      description: 'Calculate gas plume travel paths'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        {/* Tab Navigation */}
        <div className="mb-8">
          <div className="flex space-x-1 bg-white p-1 rounded-lg shadow-sm border">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex-1 flex items-center justify-center space-x-2 px-4 py-3 rounded-md transition-all duration-200 ${
                    activeTab === tab.id
                      ? 'bg-primary-500 text-white shadow-md'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <Icon size={20} />
                  <span className="font-medium">{tab.name}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Tab Content */}
        <div className="animate-fade-in">
          {activeTab === 'wind' && <WindPrediction />}
          {activeTab === 'plume' && <PlumeTravel />}
        </div>

        {/* Footer */}
        <footer className="mt-16 text-center text-gray-500 text-sm">
          <div className="flex items-center justify-center space-x-2">
            <Zap size={16} />
            <span>PlumeTrackAI - Powered by LSTM & Forecast Models</span>
            <Globe size={16} />
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App; 