import React from 'react';
import { Wind, Activity } from 'lucide-react';

const Header = () => {
  return (
    <header className="bg-white shadow-sm border-b">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className="p-2 bg-primary-500 rounded-lg">
                <Wind className="text-white" size={24} />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">PlumeTrackAI</h1>
                <p className="text-sm text-gray-600">Wind & Plume Visualization</p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <Activity size={16} />
              <span>API Status: </span>
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-green-600">Online</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header; 