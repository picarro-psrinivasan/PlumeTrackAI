const API_BASE_URL = 'http://localhost:8000';

// Helper function to handle API responses
const handleResponse = async (response) => {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }
  return response.json();
};

// API service functions
export const api = {
  // Health check
  health: async () => {
    const response = await fetch(`${API_BASE_URL}/health`);
    return handleResponse(response);
  },

  // Wind prediction
  predictWind: async (data) => {
    const response = await fetch(`${API_BASE_URL}/wind/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    return handleResponse(response);
  },

  // Simple wind prediction
  predictWindSimple: async (params) => {
    const queryString = new URLSearchParams(params).toString();
    const response = await fetch(`${API_BASE_URL}/wind/predict/simple?${queryString}`);
    return handleResponse(response);
  },

  // Plume prediction
  predictPlume: async (data) => {
    const response = await fetch(`${API_BASE_URL}/plume/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    return handleResponse(response);
  },

  // Simple plume prediction
  predictPlumeSimple: async (params) => {
    const queryString = new URLSearchParams(params).toString();
    const response = await fetch(`${API_BASE_URL}/plume/predict/simple?${queryString}`);
    return handleResponse(response);
  },
};

// Convenience functions
export const predictWind = api.predictWind;
export const predictPlume = api.predictPlume;
export const checkHealth = api.health; 