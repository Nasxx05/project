/**
 * API service for communicating with the Hospital Forecasting backend.
 */
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 300000, // 5 min for long training jobs
  headers: { 'Content-Type': 'application/json' },
});

// --- Data Upload & Management ---

export const uploadAdmissions = (file) => {
  const form = new FormData();
  form.append('file', file);
  return api.post('/api/v1/upload/admissions', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const uploadHolidays = (file) => {
  const form = new FormData();
  form.append('file', file);
  return api.post('/api/v1/upload/holidays', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const getDataSummary = (params) =>
  api.get('/api/v1/data/summary', { params });

export const getHistoricalOccupancy = (params) =>
  api.get('/api/v1/occupancy/historical', { params });

export const clearData = (dataType, confirm = 'yes') =>
  api.delete(`/api/v1/data/clear/${dataType}`, { params: { confirm } });

// --- Bed Occupancy Forecasting ---

export const generateForecast = (data) =>
  api.post('/api/v1/forecast/bed/generate', data);

export const getLatestForecast = (limit = 60) =>
  api.get('/api/v1/forecast/bed/latest', { params: { limit } });

export const getForecastHistory = (params) =>
  api.get('/api/v1/forecast/bed/history', { params });

export const getForecastById = (id) =>
  api.get(`/api/v1/forecast/bed/${id}`);

// --- LOS Prediction ---

export const predictLOS = (data) =>
  api.post('/api/v1/los/predict', data);

export const batchPredictLOS = (file) => {
  const form = new FormData();
  form.append('file', file);
  return api.post('/api/v1/los/batch-predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const getLOSStatistics = (params) =>
  api.get('/api/v1/los/statistics', { params });

// --- Model Training ---

export const trainOccupancyModel = (data) =>
  api.post('/api/v1/model/train/occupancy', data);

export const trainLOSModel = (data) =>
  api.post('/api/v1/model/train/los', data);

export const getModelMetrics = (params) =>
  api.get('/api/v1/model/metrics', { params });

export const listModels = () =>
  api.get('/api/v1/model/list');

export const activateModel = (modelId) =>
  api.post(`/api/v1/model/activate/${modelId}`);

// --- Analytics ---

export const getForecastAccuracy = (params) =>
  api.get('/api/v1/analytics/forecast-accuracy', { params });

export const getSeasonalPatterns = () =>
  api.get('/api/v1/analytics/seasonal-patterns');

export const getCapacityPlanning = (params) =>
  api.get('/api/v1/analytics/capacity-planning', { params });

export const getFeatureImportance = () =>
  api.get('/api/v1/analytics/feature-importance');

// --- Health ---

export const healthCheck = () => api.get('/health');

export default api;
