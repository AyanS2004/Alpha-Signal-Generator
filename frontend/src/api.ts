import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || '';

export const api = axios.create({
  baseURL: API_BASE || '',
});

export const endpoints = {
  optimizeBayesian: '/api/optimize/bayesian',
  walkForward: '/api/walk-forward',
  factorsPca: '/api/factors/pca',
  rlPositionSize: '/api/rl/position-size',
  costsEstimate: '/api/costs/estimate',
  ensembleFit: '/api/ensemble/fit',
  ensemblePredict: '/api/ensemble/predict',
};



