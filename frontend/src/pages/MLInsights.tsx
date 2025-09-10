import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, Grid, Chip, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, CartesianGrid, LineChart, Line } from 'recharts';
import { motion } from 'framer-motion';
import axios from 'axios';

const MLInsights: React.FC = () => {
  const [signalsData, setSignalsData] = useState<any>(null);
  const [featureImportanceData, setFeatureImportanceData] = useState<any>(null);
  const [mlEvaluationData, setMlEvaluationData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch all ML-related data in parallel
        const [signalsResponse, featureResponse, evaluationResponse] = await Promise.allSettled([
          axios.get('/api/signals'),
          axios.get('/api/ml/feature-importance'),
          axios.get('/api/ml/evaluation')
        ]);

        // Handle signals data
        if (signalsResponse.status === 'fulfilled' && signalsResponse.value.data && !signalsResponse.value.data.error) {
          setSignalsData(signalsResponse.value.data);
        }

        // Handle feature importance data
        if (featureResponse.status === 'fulfilled' && featureResponse.value.data && !featureResponse.value.data.error) {
          setFeatureImportanceData(featureResponse.value.data);
        }

        // Handle ML evaluation data
        if (evaluationResponse.status === 'fulfilled' && evaluationResponse.value.data && !evaluationResponse.value.data.error) {
          setMlEvaluationData(evaluationResponse.value.data);
        }
      } catch (error) {
        console.error('Error fetching ML data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Use real feature importance data from ML model
  const featureImportances = featureImportanceData?.feature_importances || [];

  // Colors for pie charts
  const COLORS = ['#14b8a6', '#38bdf8', '#f59e0b', '#ef4444', '#8b5cf6'];

  // Confusion matrix data
  const confusionMatrix = mlEvaluationData?.confusion_matrix;
  const confusionMatrixData = confusionMatrix ? [
    { name: 'True Positives', value: confusionMatrix.true_positives, color: '#10b981' },
    { name: 'False Positives', value: confusionMatrix.false_positives, color: '#f59e0b' },
    { name: 'True Negatives', value: confusionMatrix.true_negatives, color: '#10b981' },
    { name: 'False Negatives', value: confusionMatrix.false_negatives, color: '#ef4444' }
  ] : [];

  // Market regime distribution data
  const regimeDistribution = mlEvaluationData?.market_regime_distribution || [];

  // Sparklines data for signal components (last 20 data points)
  const sparklinesData = signalsData?.recent_signals?.slice(-20).map((signal: any, index: number) => ({
    index,
    momentum: signal.momentum_signal || 0,
    meanReversion: signal.mean_reversion_signal || 0,
    mlSignal: signal.ml_signal || 0,
    confidence: (signal.confidence || 0) * 100
  })) || [];

  if (loading) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom className="gradient-text">Machine Learning Insights</Typography>
        <Typography>Loading ML insights...</Typography>
      </Box>
    );
  }

  if (!signalsData || !signalsData.recent_signals || signalsData.recent_signals.length === 0) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom className="gradient-text">Machine Learning Insights</Typography>
        <Paper className="card p-5">
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No ML Data Available
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              To view ML insights, please run a backtest first. The ML model will be trained on your data and provide:
            </Typography>
            <Box sx={{ textAlign: 'left', maxWidth: 400, mx: 'auto' }}>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                • Feature importance analysis from the trained Random Forest model
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                • Model performance metrics and confusion matrix
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                • Market regime distribution analysis
              </Typography>
              <Typography variant="body2" color="text.secondary">
                • ML confidence distribution and signal quality metrics
              </Typography>
            </Box>
          </Box>
        </Paper>
      </Box>
    );
  }
  return (
    <Box>
      <Typography variant="h4" gutterBottom className="gradient-text">Machine Learning Insights</Typography>

      <Grid container spacing={3}>
        {/* Feature Importances Chart */}
        <Grid item xs={12} md={6}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Feature Importances (ML Model)</Typography>
              {featureImportances.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={featureImportances}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="name" 
                      interval={0} 
                      angle={-20} 
                      textAnchor="end" 
                      height={60}
                      fontSize={12}
                    />
                    <YAxis />
                    <Tooltip 
                      formatter={(value) => [typeof value === 'number' ? value.toFixed(4) : value, 'Importance']} 
                    />
                    <Bar dataKey="importance" fill="#14b8a6" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="body2" color="text.secondary">
                    No feature importance data available. Run a backtest to train the ML model.
                  </Typography>
                </Box>
              )}
            </Paper>
          </motion.div>
        </Grid>

        {/* Confusion Matrix */}
        <Grid item xs={12} md={6}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.1 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Model Performance (Confusion Matrix)</Typography>
              {confusionMatrix ? (
                <Box>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={confusionMatrixData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" fontSize={12} />
                      <YAxis />
                      <Tooltip formatter={(value) => [value, 'Count']} />
                      <Bar dataKey="value" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                  <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color="success.main">
                        {(confusionMatrix.accuracy * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="caption" color="text.secondary">Accuracy</Typography>
                    </Box>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color="primary.main">
                        {(confusionMatrix.precision * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="caption" color="text.secondary">Precision</Typography>
                    </Box>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color="secondary.main">
                        {(confusionMatrix.recall * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="caption" color="text.secondary">Recall</Typography>
                    </Box>
                  </Box>
                </Box>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="body2" color="text.secondary">
                    No confusion matrix data available. Run a backtest to evaluate model performance.
                  </Typography>
                </Box>
              )}
            </Paper>
          </motion.div>
        </Grid>

        {/* Market Regime Distribution */}
        <Grid item xs={12} md={6}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.2 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Market Regime Distribution</Typography>
              {regimeDistribution.length > 0 ? (
                <Box>
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie
                        data={regimeDistribution}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ regime, percentage }) => `${regime}: ${percentage.toFixed(1)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="count"
                      >
                        {regimeDistribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value, name, props) => [
                        `${value} periods (${props.payload.percentage.toFixed(1)}%)`,
                        'Count'
                      ]} />
                    </PieChart>
                  </ResponsiveContainer>
                </Box>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="body2" color="text.secondary">
                    No market regime data available.
                  </Typography>
                </Box>
              )}
            </Paper>
          </motion.div>
        </Grid>

        {/* Current Signal Analysis */}
        <Grid item xs={12} md={6}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.3 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Current Signal Analysis</Typography>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Typography variant="subtitle1" gutterBottom>Current Market Regime</Typography>
                  <Chip 
                    label={signalsData?.current_signal?.market_regime || 'Unknown'} 
                    color={signalsData?.current_signal?.market_regime === 'trending' ? 'success' : 
                           signalsData?.current_signal?.market_regime === 'ranging' ? 'info' : 
                           signalsData?.current_signal?.market_regime === 'volatile' ? 'warning' : 'default'} 
                    variant="filled" 
                    sx={{ mb: 2 }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    ML Confidence: {((signalsData?.current_signal?.confidence || 0) * 100).toFixed(1)}%
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="subtitle1" gutterBottom>Signal Components</Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    {/* Current Values */}
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Momentum:</Typography>
                        <Typography variant="body2" color="primary">
                          {(signalsData?.current_signal?.momentum_score || 0).toFixed(3)}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Mean Reversion:</Typography>
                        <Typography variant="body2" color="secondary">
                          {(signalsData?.current_signal?.mean_reversion_score || 0).toFixed(3)}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">ML Signal:</Typography>
                        <Typography variant="body2" color="info.main">
                          {(signalsData?.current_signal?.ml_score || 0).toFixed(3)}
                        </Typography>
                      </Box>
                    </Box>

                    {/* Sparklines */}
                    {sparklinesData.length > 0 && (
                      <Box>
                        <Typography variant="caption" color="text.secondary" gutterBottom>
                          Recent Trends (Last 20 periods)
                        </Typography>
                        <ResponsiveContainer width="100%" height={80}>
                          <LineChart data={sparklinesData}>
                            <XAxis dataKey="index" hide />
                            <YAxis hide />
                            <Tooltip 
                              formatter={(value, name) => [typeof value === 'number' ? value.toFixed(3) : value, name]}
                              labelFormatter={(index) => `Period ${index}`}
                            />
                            <Line 
                              type="monotone" 
                              dataKey="momentum" 
                              stroke="#1976d2" 
                              strokeWidth={1.5}
                              dot={false}
                              name="Momentum"
                            />
                            <Line 
                              type="monotone" 
                              dataKey="meanReversion" 
                              stroke="#9c27b0" 
                              strokeWidth={1.5}
                              dot={false}
                              name="Mean Reversion"
                            />
                            <Line 
                              type="monotone" 
                              dataKey="mlSignal" 
                              stroke="#0288d1" 
                              strokeWidth={1.5}
                              dot={false}
                              name="ML Signal"
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </Box>
                    )}
                  </Box>
                </Grid>
              </Grid>
            </Paper>
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MLInsights;


