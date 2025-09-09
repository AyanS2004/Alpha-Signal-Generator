import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, Grid, Chip } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ScatterChart, Scatter, CartesianGrid } from 'recharts';
import { motion } from 'framer-motion';
import axios from 'axios';

const MLInsights: React.FC = () => {
  const [signalsData, setSignalsData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('/api/signals');
        if (response.data && !response.data.error) {
          setSignalsData(response.data);
        }
      } catch (error) {
        console.error('Error fetching signals data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Generate feature importance data from signals
  const featureImportances = [
    { name: 'Momentum', importance: Math.abs(signalsData?.current_signal?.momentum_score || 0) },
    { name: 'Mean Reversion', importance: Math.abs(signalsData?.current_signal?.mean_reversion_score || 0) },
    { name: 'ML Signal', importance: Math.abs(signalsData?.current_signal?.ml_score || 0) },
    { name: 'RSI', importance: Math.abs((signalsData?.current_signal?.rsi || 50) - 50) / 50 },
    { name: 'Volume Ratio', importance: Math.abs((signalsData?.current_signal?.volume_ratio || 1) - 1) },
  ].filter(f => f.importance > 0);

  // Generate scatter plot data from recent signals
  const scatterData = signalsData?.recent_signals?.slice(0, 50).map((signal: any) => ({
    x: signal.momentum_signal || 0,
    y: signal.mean_reversion_signal || 0
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
            <Typography variant="body2" color="text.secondary">
              Run a backtest to generate ML insights and feature analysis.
            </Typography>
          </Box>
        </Paper>
      </Box>
    );
  }
  return (
    <Box>
      <Typography variant="h4" gutterBottom className="gradient-text">Machine Learning Insights</Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Feature Importances</Typography>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={featureImportances}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" interval={0} angle={-20} textAnchor="end" height={60} />
                  <YAxis />
                  <Tooltip formatter={(value) => [typeof value === 'number' ? value.toFixed(3) : value, 'Importance']} />
                  <Bar dataKey="importance" fill="#14b8a6" />
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </motion.div>
        </Grid>

        <Grid item xs={12} md={6}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.1 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Predicted vs Actual Returns</Typography>
              <ResponsiveContainer width="100%" height={280}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="x" name="Momentum Signal" />
                  <YAxis type="number" dataKey="y" name="Mean Reversion Signal" />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(value, name) => [typeof value === 'number' ? value.toFixed(3) : value, name]} />
                  <Scatter data={scatterData} fill="#38bdf8" />
                </ScatterChart>
              </ResponsiveContainer>
            </Paper>
          </motion.div>
        </Grid>

        <Grid item xs={12}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.2 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Market Regime Analysis</Typography>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
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
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1" gutterBottom>Signal Components</Typography>
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


