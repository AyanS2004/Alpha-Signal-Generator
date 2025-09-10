import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, Grid, Alert, Chip } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';
import { motion } from 'framer-motion';
import axios from 'axios';

const Risk: React.FC = () => {
  const [dashboardData, setDashboardData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('/api/dashboard');
        if (response.data && !response.data.error) {
          setDashboardData(response.data);
        }
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Generate equity curve data from dashboard data
  const equityData = dashboardData?.equityData?.map((point: any, index: number) => ({
    t: index,
    equity: point.value
  })) || [];

  // Check if there were any trades
  const totalTrades = dashboardData?.totalTrades || 0;
  const hasTrades = totalTrades > 0;

  // Generate drawdown data from equity curve
  const drawdownData = equityData.map((point: any, index: number) => {
    const peak = Math.max(...equityData.slice(0, index + 1).map((p: any) => p.equity));
    const drawdown = ((point.equity - peak) / peak) * 100;
    return { t: point.t, dd: Math.min(0, drawdown) };
  });

  // Generate returns data for histogram
  const returnsData = [];
  if (equityData.length > 1) {
    for (let i = 1; i < equityData.length; i++) {
      const prevEquity = equityData[i - 1].equity;
      const currentEquity = equityData[i].equity;
      if (prevEquity > 0) {
        const dailyReturn = ((currentEquity - prevEquity) / prevEquity) * 100;
        returnsData.push(dailyReturn);
      }
    }
  }

  // Create histogram data for returns distribution
  const createHistogramData = (returns: number[]) => {
    if (returns.length === 0) return [];
    
    const min = Math.min(...returns);
    const max = Math.max(...returns);
    const binCount = Math.min(10, Math.max(5, Math.floor(returns.length / 5)));
    const binSize = (max - min) / binCount;
    
    const bins = [];
    for (let i = 0; i < binCount; i++) {
      const binStart = min + i * binSize;
      const binEnd = min + (i + 1) * binSize;
      const count = returns.filter(r => r >= binStart && r < binEnd).length;
      bins.push({
        range: `${binStart.toFixed(1)}% to ${binEnd.toFixed(1)}%`,
        count,
        percentage: (count / returns.length) * 100
      });
    }
    return bins;
  };

  const histogramData = createHistogramData(returnsData);

  // Calculate rolling volatility (20-period)
  const rollingVolatilityData = [];
  if (returnsData.length >= 20) {
    for (let i = 19; i < returnsData.length; i++) {
      const window = returnsData.slice(i - 19, i + 1);
      const mean = window.reduce((a, b) => a + b, 0) / window.length;
      const variance = window.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / window.length;
      const volatility = Math.sqrt(variance);
      rollingVolatilityData.push({
        period: i,
        volatility: volatility
      });
    }
  }

  if (loading) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom className="gradient-text">Risk Management</Typography>
        <Typography>Loading risk data...</Typography>
      </Box>
    );
  }

  if (!dashboardData || !dashboardData.equityData || dashboardData.equityData.length === 0) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom className="gradient-text">Risk Management</Typography>
        <Paper className="card p-5">
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No Risk Data Available
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Run a backtest to generate risk metrics and performance analysis.
            </Typography>
          </Box>
        </Paper>
      </Box>
    );
  }

  // Show warning if no trades were executed
  if (!hasTrades) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom className="gradient-text">Risk Management</Typography>
        
        <Alert severity="warning" sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            No Trades Executed
          </Typography>
          <Typography variant="body2" sx={{ mb: 2 }}>
            The backtest completed but no trades were executed. This means the strategy parameters 
            were too restrictive to generate any trading signals. Risk metrics cannot be calculated 
            without trading activity.
          </Typography>
          <Typography variant="body2" sx={{ mb: 1 }}>
            To fix this, try adjusting your strategy parameters in the Settings page:
          </Typography>
          <Box component="ul" sx={{ pl: 2, mb: 0 }}>
            <li>Lower the momentum threshold (currently {dashboardData.riskMetrics?.momentumThreshold || 'N/A'}%)</li>
            <li>Reduce the mean reversion threshold (currently {dashboardData.riskMetrics?.meanReversionThreshold || 'N/A'}%)</li>
            <li>Lower the final signal threshold (currently {dashboardData.riskMetrics?.finalSignalThreshold || 'N/A'})</li>
            <li>Relax volume confirmation requirements</li>
          </Box>
        </Alert>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
              <Paper className="card p-5">
                <Typography variant="h6" gutterBottom>Equity Curve (No Trades)</Typography>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={equityData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="t" hide />
                    <YAxis />
                    <Tooltip formatter={(value) => [`$${typeof value === 'number' ? value.toFixed(2) : value}`, 'Equity']} />
                    <Line type="monotone" dataKey="equity" stroke="#38bdf8" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                  Flat equity curve indicates no trading activity
                </Typography>
              </Paper>
            </motion.div>
          </Grid>

          <Grid item xs={12} md={6}>
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.1 }}>
              <Paper className="card p-5">
                <Typography variant="h6" gutterBottom>Risk Metrics (No Data)</Typography>
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Risk metrics require trading activity to calculate
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 3 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Total Trades:</Typography>
                      <Chip label="0" color="error" size="small" />
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Strategy Status:</Typography>
                      <Chip label="Inactive" color="warning" size="small" />
                    </Box>
                  </Box>
                </Box>
              </Paper>
            </motion.div>
          </Grid>
        </Grid>
      </Box>
    );
  }
  return (
    <Box>
      <Typography variant="h4" gutterBottom className="gradient-text">Risk Management</Typography>

      {/* Success message for active strategy */}
      <Alert severity="success" sx={{ mb: 3 }}>
        <Typography variant="body2">
          Strategy executed <strong>{totalTrades} trades</strong> with a total return of{' '}
          <strong>{dashboardData.totalReturn?.toFixed(2) || '0.00'}%</strong>.
        </Typography>
      </Alert>

      <Grid container spacing={3}>
        {/* Drawdown Over Time Chart (Fixed Labeling) */}
        <Grid item xs={12} md={6}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Drawdown Over Time</Typography>
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={drawdownData}>
                  <defs>
                    <linearGradient id="drawdownGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.6} />
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="t" hide />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${typeof value === 'number' ? value.toFixed(2) : value}%`, 'Drawdown']} />
                  <Area type="monotone" dataKey="dd" stroke="#ef4444" fillOpacity={1} fill="url(#drawdownGrad)" />
                </AreaChart>
              </ResponsiveContainer>
            </Paper>
          </motion.div>
        </Grid>

        {/* Equity Curve Chart */}
        <Grid item xs={12} md={6}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.1 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Equity Curve</Typography>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={equityData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="t" hide />
                  <YAxis />
                  <Tooltip formatter={(value) => [`$${typeof value === 'number' ? value.toFixed(2) : value}`, 'Equity']} />
                  <Line type="monotone" dataKey="equity" stroke="#38bdf8" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Paper>
          </motion.div>
        </Grid>

        {/* Returns Distribution Histogram */}
        <Grid item xs={12} md={6}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.2 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Returns Distribution</Typography>
              {histogramData.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={histogramData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" angle={-45} textAnchor="end" height={80} fontSize={10} />
                    <YAxis />
                    <Tooltip formatter={(value, name, props) => [
                      `${value} (${props.payload.percentage.toFixed(1)}%)`,
                      'Count'
                    ]} />
                    <Bar dataKey="count" fill="#14b8a6" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="body2" color="text.secondary">
                    Insufficient data for returns distribution
                  </Typography>
                </Box>
              )}
            </Paper>
          </motion.div>
        </Grid>

        {/* Rolling Volatility Chart */}
        <Grid item xs={12} md={6}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.3 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Rolling Volatility (20-period)</Typography>
              {rollingVolatilityData.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={rollingVolatilityData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="period" hide />
                    <YAxis />
                    <Tooltip formatter={(value) => [`${typeof value === 'number' ? value.toFixed(2) : value}%`, 'Volatility']} />
                    <Line type="monotone" dataKey="volatility" stroke="#f59e0b" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="body2" color="text.secondary">
                    Insufficient data for rolling volatility (need 20+ periods)
                  </Typography>
                </Box>
              )}
            </Paper>
          </motion.div>
        </Grid>

        {/* Risk Metrics Summary */}
        <Grid item xs={12}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.4 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Risk Metrics Summary</Typography>
              <Grid container spacing={3}>
                <Grid item xs={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2 }}>
                    <Typography variant="h4" color="error.main" gutterBottom>
                      {dashboardData.riskMetrics?.var?.toFixed(1) || '0.0'}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Value at Risk (95%)
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2 }}>
                    <Typography variant="h4" color="primary" gutterBottom>
                      {dashboardData.riskMetrics?.beta?.toFixed(2) || '0.00'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Beta (vs SPY)
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2 }}>
                    <Typography variant="h4" color="success.main" gutterBottom>
                      {dashboardData.riskMetrics?.alpha?.toFixed(1) || '0.0'}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Alpha (Excess Return)
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2 }}>
                    <Typography variant="h4" color="warning.main" gutterBottom>
                      {dashboardData.riskMetrics?.volatility?.toFixed(1) || '0.0'}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Volatility
                    </Typography>
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

export default Risk;


