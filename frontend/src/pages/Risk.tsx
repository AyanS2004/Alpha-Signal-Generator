import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, Grid } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
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

  // Generate drawdown data from equity curve
  const drawdownData = equityData.map((point: any, index: number) => {
    const peak = Math.max(...equityData.slice(0, index + 1).map((p: any) => p.equity));
    const drawdown = ((point.equity - peak) / peak) * 100;
    return { t: point.t, dd: Math.min(0, drawdown) };
  });

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
  return (
    <Box>
      <Typography variant="h4" gutterBottom className="gradient-text">Risk Management</Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Value at Risk (VaR)</Typography>
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={drawdownData}>
                  <defs>
                    <linearGradient id="varGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.6} />
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="t" hide />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${typeof value === 'number' ? value.toFixed(2) : value}%`, 'Drawdown']} />
                  <Area type="monotone" dataKey="dd" stroke="#ef4444" fillOpacity={1} fill="url(#varGrad)" />
                </AreaChart>
              </ResponsiveContainer>
            </Paper>
          </motion.div>
        </Grid>

        <Grid item xs={12} md={6}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.1 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Equity & Drawdown</Typography>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={equityData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="t" hide />
                  <YAxis />
                  <Tooltip formatter={(value) => [`$${typeof value === 'number' ? value.toFixed(2) : value}`, 'Equity']} />
                  <Line type="monotone" dataKey="equity" stroke="#38bdf8" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Paper>
          </motion.div>
        </Grid>

        <Grid item xs={12}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.2 }}>
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
                      Beta
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2 }}>
                    <Typography variant="h4" color="success.main" gutterBottom>
                      {dashboardData.riskMetrics?.alpha?.toFixed(1) || '0.0'}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Alpha
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


