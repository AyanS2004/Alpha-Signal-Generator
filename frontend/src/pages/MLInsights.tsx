import React from 'react';
import { Box, Typography, Paper, Grid, Chip } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ScatterChart, Scatter, CartesianGrid } from 'recharts';
import { motion } from 'framer-motion';

const featureImportances = [
  { name: 'Momentum_20', importance: 0.32 },
  { name: 'Volatility_30', importance: 0.22 },
  { name: 'SMA_Cross', importance: 0.18 },
  { name: 'RSI_14', importance: 0.16 },
  { name: 'VolumeSpike', importance: 0.12 },
];

const scatter = Array.from({ length: 100 }).map((_, i) => ({ x: Math.random() - 0.5, y: Math.random() - 0.5 }));

const MLInsights: React.FC = () => {
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
                  <Tooltip />
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
                  <XAxis type="number" dataKey="x" name="Predicted" />
                  <YAxis type="number" dataKey="y" name="Actual" />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter data={scatter} fill="#38bdf8" />
                </ScatterChart>
              </ResponsiveContainer>
            </Paper>
          </motion.div>
        </Grid>

        <Grid item xs={12}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.2 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Volatility Regime Classification</Typography>
              <Box className="flex gap-3 flex-wrap">
                {['Low Vol', 'Moderate', 'High Vol', 'Crisis'].map((r, idx) => (
                  <Chip key={r} label={r} color={idx === 0 ? 'success' : idx === 1 ? 'info' : idx === 2 ? 'warning' : 'error'} variant="outlined" />
                ))}
              </Box>
            </Paper>
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MLInsights;


