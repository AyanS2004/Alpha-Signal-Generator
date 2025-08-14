import React from 'react';
import { Box, Typography, Paper, Grid } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { motion } from 'framer-motion';

const eq = Array.from({ length: 180 }).map((_, i) => ({ t: i, equity: 100 + i * 0.2 + Math.sin(i / 4) * 2 }));
const dd = eq.map((d) => ({ t: d.t, dd: Math.min(0, (Math.random() - 0.8) * 10) }));

const Risk: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom className="gradient-text">Risk Management</Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Value at Risk (VaR)</Typography>
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={dd}>
                  <defs>
                    <linearGradient id="varGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.6} />
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="t" hide />
                  <YAxis />
                  <Tooltip />
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
                <LineChart data={eq}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="t" hide />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="equity" stroke="#38bdf8" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Paper>
          </motion.div>
        </Grid>

        <Grid item xs={12}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.2 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Risk Heatmap</Typography>
              <Box className="grid grid-cols-12 gap-1">
                {Array.from({ length: 12 * 8 }).map((_, i) => (
                  <div
                    key={i}
                    className="h-6 rounded"
                    style={{ backgroundColor: `hsl(0, 85%, ${30 + Math.random() * 30}%)` }}
                  />
                ))}
              </Box>
            </Paper>
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Risk;


