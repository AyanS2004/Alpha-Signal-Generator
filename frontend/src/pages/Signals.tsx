import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, Grid, Chip, ToggleButtonGroup, ToggleButton } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';
import axios from 'axios';

type SignalRow = { date: string; type: string; strength: number; price: number };

const Signals: React.FC = () => {
  const [filter, setFilter] = useState<string | null>('all');
  const [signalsData, setSignalsData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSignals = async () => {
      try {
        const response = await axios.get('/api/signals');
        if (response.data && !response.data.error) {
          setSignalsData(response.data);
        }
      } catch (error) {
        console.error('Error fetching signals:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchSignals();
  }, []);

  // Convert API data to SignalRow format
  const apiSignals: SignalRow[] = signalsData?.recent_signals?.map((signal: any) => ({
    date: signal.date,
    type: signal.signal === 'BUY' ? 'momentum' : signal.signal === 'SELL' ? 'mean-reversion' : 'ml',
    strength: signal.confidence,
    price: signal.price,
  })) || [];

  const filtered = apiSignals.filter((s) => (filter === 'all' ? true : s.type === filter));

  if (loading) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom className="gradient-text">Signal Generation</Typography>
        <Typography>Loading signals data...</Typography>
      </Box>
    );
  }

  if (!signalsData || !signalsData.recent_signals || signalsData.recent_signals.length === 0) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom className="gradient-text">Signal Generation</Typography>
        <Paper className="card p-5">
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No Signal Data Available
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Run a backtest to generate trading signals and see real-time analysis.
            </Typography>
          </Box>
        </Paper>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom className="gradient-text">Signal Generation</Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={7}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
            <Paper className="card p-5">
              <Box className="flex items-center justify-between mb-3">
                <Typography variant="h6">Signals Over Time</Typography>
                <ToggleButtonGroup
                  size="small"
                  exclusive
                  value={filter}
                  onChange={(_, val) => setFilter(val)}
                >
                  <ToggleButton value="all">All</ToggleButton>
                  <ToggleButton value="momentum">Momentum</ToggleButton>
                  <ToggleButton value="mean-reversion">Mean-Reversion</ToggleButton>
                  <ToggleButton value="ml">ML</ToggleButton>
                </ToggleButtonGroup>
              </Box>

              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={filtered}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" hide />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" domain={[-1, 1]} />
                  <Tooltip />
                  <Line yAxisId="left" type="monotone" dataKey="price" stroke="#38bdf8" dot={false} />
                  <Line yAxisId="right" type="monotone" dataKey="strength" stroke="#14b8a6" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Paper>
          </motion.div>
        </Grid>

        <Grid item xs={12} md={5}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.1 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Latest Signals</Typography>
              <Box className="space-y-2">
                {filtered.slice(-10).reverse().map((s, i) => (
                  <Box key={i} className="flex items-center justify-between p-3 rounded-lg bg-[#0f1115] border border-zinc-800">
                    <div>
                      <div className="text-sm text-gray-400">{s.date}</div>
                      <div className="font-semibold">${s.price.toFixed(2)}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Chip label={s.type} color={s.type === 'momentum' ? 'info' : s.type === 'ml' ? 'secondary' : 'default'} size="small" />
                      <Chip label={`Strength: ${s.strength.toFixed(2)}`} color={s.strength > 0.5 ? 'success' : s.strength < 0.3 ? 'error' : 'warning'} size="small" />
                    </div>
                  </Box>
                ))}
              </Box>
            </Paper>
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Signals;


