import React, { useState } from 'react';
import { Box, Typography, Paper, Grid, Button, TextField, Chip, LinearProgress } from '@mui/material';
import { Tune, PlayArrow } from '@mui/icons-material';
import { motion } from 'framer-motion';
import axios from 'axios';
import { api, endpoints } from '../api';

type Range = number[];

const Optimization: React.FC = () => {
  const [momentumLookback, setMomentumLookback] = useState<Range>([10, 25, 5]);
  const [momentumThreshold, setMomentumThreshold] = useState<Range>([0.01, 0.03, 0.005]);
  const [positionSize, setPositionSize] = useState<Range>([0.05, 0.2, 0.05]);
  const [csvPath, setCsvPath] = useState<string>('AAPL_minute.csv');
  const [loading, setLoading] = useState(false);
  const [bestParams, setBestParams] = useState<Record<string, any> | null>(null);
  const [bestSharpe, setBestSharpe] = useState<number | null>(null);
  const [bayesRunning, setBayesRunning] = useState(false);
  const [wfSummary, setWfSummary] = useState<any>(null);

  const buildRangeArray = (range: Range) => {
    const [start, end, step] = range;
    const out: number[] = [];
    for (let v = start; v <= end + 1e-12; v = Number((v + step).toFixed(6))) out.push(Number(v.toFixed(6)));
    return out;
  };

  const onRun = async () => {
    setLoading(true);
    try {
      const paramRanges = {
        momentum_lookback: buildRangeArray(momentumLookback),
        momentum_threshold: buildRangeArray(momentumThreshold),
        position_size: buildRangeArray(positionSize),
      };
      const { data } = await axios.post('/api/optimization', { paramRanges, csvPath });
      setBestParams(data.bestParams);
      setBestSharpe(data.bestSharpe);
    } catch (e) {
      // Silently ignore for now
    } finally {
      setLoading(false);
    }
  };

  const onRunBayesian = async () => {
    setBayesRunning(true);
    try {
      const { data } = await api.post(endpoints.optimizeBayesian, { csvPath });
      setBestParams(data.bestParams);
      setBestSharpe(data.bestSharpe);
    } catch {}
    finally { setBayesRunning(false); }
  };

  const onWalkForward = async () => {
    setBayesRunning(true);
    try {
      const { data } = await api.post(endpoints.walkForward, { csvPath });
      setWfSummary(data);
    } catch {}
    finally { setBayesRunning(false); }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom className="gradient-text">Parameter Optimization</Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.4 }}>
            <Paper className="card p-5">
              <Box className="flex items-center gap-2 mb-4">
                <Tune />
                <Typography variant="h6">Grid Search Space</Typography>
              </Box>

              <Box className="space-y-6">
                <Box>
                  <Typography gutterBottom>Momentum Lookback [start, end, step]</Typography>
                  <Box className="grid grid-cols-3 gap-2">
                    <TextField size="small" type="number" value={momentumLookback[0]} onChange={(e) => setMomentumLookback([Number(e.target.value), momentumLookback[1], momentumLookback[2]])} />
                    <TextField size="small" type="number" value={momentumLookback[1]} onChange={(e) => setMomentumLookback([momentumLookback[0], Number(e.target.value), momentumLookback[2]])} />
                    <TextField size="small" type="number" value={momentumLookback[2]} onChange={(e) => setMomentumLookback([momentumLookback[0], momentumLookback[1], Number(e.target.value)])} />
                  </Box>
                </Box>

                <Box>
                  <Typography gutterBottom>Momentum Threshold [start, end, step]</Typography>
                  <Box className="grid grid-cols-3 gap-2">
                    <TextField size="small" type="number" value={momentumThreshold[0]} onChange={(e) => setMomentumThreshold([Number(e.target.value), momentumThreshold[1], momentumThreshold[2]])} />
                    <TextField size="small" type="number" value={momentumThreshold[1]} onChange={(e) => setMomentumThreshold([momentumThreshold[0], Number(e.target.value), momentumThreshold[2]])} />
                    <TextField size="small" type="number" value={momentumThreshold[2]} onChange={(e) => setMomentumThreshold([momentumThreshold[0], momentumThreshold[1], Number(e.target.value)])} />
                  </Box>
                </Box>

                <Box>
                  <Typography gutterBottom>Position Size [start, end, step]</Typography>
                  <Box className="grid grid-cols-3 gap-2">
                    <TextField size="small" type="number" value={positionSize[0]} onChange={(e) => setPositionSize([Number(e.target.value), positionSize[1], positionSize[2]])} />
                    <TextField size="small" type="number" value={positionSize[1]} onChange={(e) => setPositionSize([positionSize[0], Number(e.target.value), positionSize[2]])} />
                    <TextField size="small" type="number" value={positionSize[2]} onChange={(e) => setPositionSize([positionSize[0], positionSize[1], Number(e.target.value)])} />
                  </Box>
                </Box>

                <TextField label="CSV Path" size="small" value={csvPath} onChange={(e) => setCsvPath(e.target.value)} fullWidth />

                <Button variant="contained" startIcon={<PlayArrow />} onClick={onRun} disabled={loading} fullWidth>
                  {loading ? 'Running Optimization...' : 'Run Optimization'}
                </Button>

                <Box className="grid grid-cols-2 gap-2">
                  <Button variant="outlined" onClick={onRunBayesian} disabled={bayesRunning}>
                    {bayesRunning ? 'Running Bayesian...' : 'Bayesian Optimize'}
                  </Button>
                  <Button variant="outlined" onClick={onWalkForward} disabled={bayesRunning}>
                    {bayesRunning ? 'Running WFO...' : 'Walk-Forward Test'}
                  </Button>
                </Box>

                {loading && <LinearProgress />}
              </Box>
            </Paper>
          </motion.div>
        </Grid>

        <Grid item xs={12} md={6}>
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.4, delay: 0.1 }}>
            <Paper className="card p-5">
              <Typography variant="h6" gutterBottom>Results</Typography>
              {bestParams ? (
                <Box className="space-y-3">
                  <Box className="kpi">
                    <Typography variant="body2" color="text.secondary">Best Sharpe</Typography>
                    <Typography variant="h4" className="text-brand-blue">{bestSharpe?.toFixed(3)}</Typography>
                  </Box>
                  <Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>Best Parameters</Typography>
                    <Box className="flex flex-wrap gap-2">
                      {Object.entries(bestParams).map(([k, v]) => (
                        <Chip key={k} label={`${k}: ${v}`} color="primary" variant="outlined" />
                      ))}
                    </Box>
                  </Box>
                </Box>
              ) : (
                <Typography color="text.secondary">Run an optimization to see results.</Typography>
              )}

              {wfSummary && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle1">Walk-Forward Summary</Typography>
                  <Box className="flex flex-wrap gap-2 mt-2">
                    <Chip label={`Agg Sharpe: ${wfSummary.aggregated_sharpe.toFixed(2)}`} />
                    <Chip label={`Agg Return: ${(wfSummary.aggregated_return * 100).toFixed(1)}%`} />
                    <Chip label={`Segments: ${wfSummary.segments.length}`} />
                  </Box>
                </Box>
              )}
            </Paper>
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Optimization;


