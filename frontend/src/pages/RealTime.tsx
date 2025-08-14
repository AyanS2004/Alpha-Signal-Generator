import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  Chip,
  Switch,
  FormControlLabel,
  Alert,
  LinearProgress,
  TextField,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  TrendingUp,
  TrendingDown,
  ShowChart,
  SignalCellular4Bar,
  SignalCellularConnectedNoInternet0Bar,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import axios from 'axios';

interface LiveData {
  timestamp: string;
  price: number;
  volume: number;
  signal: string;
  confidence: number;
}

const RealTime: React.FC = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [liveData, setLiveData] = useState<LiveData[]>([]);
  const [currentSignal, setCurrentSignal] = useState<string>('HOLD');
  const [currentPrice, setCurrentPrice] = useState<number>(150.0);
  const [autoTrading, setAutoTrading] = useState(false);
  const [symbol, setSymbol] = useState<string>('AAPL');

  const symbolToName: Record<string, string> = {
    AAPL: 'Apple Inc.',
    MSFT: 'Microsoft Corp.',
    GOOGL: 'Alphabet Inc.',
    AMZN: 'Amazon.com Inc.',
    TSLA: 'Tesla Inc.',
    META: 'Meta Platforms Inc.',
  };
  const companyName = symbolToName[symbol.toUpperCase()] || '';

  const randn = () => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isConnected) {
      interval = setInterval(async () => {
        try {
          const { data } = await axios.get('/api/realtime/latest');
          if (data?.connected && data?.tick) {
            const t = data.tick as any;
            const price = Number(((t.price ?? t.close ?? currentPrice) as number).toFixed(2));
            const vol = (t.size ?? t.volume ?? Math.floor(Math.random() * 10000) + 1000) as number;
            const newData: LiveData = {
              timestamp: new Date().toLocaleTimeString(),
              price,
              volume: vol,
              signal: currentSignal,
              confidence: 0.75,
            };
            setLiveData(prev => [...prev.slice(-19), newData]);
            setCurrentPrice(price);
          }
        } catch {}
      }, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isConnected, currentPrice, currentSignal]);

  const startFeed = async () => {
    try {
      await axios.post('/api/realtime/start', { symbol: symbol.toUpperCase(), provider: 'yahoo' });
      setIsConnected(true);
      setLiveData([]);
      toast.success('Real-time feed started!');
    } catch (e) {
      toast.error('Failed to start real-time feed');
    }
  };

  const stopFeed = async () => {
    try {
      await axios.post('/api/realtime/stop');
    } catch {}
    setIsConnected(false);
    toast.error('Real-time feed stopped!');
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BUY':
        return 'success';
      case 'SELL':
        return 'error';
      default:
        return 'warning';
    }
  };

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case 'BUY':
        return <TrendingUp />;
      case 'SELL':
        return <TrendingDown />;
      default:
        return <ShowChart />;
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Real-time Trading
      </Typography>

      <Grid container spacing={3}>
        {/* Connection Status */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Paper sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Connection Status
                </Typography>
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                  <TextField
                    size="small"
                    label="Symbol"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                    sx={{ width: 120 }}
                    disabled={isConnected}
                  />
                  <Button
                    variant="contained"
                    startIcon={isConnected ? <Stop /> : <PlayArrow />}
                    onClick={isConnected ? stopFeed : startFeed}
                    color={isConnected ? 'error' : 'success'}
                  >
                    {isConnected ? 'Stop Feed' : 'Start Feed'}
                  </Button>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={autoTrading}
                        onChange={(e) => setAutoTrading(e.target.checked)}
                        disabled={!isConnected}
                      />
                    }
                    label="Auto Trading"
                  />
                </Box>
              </Box>

              <Alert severity={isConnected ? 'success' : 'warning'} sx={{ mb: 2 }}>
                {isConnected ? (
                  <>
                    <SignalCellular4Bar sx={{ mr: 1 }} />
                    Connected to live data feed
                  </>
                ) : (
                  <>
                    <SignalCellularConnectedNoInternet0Bar sx={{ mr: 1 }} />
                    Disconnected from data feed
                  </>
                )}
              </Alert>

              {autoTrading && (
                <Alert severity="info">
                  Auto trading is enabled. Trades will be executed automatically based on signals.
                </Alert>
              )}
            </Paper>
          </motion.div>
        </Grid>

        {/* Current Signal */}
        <Grid item xs={12} md={4}>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {companyName ? `${companyName} (${symbol})` : symbol} — Current Signal
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  {getSignalIcon(currentSignal)}
                  <Chip
                    label={currentSignal}
                    color={getSignalColor(currentSignal) as any}
                    sx={{ ml: 1 }}
                  />
                </Box>
                <Typography variant="h4" color="primary">
                  ${currentPrice.toFixed(2)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Last updated: {new Date().toLocaleTimeString()}
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Price Chart */}
        <Grid item xs={12} md={8}>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Paper sx={{ p: 3, height: 320 }}>
              <Typography variant="h6" gutterBottom>
                {companyName ? `${companyName} (${symbol})` : symbol} — Live Price
              </Typography>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={liveData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis domain={[
                    (dataMin: number) => (dataMin ? Number((dataMin - 0.5).toFixed(2)) : 'auto'),
                    (dataMax: number) => (dataMax ? Number((dataMax + 0.5).toFixed(2)) : 'auto'),
                  ]} />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="price"
                    stroke="#2196f3"
                    strokeWidth={2}
                    dot={(props: any) => {
                      const { cx, cy, payload } = props;
                      if (!payload || payload.signal === 'HOLD') return null;
                      const isBuy = payload.signal === 'BUY';
                      const color = isBuy ? '#22c55e' : '#ef4444';
                      return (
                        <svg x={cx - 6} y={cy - 6} width={12} height={12}>
                          {isBuy ? (
                            <polygon points="6,0 12,12 0,12" fill={color} />
                          ) : (
                            <polygon points="0,0 12,0 6,12" fill={color} />
                          )}
                        </svg>
                      );
                    }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Paper>
          </motion.div>
        </Grid>

        {/* Recent Signals */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Recent Signals
              </Typography>
              <Grid container spacing={2}>
                {liveData.slice(-10).reverse().map((data, index) => (
                  <Grid item xs={12} sm={6} md={4} key={index}>
                    <Card variant="outlined">
                      <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Box>
                            <Typography variant="body2" color="text.secondary">
                              {data.timestamp}
                            </Typography>
                            <Typography variant="h6">
                              ${data.price.toFixed(2)}
                            </Typography>
                          </Box>
                          <Chip
                            label={data.signal}
                            color={getSignalColor(data.signal) as any}
                            size="small"
                          />
                        </Box>
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                          Confidence: {(data.confidence * 100).toFixed(1)}%
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  );
};

export default RealTime;

