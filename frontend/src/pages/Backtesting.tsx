import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Alert,
  Chip,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Switch,
  FormControlLabel,
  Divider,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  CloudUpload,
  PlayArrow,
  Download,
  Assessment,
  ExpandMore,
  Search,
  Settings,
  TrendingUp,
  Speed,
  Warning,
  Assessment as AssessmentIcon,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';

interface BacktestResults {
  symbol?: string;
  startDate?: string;
  endDate?: string;
  dataPoints?: number;
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  totalTrades: number;
  winRate: number;
  profitFactor: number;
  sortinoRatio?: number;
  calmarRatio?: number;
  equityData: Array<{ date: string; value: number }>;
  signals: Array<{ date: string; signal: string; price: number }>;
  trades?: Array<{ entryDate: string; exitDate: string; entryPrice: number; exitPrice: number; pnl: number; return: number }>;
}

interface StockOption {
  symbol: string;
  name: string;
}

const Backtesting: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<BacktestResults | null>(null);
  
  // Custom backtest options
  const [useCustomBacktest, setUseCustomBacktest] = useState(false);
  const [selectedStock, setSelectedStock] = useState('AAPL');
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-01-01');
  const [interval, setInterval] = useState('1d');
  const [searchQuery, setSearchQuery] = useState('');
  const [stockOptions, setStockOptions] = useState<StockOption[]>([]);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  
  // Advanced settings
  const [config, setConfig] = useState({
    initialCapital: 100000,
    positionSize: 0.1,
    momentumLookback: 20,
    momentumThreshold: 0.02,
    meanReversionLookback: 50,
    meanReversionThreshold: 0.01,
    transactionCost: 1.0,
    stopLoss: 50.0,
    takeProfit: 100.0,
  });

  // Common stock options
  const commonStocks: StockOption[] = [
    { symbol: 'AAPL', name: 'Apple Inc.' },
    { symbol: 'MSFT', name: 'Microsoft Corporation' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.' },
    { symbol: 'TSLA', name: 'Tesla Inc.' },
    { symbol: 'META', name: 'Meta Platforms Inc.' },
    { symbol: 'NVDA', name: 'NVIDIA Corporation' },
    { symbol: 'NFLX', name: 'Netflix Inc.' },
    { symbol: 'AMD', name: 'Advanced Micro Devices' },
    { symbol: 'INTC', name: 'Intel Corporation' },
    { symbol: 'SPY', name: 'SPDR S&P 500 ETF' },
    { symbol: 'QQQ', name: 'Invesco QQQ Trust' },
  ];

  useEffect(() => {
    setStockOptions(commonStocks);
  }, []);

  const onDrop = (acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
        setFile(file);
        setUseCustomBacktest(false);
        toast.success('File uploaded successfully!');
      } else {
        toast.error('Please upload a CSV file');
      }
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
    },
    multiple: false,
  });

  const searchStocks = async (query: string) => {
    if (query.length < 1) {
      setStockOptions(commonStocks);
      return;
    }

    try {
      const response = await axios.get(`/api/stocks/search?q=${query}`);
      const stocks = response.data.stocks || [];
      setStockOptions(stocks.map((symbol: string) => ({ symbol, name: symbol })));
    } catch (error) {
      // Fallback to filtering common stocks
      const filtered = commonStocks.filter(stock => 
        stock.symbol.toLowerCase().includes(query.toLowerCase())
      );
      setStockOptions(filtered);
    }
  };

  const runBacktest = async () => {
    if (useCustomBacktest) {
      await runCustomBacktest();
    } else {
      await runFileBacktest();
    }
  };

  const runFileBacktest = async () => {
    if (!file) {
      toast.error('Please upload a file first');
      return;
    }

    try {
      setLoading(true);
      const form = new FormData();
      form.append('file', file);
      const { data } = await axios.post('/api/backtest', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResults(data as BacktestResults);
      toast.success('Backtest completed successfully!');
    } catch (e: any) {
      toast.error(e?.response?.data?.error || 'Backtest failed');
    } finally {
      setLoading(false);
    }
  };

  const runCustomBacktest = async () => {
    try {
      setLoading(true);
      const requestData = {
        symbol: selectedStock,
        startDate,
        endDate,
        interval,
        config: showAdvancedSettings ? config : undefined,
      };

      const { data } = await axios.post('/api/backtest/custom', requestData);
      setResults(data as BacktestResults);
      toast.success('Custom backtest completed successfully!');
    } catch (e: any) {
      toast.error(e?.response?.data?.error || 'Custom backtest failed');
    } finally {
      setLoading(false);
    }
  };

  const downloadResults = async () => {
    if (!results) return;
    try {
      const { data } = await axios.post('/api/data/download', { type: 'backtest' }, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'backtest_results.csv');
      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (e) {
      toast.error('Failed to download');
    }
  };

  const handleConfigChange = (field: string, value: number) => {
    setConfig(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Backtesting
      </Typography>

      <Grid container spacing={3}>
        {/* Configuration Panel */}
        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Paper sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <Typography variant="h6">
                  Backtest Configuration
                </Typography>
                <Box sx={{ ml: 'auto' }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={useCustomBacktest}
                        onChange={(e) => setUseCustomBacktest(e.target.checked)}
                      />
                    }
                    label="Use Custom Stock"
                  />
                </Box>
              </Box>

              {useCustomBacktest ? (
                // Custom Stock Backtest
                <Box>
                  {/* Stock Selection */}
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Stock Symbol</InputLabel>
                    <Select
                      value={selectedStock}
                      onChange={(e) => setSelectedStock(e.target.value)}
                      label="Stock Symbol"
                    >
                      {stockOptions.map((stock) => (
                        <MenuItem key={stock.symbol} value={stock.symbol}>
                          {stock.symbol} - {stock.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>

                  {/* Stock Search */}
                  <TextField
                    fullWidth
                    label="Search Stocks"
                    value={searchQuery}
                    onChange={(e) => {
                      setSearchQuery(e.target.value);
                      searchStocks(e.target.value);
                    }}
                    sx={{ mb: 2 }}
                    InputProps={{
                      endAdornment: (
                        <IconButton>
                          <Search />
                        </IconButton>
                      ),
                    }}
                  />

                  {/* Date Range */}
                  <Grid container spacing={2} sx={{ mb: 2 }}>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="Start Date"
                        type="date"
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                        InputLabelProps={{ shrink: true }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="End Date"
                        type="date"
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                        InputLabelProps={{ shrink: true }}
                      />
                    </Grid>
                  </Grid>

                  {/* Interval */}
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Data Interval</InputLabel>
                    <Select
                      value={interval}
                      onChange={(e) => setInterval(e.target.value)}
                      label="Data Interval"
                    >
                      <MenuItem value="1d">Daily</MenuItem>
                      <MenuItem value="1h">Hourly</MenuItem>
                      <MenuItem value="5m">5 Minutes</MenuItem>
                      <MenuItem value="1m">1 Minute</MenuItem>
                    </Select>
                  </FormControl>

                  {/* Advanced Settings */}
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Settings sx={{ mr: 1 }} />
                        <Typography>Advanced Settings</Typography>
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <TextField
                            fullWidth
                            label="Initial Capital ($)"
                            type="number"
                            value={config.initialCapital}
                            onChange={(e) => handleConfigChange('initialCapital', Number(e.target.value))}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <TextField
                            fullWidth
                            label="Position Size (%)"
                            type="number"
                            value={config.positionSize * 100}
                            onChange={(e) => handleConfigChange('positionSize', Number(e.target.value) / 100)}
                            inputProps={{ step: 0.1 }}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <TextField
                            fullWidth
                            label="Momentum Lookback"
                            type="number"
                            value={config.momentumLookback}
                            onChange={(e) => handleConfigChange('momentumLookback', Number(e.target.value))}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <TextField
                            fullWidth
                            label="Momentum Threshold (%)"
                            type="number"
                            value={config.momentumThreshold * 100}
                            onChange={(e) => handleConfigChange('momentumThreshold', Number(e.target.value) / 100)}
                            inputProps={{ step: 0.1 }}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <TextField
                            fullWidth
                            label="Transaction Cost (bps)"
                            type="number"
                            value={config.transactionCost}
                            onChange={(e) => handleConfigChange('transactionCost', Number(e.target.value))}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <TextField
                            fullWidth
                            label="Stop Loss (bps)"
                            type="number"
                            value={config.stopLoss}
                            onChange={(e) => handleConfigChange('stopLoss', Number(e.target.value))}
                          />
                        </Grid>
                      </Grid>
                    </AccordionDetails>
                  </Accordion>
                </Box>
              ) : (
                // File Upload
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Upload Data File
                  </Typography>
                  
                  <Box
                    {...getRootProps()}
                    sx={{
                      border: '2px dashed',
                      borderColor: isDragActive ? 'primary.main' : 'grey.500',
                      borderRadius: 2,
                      p: 3,
                      textAlign: 'center',
                      cursor: 'pointer',
                      '&:hover': {
                        borderColor: 'primary.main',
                      },
                    }}
                  >
                    <input {...getInputProps()} />
                    <CloudUpload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                    <Typography variant="h6" gutterBottom>
                      {isDragActive ? 'Drop the file here' : 'Drag & drop a CSV file here'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      or click to select a file
                    </Typography>
                    {file && (
                      <Alert severity="success" sx={{ mt: 2 }}>
                        File uploaded: {file.name}
                      </Alert>
                    )}
                  </Box>
                </Box>
              )}

              <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<PlayArrow />}
                  onClick={runBacktest}
                  disabled={loading || (!useCustomBacktest && !file)}
                  fullWidth
                >
                  {loading ? 'Running Backtest...' : 'Run Backtest'}
                </Button>
              </Box>

              {loading && (
                <Box sx={{ mt: 2 }}>
                  <LinearProgress />
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    {useCustomBacktest 
                      ? `Processing ${selectedStock} data and running backtest...`
                      : 'Processing data and running backtest...'
                    }
                  </Typography>
                </Box>
              )}
            </Paper>
          </motion.div>
        </Grid>

        {/* Results */}
        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Backtest Results
                {results?.symbol && (
                  <Chip 
                    label={`${results.symbol} (${results.startDate} - ${results.endDate})`}
                    size="small"
                    sx={{ ml: 1 }}
                  />
                )}
              </Typography>

              {results ? (
                <Box>
                  {/* Key Metrics */}
                  <Grid container spacing={2} sx={{ mb: 3 }}>
                    <Grid item xs={6}>
                      <Card>
                        <CardContent>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <TrendingUp color="primary" />
                            <Typography variant="h6" sx={{ ml: 1 }}>
                              {results.totalReturn.toFixed(2)}%
                            </Typography>
                          </Box>
                          <Typography variant="body2" color="text.secondary">
                            Total Return
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>

                    <Grid item xs={6}>
                      <Card>
                        <CardContent>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <Speed color="secondary" />
                            <Typography variant="h6" sx={{ ml: 1 }}>
                              {results.sharpeRatio.toFixed(3)}
                            </Typography>
                          </Box>
                          <Typography variant="body2" color="text.secondary">
                            Sharpe Ratio
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>

                    <Grid item xs={6}>
                      <Card>
                        <CardContent>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <Warning color="error" />
                            <Typography variant="h6" sx={{ ml: 1 }}>
                              {results.maxDrawdown.toFixed(2)}%
                            </Typography>
                          </Box>
                          <Typography variant="body2" color="text.secondary">
                            Max Drawdown
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>

                    <Grid item xs={6}>
                      <Card>
                        <CardContent>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <AssessmentIcon color="info" />
                            <Typography variant="h6" sx={{ ml: 1 }}>
                              {results.winRate.toFixed(1)}%
                            </Typography>
                          </Box>
                          <Typography variant="body2" color="text.secondary">
                            Win Rate
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>

                  {/* Additional Metrics */}
                  <Grid container spacing={2} sx={{ mb: 3 }}>
                    <Grid item xs={4}>
                      <Typography variant="body2" color="text.secondary">
                        Total Trades
                      </Typography>
                      <Typography variant="h6">
                        {results.totalTrades}
                      </Typography>
                    </Grid>
                    <Grid item xs={4}>
                      <Typography variant="body2" color="text.secondary">
                        Profit Factor
                      </Typography>
                      <Typography variant="h6">
                        {results.profitFactor.toFixed(2)}
                      </Typography>
                    </Grid>
                    <Grid item xs={4}>
                      <Typography variant="body2" color="text.secondary">
                        Data Points
                      </Typography>
                      <Typography variant="h6">
                        {results.dataPoints || 'N/A'}
                      </Typography>
                    </Grid>
                  </Grid>

                  {/* Equity Curve */}
                  {results.equityData.length > 0 && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle1" gutterBottom>
                        Equity Curve
                      </Typography>
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={results.equityData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" />
                          <YAxis />
                          <RechartsTooltip />
                          <Line
                            type="monotone"
                            dataKey="value"
                            stroke="#2196f3"
                            strokeWidth={2}
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </Box>
                  )}

                  {/* Signals */}
                  {results.signals.length > 0 && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle1" gutterBottom>
                        Recent Signals
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        {results.signals.slice(0, 8).map((signal, index) => (
                          <Chip
                            key={index}
                            label={`${signal.signal} @ $${signal.price.toFixed(2)}`}
                            color={signal.signal === 'BUY' ? 'success' : 'error'}
                            size="small"
                          />
                        ))}
                      </Box>
                    </Box>
                  )}

                  <Button
                    variant="outlined"
                    startIcon={<Download />}
                    onClick={downloadResults}
                    fullWidth
                  >
                    Download Results
                  </Button>
                </Box>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Assessment sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="body1" color="text.secondary">
                    {useCustomBacktest 
                      ? 'Configure settings and run backtest to see results'
                      : 'Upload a file and run backtest to see results'
                    }
                  </Typography>
                </Box>
              )}
            </Paper>
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Backtesting;

