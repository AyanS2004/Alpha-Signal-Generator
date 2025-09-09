import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Paper,
  Chip,
  LinearProgress,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Alert,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  ShowChart,
  Speed,
  Assessment,
  Warning,
  AccountBalance,
  Analytics,
  Refresh,
  PlayArrow,
  Star,
  StarBorder,
  MoreVert,
  ArrowUpward,
  ArrowDownward,
  TrendingFlat,
  ShowChart as ChartIcon,
  Assessment as AnalyticsIcon,
  AccountBalance as PortfolioIcon,
  Timeline as TimelineIcon,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area } from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

interface DashboardData {
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  totalTrades: number;
  winRate: number;
  currentSignal: string;
  equityData: Array<{ date: string; value: number }>;
  benchmarkData: Array<{ date: string; value: number }>;
  portfolioValue: number;
  dailyPnL: number;
  openPositions: number;
  riskMetrics: {
    var: number;
    beta: number;
    alpha: number;
    volatility: number;
  };
}

interface StockData {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  pe: number;
  sector: string;
  isFavorite: boolean;
}

interface PortfolioPosition {
  symbol: string;
  name: string;
  shares: number;
  avgPrice: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  weight: number;
}

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [selectedStock, setSelectedStock] = useState('AAPL');
  const [timeframe, setTimeframe] = useState('1M');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [activeTab, setActiveTab] = useState(0);
  
  const [dashboardData, setDashboardData] = useState<DashboardData>({
    totalReturn: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    totalTrades: 0,
    winRate: 0,
    currentSignal: 'HOLD',
    equityData: [],
    benchmarkData: [],
    portfolioValue: 0,
    dailyPnL: 0,
    openPositions: 0,
    riskMetrics: {
      var: 0,
      beta: 0,
      alpha: 0,
      volatility: 0,
    },
  });

  const [watchlist, setWatchlist] = useState<StockData[]>([
    { symbol: 'AAPL', name: 'Apple Inc.', price: 175.43, change: 2.15, changePercent: 1.24, volume: 45678900, marketCap: 2750000000000, pe: 28.5, sector: 'Technology', isFavorite: true },
    { symbol: 'MSFT', name: 'Microsoft Corporation', price: 338.11, change: -1.23, changePercent: -0.36, volume: 23456700, marketCap: 2510000000000, pe: 32.1, sector: 'Technology', isFavorite: true },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 142.56, change: 0.89, changePercent: 0.63, volume: 34567800, marketCap: 1790000000000, pe: 25.8, sector: 'Technology', isFavorite: false },
    { symbol: 'TSLA', name: 'Tesla Inc.', price: 248.42, change: 5.67, changePercent: 2.34, volume: 56789000, marketCap: 789000000000, pe: 45.2, sector: 'Automotive', isFavorite: true },
    { symbol: 'NVDA', name: 'NVIDIA Corporation', price: 485.09, change: 12.34, changePercent: 2.61, volume: 67890100, marketCap: 1198000000000, pe: 38.7, sector: 'Technology', isFavorite: false },
  ]);

  const [portfolio, setPortfolio] = useState<PortfolioPosition[]>([]);
  const [newSymbol, setNewSymbol] = useState('AAPL');
  const [newShares, setNewShares] = useState<number>(10);
  const [signalsData, setSignalsData] = useState<any>({
    current_signal: null,
    recent_signals: [],
    total_signals: 0
  });

  // Dynamic performance data from dashboard API
  const performanceData = React.useMemo(() => {
    if (!dashboardData.equityData || !dashboardData.benchmarkData) {
      return [];
    }
    
    const portfolio = dashboardData.equityData;
    const benchmark = dashboardData.benchmarkData;
    const maxLength = Math.max(portfolio.length, benchmark.length);
    
    return Array.from({ length: maxLength }, (_, i) => ({
      date: portfolio[i]?.date || benchmark[i]?.date || `Day ${i}`,
      portfolio: portfolio[i]?.value || 0,
      benchmark: benchmark[i]?.value || 0,
      alpha: (portfolio[i]?.value || 0) - (benchmark[i]?.value || 0)
    }));
  }, [dashboardData.equityData, dashboardData.benchmarkData]);


  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  useEffect(() => {
    const fetchData = async () => {
      try {
        const { data } = await axios.get('/api/dashboard');
        if (data && !data.error) {
          setDashboardData({
            totalReturn: data.totalReturn ?? 0,
            sharpeRatio: data.sharpeRatio ?? 0,
            maxDrawdown: data.maxDrawdown ?? 0,
            totalTrades: data.totalTrades ?? 0,
            winRate: data.winRate ?? 0,
            currentSignal: data.currentSignal ?? 'HOLD',
            equityData: data.equityData ?? [],
            benchmarkData: data.benchmarkData ?? [],
            portfolioValue: data.portfolioValue ?? 0,
            dailyPnL: data.dailyPnL ?? 0,
            openPositions: data.openPositions ?? 0,
            riskMetrics: data.riskMetrics ?? { var: 0, beta: 0, alpha: 0, volatility: 0 },
          });
        }
        // Load portfolio
        try {
          const pf = await axios.get('/api/portfolio');
          const positions = (pf.data.positions || []) as Array<{symbol: string; shares: number; avgPrice: number}>;
          // Enrich with simple pricing via dashboard equityData latest; fallback to avgPrice
          const mapped: PortfolioPosition[] = positions.map(p => {
            const currentPrice = p.avgPrice; // backend can be extended to pass live
            const marketValue = currentPrice * p.shares;
            const unrealizedPnL = (currentPrice - p.avgPrice) * p.shares;
            const unrealizedPnLPercent = p.avgPrice ? ((currentPrice - p.avgPrice) / p.avgPrice) * 100 : 0;
            return {
              symbol: p.symbol,
              name: p.symbol,
              shares: p.shares,
              avgPrice: p.avgPrice,
              currentPrice,
              marketValue,
              unrealizedPnL,
              unrealizedPnLPercent,
              weight: 0,
            };
          });
          const totalMV = mapped.reduce((s, m) => s + m.marketValue, 0) || 1;
          mapped.forEach(m => m.weight = (m.marketValue / totalMV) * 100);
          setPortfolio(mapped);
        } catch {}
        
        // Load signals data
        try {
          const signals = await axios.get('/api/signals');
          if (signals.data && !signals.data.error) {
            setSignalsData(signals.data);
          } else {
            // Set default empty signals data
            setSignalsData({
              current_signal: null,
              recent_signals: [],
              total_signals: 0
            });
          }
        } catch {
          // Set default empty signals data on error
          setSignalsData({
            current_signal: null,
            recent_signals: [],
            total_signals: 0
          });
        }
      } catch (e) {
        // Fallback to zeros if no backtest yet
        setDashboardData((prev) => ({ ...prev, equityData: [] }));
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    const interval = setInterval(() => {
      if (autoRefresh) fetchData();
    }, 30000);
    return () => clearInterval(interval);
  }, [autoRefresh, dashboardData.equityData]);

  const addToPortfolio = async () => {
    try {
      await axios.post('/api/portfolio/add', { symbol: newSymbol, shares: newShares });
      const pf = await axios.get('/api/portfolio');
      const positions = (pf.data.positions || []) as Array<{symbol: string; shares: number; avgPrice: number}>;
      const mapped: PortfolioPosition[] = positions.map(p => {
        const currentPrice = p.avgPrice;
        const marketValue = currentPrice * p.shares;
        const unrealizedPnL = (currentPrice - p.avgPrice) * p.shares;
        const unrealizedPnLPercent = p.avgPrice ? ((currentPrice - p.avgPrice) / p.avgPrice) * 100 : 0;
        return { symbol: p.symbol, name: p.symbol, shares: p.shares, avgPrice: p.avgPrice, currentPrice, marketValue, unrealizedPnL, unrealizedPnLPercent, weight: 0 };
      });
      const totalMV = mapped.reduce((s, m) => s + m.marketValue, 0) || 1;
      mapped.forEach(m => m.weight = (m.marketValue / totalMV) * 100);
      setPortfolio(mapped);
    } catch {}
  };

  const removeFromPortfolio = async (symbol: string) => {
    try {
      await axios.post('/api/portfolio/remove', { symbol });
      setPortfolio(prev => prev.filter(p => p.symbol !== symbol));
    } catch {}
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BUY': return 'success';
      case 'SELL': return 'error';
      default: return 'warning';
    }
  };

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case 'BUY': return <TrendingUp />;
      case 'SELL': return <TrendingDown />;
      default: return <ShowChart />;
    }
  };

  const getChangeIcon = (change: number) => {
    if (change > 0) return <ArrowUpward color="success" />;
    if (change < 0) return <ArrowDownward color="error" />;
    return <TrendingFlat color="action" />;
  };

  const toggleFavorite = (symbol: string) => {
    setWatchlist(prev => prev.map(stock => 
      stock.symbol === symbol ? { ...stock, isFavorite: !stock.isFavorite } : stock
    ));
  };

  const runQuickBacktest = async () => {
    try {
      setLoading(true);
      // Simulate backtest
      await new Promise(resolve => setTimeout(resolve, 2000));
      setDashboardData(prev => ({
        ...prev,
        totalReturn: Math.random() * 30 + 5,
        sharpeRatio: Math.random() * 2 + 0.5,
        maxDrawdown: -(Math.random() * 15 + 5),
        winRate: Math.random() * 40 + 50,
      }));
    } catch (e) {
      console.error('Error running backtest:', e);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box sx={{ width: '100%', p: 3 }}>
        <LinearProgress />
        <Typography variant="h6" sx={{ mt: 2, textAlign: 'center' }}>
          Loading Alpha Signal Dashboard...
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold' }}>
            Alpha Signal Dashboard
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Real-time portfolio analytics and algorithmic trading insights
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
            }
            label="Auto Refresh"
          />
          <IconButton onClick={() => window.location.reload()}>
            <Refresh />
          </IconButton>
          <Button
            variant="contained"
            startIcon={<PlayArrow />}
            onClick={runQuickBacktest}
            disabled={loading}
          >
            Quick Backtest
          </Button>
        </Box>
      </Box>

      {/* Portfolio Overview Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
              <CardContent sx={{ color: 'white' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <AccountBalance sx={{ mr: 1 }} />
                  <Typography variant="h6">Portfolio Value</Typography>
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 'bold', mb: 1 }}>
                  ${dashboardData.portfolioValue.toLocaleString()}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  {getChangeIcon(dashboardData.dailyPnL)}
                  <Typography variant="body2" sx={{ ml: 0.5 }}>
                    +${dashboardData.dailyPnL.toLocaleString()} today
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' }}>
              <CardContent sx={{ color: 'white' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <TrendingUp sx={{ mr: 1 }} />
                  <Typography variant="h6">Total Return</Typography>
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 'bold', mb: 1 }}>
                  {dashboardData.totalReturn.toFixed(1)}%
                </Typography>
                <Typography variant="body2">
                  {dashboardData.totalTrades} trades executed
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)' }}>
              <CardContent sx={{ color: 'white' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Speed sx={{ mr: 1 }} />
                  <Typography variant="h6">Sharpe Ratio</Typography>
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 'bold', mb: 1 }}>
                  {dashboardData.sharpeRatio.toFixed(2)}
                </Typography>
                <Typography variant="body2">
                  Risk-adjusted performance
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)' }}>
              <CardContent sx={{ color: 'white' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Assessment sx={{ mr: 1 }} />
                  <Typography variant="h6">Win Rate</Typography>
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 'bold', mb: 1 }}>
                  {dashboardData.winRate.toFixed(1)}%
                </Typography>
                <Typography variant="body2">
                  Profitable trades
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Main Content Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
          <Tab icon={<PortfolioIcon />} label="Portfolio" />
          <Tab icon={<ChartIcon />} label="Analytics" />
          <Tab icon={<TimelineIcon />} label="Signals" />
          <Tab icon={<Assessment />} label="Watchlist" />
          <Tab icon={<AnalyticsIcon />} label="Risk Metrics" />
        </Tabs>
      </Paper>

      <AnimatePresence mode="wait">
        {activeTab === 0 && (
          <motion.div
            key="portfolio"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Grid container spacing={3}>
              {/* Portfolio Performance Chart */}
              <Grid item xs={12} lg={8}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Portfolio Performance vs Benchmark
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={performanceData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <RechartsTooltip />
                        <Area type="monotone" dataKey="portfolio" stroke="#2196f3" fill="#2196f3" fillOpacity={0.3} />
                        <Area type="monotone" dataKey="benchmark" stroke="#ff9800" fill="#ff9800" fillOpacity={0.3} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>

              {/* Sector Allocation (dynamic by portfolio weights) */}
              <Grid item xs={12} lg={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Sector Allocation
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={portfolio.map(p => ({ name: p.symbol, value: p.weight }))}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {portfolio.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <RechartsTooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>

              {/* Add Position */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Add Position</Typography>
                    <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                      <TextField size="small" label="Symbol" value={newSymbol} onChange={(e) => setNewSymbol(e.target.value.toUpperCase())} sx={{ width: 120 }} />
                      <TextField size="small" label="Shares" type="number" value={newShares} onChange={(e) => setNewShares(Number(e.target.value))} sx={{ width: 120 }} />
                      <Button variant="contained" onClick={addToPortfolio}>Add</Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              {/* Portfolio Holdings Table */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Portfolio Holdings
                    </Typography>
                    <TableContainer>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Symbol</TableCell>
                            <TableCell>Name</TableCell>
                            <TableCell align="right">Shares</TableCell>
                            <TableCell align="right">Avg Price</TableCell>
                            <TableCell align="right">Current Price</TableCell>
                            <TableCell align="right">Market Value</TableCell>
                            <TableCell align="right">Unrealized P&L</TableCell>
                            <TableCell align="right">Weight</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {portfolio.map((position) => (
                            <TableRow key={position.symbol}>
                              <TableCell>
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                  <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                    {position.symbol}
                                  </Typography>
                                </Box>
                              </TableCell>
                              <TableCell>{position.name}</TableCell>
                              <TableCell align="right">{position.shares}</TableCell>
                              <TableCell align="right">${position.avgPrice.toFixed(2)}</TableCell>
                              <TableCell align="right">${position.currentPrice.toFixed(2)}</TableCell>
                              <TableCell align="right">${position.marketValue.toLocaleString()}</TableCell>
                              <TableCell align="right">
                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                  {getChangeIcon(position.unrealizedPnL)}
                                  <Typography
                                    variant="body2"
                                    color={position.unrealizedPnL >= 0 ? 'success.main' : 'error.main'}
                                  >
                                    ${position.unrealizedPnL.toLocaleString()} ({position.unrealizedPnLPercent.toFixed(2)}%)
                                  </Typography>
                                </Box>
                              </TableCell>
                              <TableCell align="right">{position.weight.toFixed(1)}%</TableCell>
                              <TableCell align="center">
                                <Button size="small" color="error" onClick={() => removeFromPortfolio(position.symbol)}>Remove</Button>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </motion.div>
        )}

        {activeTab === 1 && (
          <motion.div
            key="analytics"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Grid container spacing={3}>
              {/* Stock Selection and Quick Analysis */}
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Quick Analysis
                    </Typography>
                    <FormControl fullWidth sx={{ mb: 2 }}>
                      <InputLabel>Select Stock</InputLabel>
                      <Select
                        value={selectedStock}
                        onChange={(e) => setSelectedStock(e.target.value)}
                        label="Select Stock"
                      >
                        {watchlist.map((stock) => (
                          <MenuItem key={stock.symbol} value={stock.symbol}>
                            {stock.symbol} - {stock.name}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <FormControl fullWidth sx={{ mb: 2 }}>
                      <InputLabel>Timeframe</InputLabel>
                      <Select
                        value={timeframe}
                        onChange={(e) => setTimeframe(e.target.value)}
                        label="Timeframe"
                      >
                        <MenuItem value="1D">1 Day</MenuItem>
                        <MenuItem value="1W">1 Week</MenuItem>
                        <MenuItem value="1M">1 Month</MenuItem>
                        <MenuItem value="3M">3 Months</MenuItem>
                        <MenuItem value="1Y">1 Year</MenuItem>
                      </Select>
                    </FormControl>
                    <Button
                      variant="contained"
                      fullWidth
                      startIcon={<Analytics />}
                      onClick={runQuickBacktest}
                    >
                      Analyze {selectedStock}
                    </Button>
                  </CardContent>
                </Card>
              </Grid>

              {/* Current Signal */}
              <Grid item xs={12} md={8}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="h6">
                        Current Signal for {selectedStock}
                      </Typography>
                      <Chip
                        icon={getSignalIcon(dashboardData.currentSignal)}
                        label={dashboardData.currentSignal}
                        color={getSignalColor(dashboardData.currentSignal) as any}
                        variant="filled"
                      />
                    </Box>
                    <Alert severity="info" sx={{ mb: 2 }}>
                      <Typography variant="body2">
                        Based on momentum and mean reversion analysis, {selectedStock} shows a {dashboardData.currentSignal.toLowerCase()} signal.
                        {signalsData?.current_signal ? ` Confidence level: ${(signalsData.current_signal.confidence * 100).toFixed(1)}%` : ' Confidence level: N/A'}
                      </Typography>
                    </Alert>
                    <Grid container spacing={2}>
                      <Grid item xs={6} md={3}>
                        <Typography variant="body2" color="text.secondary">Momentum Score</Typography>
                        <Typography variant="h6" color="primary">
                          {signalsData?.current_signal?.momentum_score?.toFixed(3) || 'N/A'}
                        </Typography>
                      </Grid>
                      <Grid item xs={6} md={3}>
                        <Typography variant="body2" color="text.secondary">Volatility</Typography>
                        <Typography variant="h6" color="secondary">
                          {dashboardData.riskMetrics?.volatility?.toFixed(1) || 'N/A'}%
                        </Typography>
                      </Grid>
                      <Grid item xs={6} md={3}>
                        <Typography variant="body2" color="text.secondary">RSI</Typography>
                        <Typography variant="h6" color="info.main">
                          {signalsData?.current_signal?.rsi?.toFixed(1) || 'N/A'}
                        </Typography>
                      </Grid>
                      <Grid item xs={6} md={3}>
                        <Typography variant="body2" color="text.secondary">Volume Ratio</Typography>
                        <Typography variant="h6" color="success.main">
                          {signalsData?.current_signal?.volume_ratio ? `${(signalsData.current_signal.volume_ratio * 100 - 100).toFixed(1)}%` : 'N/A'}
                        </Typography>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>

              {/* Performance Metrics */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Performance Metrics
                    </Typography>
                    <Grid container spacing={3}>
                      <Grid item xs={12} md={6}>
                        <ResponsiveContainer width="100%" height={200}>
                          <BarChart data={performanceData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="date" />
                            <YAxis />
                            <RechartsTooltip />
                            <Bar dataKey="alpha" fill="#8884d8" />
                          </BarChart>
                        </ResponsiveContainer>
                        <Typography variant="body2" color="text.secondary" align="center">
                          Alpha Generation Over Time
                        </Typography>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <ResponsiveContainer width="100%" height={200}>
                          <LineChart data={dashboardData.equityData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="date" />
                            <YAxis />
                            <RechartsTooltip />
                            <Line type="monotone" dataKey="value" stroke="#2196f3" strokeWidth={2} />
                          </LineChart>
                        </ResponsiveContainer>
                        <Typography variant="body2" color="text.secondary" align="center">
                          Equity Curve
                        </Typography>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </motion.div>
        )}

        {activeTab === 2 && (
          <motion.div
            key="signals"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Grid container spacing={3}>
              {/* Current Signal Analysis */}
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Current Signal Analysis
                    </Typography>
                    {signalsData?.current_signal && signalsData.total_signals > 0 && signalsData.current_signal.price > 0 ? (
                      <>
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                          <Typography variant="h4">
                            {signalsData.current_signal.signal}
                          </Typography>
                          <Chip
                            label={`${(signalsData.current_signal.confidence * 100).toFixed(1)}% Confidence`}
                            color={signalsData.current_signal.confidence > 0.7 ? 'success' : signalsData.current_signal.confidence > 0.5 ? 'warning' : 'error'}
                            variant="filled"
                          />
                        </Box>
                        <Grid container spacing={2}>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="text.secondary">Price</Typography>
                            <Typography variant="h6">${signalsData.current_signal.price.toFixed(2)}</Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="text.secondary">Market Regime</Typography>
                            <Typography variant="h6" sx={{ textTransform: 'capitalize' }}>
                              {signalsData.current_signal.market_regime}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="text.secondary">RSI</Typography>
                            <Typography variant="h6">{signalsData.current_signal.rsi.toFixed(1)}</Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="text.secondary">Volume Ratio</Typography>
                            <Typography variant="h6">{signalsData.current_signal.volume_ratio.toFixed(2)}x</Typography>
                          </Grid>
                        </Grid>
                      </>
                    ) : (
                      <Box sx={{ textAlign: 'center', py: 4 }}>
                        <Typography variant="h6" color="text.secondary" gutterBottom>
                          No Signal Data Available
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Run a backtest to generate trading signals and see real-time analysis.
                        </Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Signal Components */}
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Signal Components
                    </Typography>
                    {signalsData?.current_signal && signalsData.total_signals > 0 && signalsData.current_signal.price > 0 ? (
                      <Grid container spacing={2}>
                        <Grid item xs={12}>
                          <Typography variant="body2" color="text.secondary">Momentum Score</Typography>
                          <LinearProgress 
                            variant="determinate" 
                            value={Math.abs(signalsData.current_signal.momentum_score) * 50 + 50} 
                            sx={{ mb: 1 }}
                          />
                          <Typography variant="body2">
                            {signalsData.current_signal.momentum_score.toFixed(3)}
                          </Typography>
                        </Grid>
                        <Grid item xs={12}>
                          <Typography variant="body2" color="text.secondary">Mean Reversion Score</Typography>
                          <LinearProgress 
                            variant="determinate" 
                            value={Math.abs(signalsData.current_signal.mean_reversion_score) * 50 + 50} 
                            sx={{ mb: 1 }}
                          />
                          <Typography variant="body2">
                            {signalsData.current_signal.mean_reversion_score.toFixed(3)}
                          </Typography>
                        </Grid>
                        <Grid item xs={12}>
                          <Typography variant="body2" color="text.secondary">ML Score</Typography>
                          <LinearProgress 
                            variant="determinate" 
                            value={Math.abs(signalsData.current_signal.ml_score) * 50 + 50} 
                            sx={{ mb: 1 }}
                          />
                          <Typography variant="body2">
                            {signalsData.current_signal.ml_score.toFixed(3)}
                          </Typography>
                        </Grid>
                      </Grid>
                    ) : (
                      <Box sx={{ textAlign: 'center', py: 4 }}>
                        <Typography variant="body2" color="text.secondary">
                          Signal components will appear after running a backtest.
                        </Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Recent Signals Table */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Recent Signals ({signalsData?.total_signals || 0})
                    </Typography>
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Date</TableCell>
                            <TableCell>Signal</TableCell>
                            <TableCell align="right">Price</TableCell>
                            <TableCell align="right">Confidence</TableCell>
                            <TableCell align="right">RSI</TableCell>
                            <TableCell align="right">Volume Ratio</TableCell>
                            <TableCell>Regime</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {signalsData?.recent_signals && signalsData.recent_signals.length > 0 ? (
                            signalsData.recent_signals.slice(0, 20).map((signal: any, index: number) => (
                              <TableRow key={index}>
                                <TableCell>{signal.date}</TableCell>
                                <TableCell>
                                  <Chip
                                    label={signal.signal}
                                    color={signal.signal === 'BUY' ? 'success' : signal.signal === 'SELL' ? 'error' : 'default'}
                                    size="small"
                                  />
                                </TableCell>
                                <TableCell align="right">${signal.price.toFixed(2)}</TableCell>
                                <TableCell align="right">{(signal.confidence * 100).toFixed(1)}%</TableCell>
                                <TableCell align="right">{signal.rsi.toFixed(1)}</TableCell>
                                <TableCell align="right">{signal.volume_ratio.toFixed(2)}x</TableCell>
                                <TableCell sx={{ textTransform: 'capitalize' }}>{signal.market_regime}</TableCell>
                              </TableRow>
                            ))
                          ) : (
                            <TableRow>
                              <TableCell colSpan={7} align="center">
                                <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
                                  No recent signals available. Run a backtest to generate signals.
                                </Typography>
                              </TableCell>
                            </TableRow>
                          )}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </motion.div>
        )}

        {activeTab === 3 && (
          <motion.div
            key="watchlist"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Market Watchlist
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Symbol</TableCell>
                        <TableCell>Name</TableCell>
                        <TableCell align="right">Price</TableCell>
                        <TableCell align="right">Change</TableCell>
                        <TableCell align="right">Volume</TableCell>
                        <TableCell align="right">Market Cap</TableCell>
                        <TableCell align="right">P/E</TableCell>
                        <TableCell align="center">Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {watchlist.map((stock) => (
                        <TableRow key={stock.symbol}>
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                {stock.symbol}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell>{stock.name}</TableCell>
                          <TableCell align="right">${stock.price.toFixed(2)}</TableCell>
                          <TableCell align="right">
                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                              {getChangeIcon(stock.change)}
                              <Typography
                                variant="body2"
                                color={stock.change >= 0 ? 'success.main' : 'error.main'}
                              >
                                ${stock.change.toFixed(2)} ({stock.changePercent.toFixed(2)}%)
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell align="right">{stock.volume.toLocaleString()}</TableCell>
                          <TableCell align="right">${(stock.marketCap / 1000000000).toFixed(1)}B</TableCell>
                          <TableCell align="right">{stock.pe.toFixed(1)}</TableCell>
                          <TableCell align="center">
                            <IconButton onClick={() => toggleFavorite(stock.symbol)}>
                              {stock.isFavorite ? <Star color="warning" /> : <StarBorder />}
                            </IconButton>
                            <IconButton>
                              <MoreVert />
                            </IconButton>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {activeTab === 4 && (
          <motion.div
            key="risk"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Risk Metrics
                    </Typography>
                    {dashboardData.totalTrades > 0 ? (
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">Value at Risk (95%)</Typography>
                          <Typography variant="h6" color="error.main">
                            {dashboardData.riskMetrics.var.toFixed(1)}%
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">Beta</Typography>
                          <Typography variant="h6" color="primary">
                            {dashboardData.riskMetrics.beta.toFixed(2)}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">Alpha</Typography>
                          <Typography variant="h6" color="success.main">
                            {dashboardData.riskMetrics.alpha.toFixed(1)}%
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">Volatility</Typography>
                          <Typography variant="h6" color="warning.main">
                            {dashboardData.riskMetrics.volatility.toFixed(1)}%
                          </Typography>
                        </Grid>
                      </Grid>
                    ) : (
                      <Box sx={{ textAlign: 'center', py: 4 }}>
                        <Typography variant="h6" color="text.secondary" gutterBottom>
                          No Risk Data Available
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Run a backtest to generate risk metrics and performance analysis.
                        </Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Drawdown Analysis
                    </Typography>
                    {dashboardData.totalTrades > 0 ? (
                      <>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <Warning color="error" sx={{ mr: 1 }} />
                          <Typography variant="h6" color="error.main">
                            Max Drawdown: {dashboardData.maxDrawdown.toFixed(1)}%
                          </Typography>
                        </Box>
                        <Typography variant="body2" color="text.secondary">
                          Current drawdown: {dashboardData.maxDrawdown.toFixed(1)}% 
                          {Math.abs(dashboardData.maxDrawdown) < 10 ? ' (within acceptable range)' : ' (high risk)'}
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={Math.abs(dashboardData.maxDrawdown) / 20 * 100}
                          sx={{ mt: 1, height: 8, borderRadius: 4 }}
                          color={Math.abs(dashboardData.maxDrawdown) > 10 ? 'error' : 'success'}
                        />
                      </>
                    ) : (
                      <Box sx={{ textAlign: 'center', py: 4 }}>
                        <Typography variant="body2" color="text.secondary">
                          Drawdown analysis will appear after running a backtest.
                        </Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Risk-Adjusted Performance
                    </Typography>
                    {dashboardData.totalTrades > 0 ? (
                      <Grid container spacing={3}>
                        <Grid item xs={12} md={4}>
                          <Box sx={{ textAlign: 'center', p: 2 }}>
                            <Typography variant="h4" color="primary" gutterBottom>
                              {dashboardData.sharpeRatio.toFixed(2)}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Sharpe Ratio
                            </Typography>
                            <Typography variant="caption" color={dashboardData.sharpeRatio > 1.0 ? 'success.main' : dashboardData.sharpeRatio > 0.5 ? 'warning.main' : 'error.main'}>
                              {dashboardData.sharpeRatio > 1.0 ? 'Excellent (>1.0)' : dashboardData.sharpeRatio > 0.5 ? 'Good (>0.5)' : 'Poor (<0.5)'}
                            </Typography>
                          </Box>
                        </Grid>
                        <Grid item xs={12} md={4}>
                          <Box sx={{ textAlign: 'center', p: 2 }}>
                            <Typography variant="h4" color="secondary" gutterBottom>
                              {dashboardData.maxDrawdown !== 0 ? (dashboardData.totalReturn / Math.abs(dashboardData.maxDrawdown)).toFixed(2) : 'N/A'}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Calmar Ratio
                            </Typography>
                            <Typography variant="caption" color={dashboardData.maxDrawdown !== 0 && (dashboardData.totalReturn / Math.abs(dashboardData.maxDrawdown)) > 0.5 ? 'success.main' : 'warning.main'}>
                              {dashboardData.maxDrawdown !== 0 ? (dashboardData.totalReturn / Math.abs(dashboardData.maxDrawdown)) > 0.5 ? 'Good (>0.5)' : 'Poor (<0.5)' : 'N/A'}
                            </Typography>
                          </Box>
                        </Grid>
                        <Grid item xs={12} md={4}>
                          <Box sx={{ textAlign: 'center', p: 2 }}>
                            <Typography variant="h4" color="info.main" gutterBottom>
                              {dashboardData.winRate.toFixed(1)}%
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Win Rate
                            </Typography>
                            <Typography variant="caption" color={dashboardData.winRate > 60 ? 'success.main' : dashboardData.winRate > 40 ? 'warning.main' : 'error.main'}>
                              {dashboardData.winRate > 60 ? 'Above Average (>60%)' : dashboardData.winRate > 40 ? 'Average (40-60%)' : 'Below Average (<40%)'}
                            </Typography>
                          </Box>
                        </Grid>
                      </Grid>
                    ) : (
                      <Box sx={{ textAlign: 'center', py: 4 }}>
                        <Typography variant="h6" color="text.secondary" gutterBottom>
                          No Performance Data Available
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Run a backtest to generate risk-adjusted performance metrics.
                        </Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </motion.div>
        )}
      </AnimatePresence>
    </Box>
  );
};

export default Dashboard;

