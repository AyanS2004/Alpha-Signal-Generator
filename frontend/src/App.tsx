import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box } from '@mui/material';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Backtesting from './pages/Backtesting';
import RealTime from './pages/RealTime';
import Optimization from './pages/Optimization';
import Settings from './pages/Settings';
import Signals from './pages/Signals';
import MLInsights from './pages/MLInsights';
import Risk from './pages/Risk';

function App() {
  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/backtesting" element={<Backtesting />} />
          <Route path="/signals" element={<Signals />} />
          <Route path="/ml" element={<MLInsights />} />
          <Route path="/risk" element={<Risk />} />
          <Route path="/realtime" element={<RealTime />} />
          <Route path="/optimization" element={<Optimization />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Layout>
    </Box>
  );
}

export default App;

