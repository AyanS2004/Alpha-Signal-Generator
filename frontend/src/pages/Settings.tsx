import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  TextField,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import {
  Settings,
  Save,
  Restore,
  Security,
  Notifications,
  DataUsage,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';

interface SettingsConfig {
  // Trading Settings
  initialCapital: number;
  positionSize: number;
  transactionCost: number;
  stopLoss: number;
  takeProfit: number;
  
  // Signal Settings
  momentumLookback: number;
  momentumThreshold: number;
  meanReversionLookback: number;
  meanReversionThreshold: number;
  
  // System Settings
  autoTrading: boolean;
  notifications: boolean;
  dataRetention: number;
  riskManagement: boolean;
  
  // API Settings
  apiKey: string;
  apiSecret: string;
  dataProvider: string;
}

const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<SettingsConfig>({
    initialCapital: 100000,
    positionSize: 0.1,
    transactionCost: 1.0,
    stopLoss: 50,
    takeProfit: 100,
    momentumLookback: 20,
    momentumThreshold: 0.02,
    meanReversionLookback: 50,
    meanReversionThreshold: 0.01,
    autoTrading: false,
    notifications: true,
    dataRetention: 30,
    riskManagement: true,
    apiKey: '',
    apiSecret: '',
    dataProvider: 'alpha_vantage',
  });

  const [saving, setSaving] = useState(false);

  const handleSettingChange = (key: keyof SettingsConfig, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const saveSettings = async () => {
    setSaving(true);
    
    // Simulate API call
    setTimeout(() => {
      setSaving(false);
      toast.success('Settings saved successfully!');
    }, 1000);
  };

  const resetSettings = () => {
    setSettings({
      initialCapital: 100000,
      positionSize: 0.1,
      transactionCost: 1.0,
      stopLoss: 50,
      takeProfit: 100,
      momentumLookback: 20,
      momentumThreshold: 0.02,
      meanReversionLookback: 50,
      meanReversionThreshold: 0.01,
      autoTrading: false,
      notifications: true,
      dataRetention: 30,
      riskManagement: true,
      apiKey: '',
      apiSecret: '',
      dataProvider: 'alpha_vantage',
    });
    toast.success('Settings reset to defaults!');
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>

      <Grid container spacing={3}>
        {/* Trading Settings */}
        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Trading Configuration
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Initial Capital ($)"
                    type="number"
                    value={settings.initialCapital}
                    onChange={(e) => handleSettingChange('initialCapital', Number(e.target.value))}
                  />
                </Grid>

                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Position Size (%)"
                    type="number"
                    value={settings.positionSize * 100}
                    onChange={(e) => handleSettingChange('positionSize', Number(e.target.value) / 100)}
                  />
                </Grid>

                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Transaction Cost (bps)"
                    type="number"
                    value={settings.transactionCost}
                    onChange={(e) => handleSettingChange('transactionCost', Number(e.target.value))}
                  />
                </Grid>

                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Stop Loss (bps)"
                    type="number"
                    value={settings.stopLoss}
                    onChange={(e) => handleSettingChange('stopLoss', Number(e.target.value))}
                  />
                </Grid>

                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Take Profit (bps)"
                    type="number"
                    value={settings.takeProfit}
                    onChange={(e) => handleSettingChange('takeProfit', Number(e.target.value))}
                  />
                </Grid>
              </Grid>
            </Paper>
          </motion.div>
        </Grid>

        {/* Signal Settings */}
        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Signal Configuration
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Momentum Lookback"
                    type="number"
                    value={settings.momentumLookback}
                    onChange={(e) => handleSettingChange('momentumLookback', Number(e.target.value))}
                  />
                </Grid>

                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Momentum Threshold (%)"
                    type="number"
                    value={settings.momentumThreshold * 100}
                    onChange={(e) => handleSettingChange('momentumThreshold', Number(e.target.value) / 100)}
                  />
                </Grid>

                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Mean Reversion Lookback"
                    type="number"
                    value={settings.meanReversionLookback}
                    onChange={(e) => handleSettingChange('meanReversionLookback', Number(e.target.value))}
                  />
                </Grid>

                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Mean Reversion Threshold (%)"
                    type="number"
                    value={settings.meanReversionThreshold * 100}
                    onChange={(e) => handleSettingChange('meanReversionThreshold', Number(e.target.value) / 100)}
                  />
                </Grid>
              </Grid>
            </Paper>
          </motion.div>
        </Grid>

        {/* System Settings */}
        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                System Configuration
              </Typography>

              <Box sx={{ mb: 2 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.autoTrading}
                      onChange={(e) => handleSettingChange('autoTrading', e.target.checked)}
                    />
                  }
                  label="Auto Trading"
                />
              </Box>

              <Box sx={{ mb: 2 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.notifications}
                      onChange={(e) => handleSettingChange('notifications', e.target.checked)}
                    />
                  }
                  label="Notifications"
                />
              </Box>

              <Box sx={{ mb: 2 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.riskManagement}
                      onChange={(e) => handleSettingChange('riskManagement', e.target.checked)}
                    />
                  }
                  label="Risk Management"
                />
              </Box>

              <TextField
                fullWidth
                label="Data Retention (days)"
                type="number"
                value={settings.dataRetention}
                onChange={(e) => handleSettingChange('dataRetention', Number(e.target.value))}
              />
            </Paper>
          </motion.div>
        </Grid>

        {/* API Settings */}
        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                API Configuration
              </Typography>

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Data Provider</InputLabel>
                <Select
                  value={settings.dataProvider}
                  onChange={(e) => handleSettingChange('dataProvider', e.target.value)}
                  label="Data Provider"
                >
                  <MenuItem value="alpha_vantage">Alpha Vantage</MenuItem>
                  <MenuItem value="yahoo_finance">Yahoo Finance</MenuItem>
                  <MenuItem value="polygon">Polygon.io</MenuItem>
                </Select>
              </FormControl>

              <TextField
                fullWidth
                label="API Key"
                type="password"
                value={settings.apiKey}
                onChange={(e) => handleSettingChange('apiKey', e.target.value)}
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="API Secret"
                type="password"
                value={settings.apiSecret}
                onChange={(e) => handleSettingChange('apiSecret', e.target.value)}
              />

              <Alert severity="info" sx={{ mt: 2 }}>
                API credentials are encrypted and stored securely.
              </Alert>
            </Paper>
          </motion.div>
        </Grid>

        {/* Action Buttons */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Paper sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
                <Button
                  variant="contained"
                  startIcon={<Save />}
                  onClick={saveSettings}
                  disabled={saving}
                  size="large"
                >
                  {saving ? 'Saving...' : 'Save Settings'}
                </Button>

                <Button
                  variant="outlined"
                  startIcon={<Restore />}
                  onClick={resetSettings}
                  size="large"
                >
                  Reset to Defaults
                </Button>
              </Box>
            </Paper>
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SettingsPage;

