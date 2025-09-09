"""
Data loader module for Alpha Signal Engine.
Handles CSV data ingestion and preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings


class DataLoader:
    """Handles loading and preprocessing of OHLCV data from CSV files."""
    
    def __init__(self, config):
        """Initialize DataLoader with configuration."""
        self.config = config
        
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file.
        
        Args:
            file_path: Path to CSV file. If None, uses config default.
            
        Returns:
            DataFrame with OHLCV data and datetime index.
        """
        if file_path is None:
            file_path = self.config.csv_file_path
            
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Clean the data - remove header rows and fix column names
            if 'Price' in df.columns and df.iloc[0, 0] == 'Ticker':
                # Skip the first two rows (header info)
                df = df.iloc[2:].reset_index(drop=True)
                
            # Rename columns to standard OHLCV format
            column_mapping = {
                'Price': 'Close',
                'Close': 'Close', 
                'High': 'High',
                'Low': 'Low',
                'Open': 'Open',
                'Volume': 'Volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Convert datetime column
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df = df.set_index('Datetime')
            else:
                # Create datetime index from first column if it's datetime
                first_col = df.columns[0]
                if pd.api.types.is_datetime64_any_dtype(df[first_col]):
                    df = df.set_index(first_col)
                else:
                    # Try to parse as datetime
                    try:
                        df.index = pd.to_datetime(df.iloc[:, 0])
                        df = df.iloc[:, 1:]
                    except:
                        warnings.warn("Could not parse datetime index. Using numeric index.")
            
            # Convert numeric columns
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with NaN values
            df = df.dropna()
            
            # Sort by datetime
            df = df.sort_index()
            
            # Ensure all required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading data from {file_path}: {str(e)}")
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data for signal generation.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Preprocessed DataFrame with technical indicators
        """
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        
        # Calculate log returns
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Calculate moving averages
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        
        # Calculate exponential moving averages
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df['Close'])
        
        # Calculate volume indicators
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Calculate volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Calculate price momentum indicators
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Average True Range (ATR) for volatility-aware risk
        df['atr'] = self._calculate_atr(df, period=14)

        # Remove NaN values conservatively: only drop rows where essential indicators are missing
        essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'bb_upper', 'bb_lower', 'rsi']
        keep_cols = [c for c in essential_cols if c in df.columns]
        df = df.dropna(subset=keep_cols)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR) using Wilder's smoothing."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics of the loaded data."""
        # Robust trading days calculation across index types
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                trading_days = df.index.normalize().nunique()
            else:
                trading_days = pd.Index(df.index).nunique()
        except Exception:
            trading_days = len(df)

        return {
            'start_date': df.index.min(),
            'end_date': df.index.max(),
            'total_periods': len(df),
            'trading_days': int(trading_days),
            'price_range': (df['Close'].min(), df['Close'].max()),
            'avg_volume': float(df['Volume'].mean()),
            'avg_daily_volatility': float(df['returns'].std() * np.sqrt(252)),
            'total_return': float((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1),
        }

