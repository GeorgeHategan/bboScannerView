"""
Kristjan Qullamaggie Breakout Scanner
Detects breakouts based on Qullamaggie's methodology:
- Price breaks above 20-day high
- Volume spike (1.5x+ average)
- Price above key moving averages
- Uptrend confirmation (SMA 10 > SMA 20)
"""

import numpy as np
import pandas as pd


def detect_qullamaggie_breakout(df, lookback_days=20, volume_multiplier=1.5):
    """
    Detect Qullamaggie-style breakout pattern.
    
    Args:
        df: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
        lookback_days: Days to look back for high (default: 20)
        volume_multiplier: Minimum volume increase vs average (default: 1.5)
    
    Returns:
        'bullish' if breakout detected, None otherwise
    """
    if df is None or len(df) < lookback_days + 20:
        return None
    
    try:
        # Get recent data (last 50 days)
        recent = df.tail(50).copy()
        
        # Ensure we have required columns
        required_cols = ['Close', 'High', 'Volume']
        if not all(col in recent.columns for col in required_cols):
            return None
        
        # Calculate moving averages
        recent['SMA_10'] = recent['Close'].rolling(window=10).mean()
        recent['SMA_20'] = recent['Close'].rolling(window=20).mean()
        recent['SMA_50'] = recent['Close'].rolling(window=50).mean()
        
        # Get today's data (most recent)
        today = recent.iloc[-1]
        
        # Get lookback period (exclude today)
        lookback_data = recent.iloc[-lookback_days-1:-1]
        
        # Skip if not enough data
        if len(lookback_data) < lookback_days:
            return None
        
        # Calculate key metrics
        prev_high = lookback_data['High'].max()
        avg_volume = lookback_data['Volume'].mean()
        
        # Check for valid SMA values
        sma10_valid = not pd.isna(today['SMA_10'])
        sma20_valid = not pd.isna(today['SMA_20'])
        
        # Qullamaggie Breakout Criteria
        price_breakout = today['Close'] > prev_high
        volume_spike = today['Volume'] > (avg_volume * volume_multiplier)
        above_sma10 = today['Close'] > today['SMA_10'] if sma10_valid else False
        above_sma20 = today['Close'] > today['SMA_20'] if sma20_valid else False
        uptrend = (today['SMA_10'] > today['SMA_20'] 
                   if sma10_valid and sma20_valid else False)
        
        # All conditions must be met
        if (price_breakout and volume_spike and above_sma10 and 
            above_sma20 and uptrend):
            return 'bullish'
        
        return None
        
    except Exception as e:
        print(f'Error in qullamaggie scanner: {e}')
        return None


def get_qullamaggie_details(df, lookback_days=20):
    """
    Get detailed breakout metrics for display.
    
    Returns:
        Dict with breakout details or None
    """
    if df is None or len(df) < lookback_days + 20:
        return None
    
    try:
        recent = df.tail(50).copy()
        recent['SMA_10'] = recent['Close'].rolling(window=10).mean()
        recent['SMA_20'] = recent['Close'].rolling(window=20).mean()
        
        today = recent.iloc[-1]
        lookback_data = recent.iloc[-lookback_days-1:-1]
        
        prev_high = lookback_data['High'].max()
        avg_volume = lookback_data['Volume'].mean()
        
        breakout_pct = ((today['Close'] - prev_high) / prev_high) * 100
        vol_ratio = today['Volume'] / avg_volume
        
        return {
            'close': round(today['Close'], 2),
            'prev_high': round(prev_high, 2),
            'breakout_pct': round(breakout_pct, 2),
            'volume': int(today['Volume']),
            'avg_volume': int(avg_volume),
            'vol_ratio': round(vol_ratio, 2),
            'sma_10': round(today['SMA_10'], 2) if not pd.isna(today['SMA_10']) else 0,
            'sma_20': round(today['SMA_20'], 2) if not pd.isna(today['SMA_20']) else 0,
        }
    except:
        return None


# Pattern definition for integration
qullamaggie_pattern = {
    'QULLAMAGGIE_BREAKOUT': 'Qullamaggie Breakout'
}
