"""
Momentum Burst Scanner (Stockbee-Style)
Identifies explosive short-term momentum moves with volume confirmation

Based on Stockbee's Momentum Burst methodology:
- Looks for 4%+ gain in 1-5 days
- Strong volume confirmation (2x+ average)
- Price above 20 SMA (short-term trend)

Signal Types:
- momentum_burst_1d: Explosive 1-day move (4%+)
- momentum_burst_3d: Strong 3-day move (6%+)
- momentum_burst_5d: Sustained 5-day move (8%+)
"""

import pandas as pd


def detect_momentum_burst(df, burst_type='any'):
    """
    Detect momentum burst patterns.
    
    Args:
        df: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
        burst_type: 'any', '1d', '3d', or '5d'
    
    Returns:
        'bullish' if momentum burst detected, None otherwise
    """
    if df is None or len(df) < 20:
        return None
    
    try:
        # Get recent data
        recent = df.tail(30).copy()
        
        # Ensure required columns exist
        if 'Close' not in recent.columns or 'Volume' not in recent.columns:
            return None
        
        # Calculate 20 SMA
        recent['SMA_20'] = recent['Close'].rolling(window=20).mean()
        
        # Get latest data
        latest = recent.iloc[-1]
        price = latest['Close']
        volume = latest['Volume']
        sma_20 = latest['SMA_20']
        
        # Skip if invalid data
        if pd.isna(price) or pd.isna(volume) or pd.isna(sma_20):
            return None
        
        # Get average volume (20-day)
        avg_volume = recent['Volume'].tail(20).mean()
        
        if avg_volume == 0:
            return None
        
        # Price must be above 20 SMA (in uptrend)
        if price < sma_20:
            return None
        
        # Calculate price changes
        pct_change_1d = 0
        pct_change_3d = 0
        pct_change_5d = 0
        
        if len(recent) >= 2:
            price_1d_ago = recent.iloc[-2]['Close']
            if price_1d_ago > 0:
                pct_change_1d = ((price - price_1d_ago) / price_1d_ago * 100)
        
        if len(recent) >= 4:
            price_3d_ago = recent.iloc[-4]['Close']
            if price_3d_ago > 0:
                pct_change_3d = ((price - price_3d_ago) / price_3d_ago * 100)
        
        if len(recent) >= 6:
            price_5d_ago = recent.iloc[-6]['Close']
            if price_5d_ago > 0:
                pct_change_5d = ((price - price_5d_ago) / price_5d_ago * 100)
        
        # Volume confirmation
        volume_ratio = volume / avg_volume
        
        # Check 1-day burst (most explosive)
        if (burst_type in ['any', '1d'] and 
            pct_change_1d >= 4.0 and volume_ratio >= 2.0):
            return 'bullish'
        
        # Check 3-day burst (sustained move)
        if burst_type in ['any', '3d'] and pct_change_3d >= 6.0:
            # Check volume on recent 3 days
            recent_3d = recent.tail(3)
            avg_vol_3d = recent_3d['Volume'].mean()
            if avg_vol_3d >= (avg_volume * 1.5):
                return 'bullish'
        
        # Check 5-day burst (strong sustained)
        if burst_type in ['any', '5d'] and pct_change_5d >= 8.0:
            # Check volume on recent 5 days
            recent_5d = recent.tail(5)
            avg_vol_5d = recent_5d['Volume'].mean()
            if avg_vol_5d >= (avg_volume * 1.3):
                return 'bullish'
        
        return None
        
    except Exception as e:
        print(f'Error in momentum burst scanner: {e}')
        return None


def detect_momentum_burst_1d(df):
    """1-day explosive move (4%+ on 2x volume)"""
    return detect_momentum_burst(df, burst_type='1d')


def detect_momentum_burst_3d(df):
    """3-day sustained move (6%+ with volume)"""
    return detect_momentum_burst(df, burst_type='3d')


def detect_momentum_burst_5d(df):
    """5-day strong move (8%+ with volume)"""
    return detect_momentum_burst(df, burst_type='5d')


# Pattern definitions for integration
momentum_burst_patterns = {
    'MOMENTUM_BURST_1D': 'Momentum Burst (1-Day)',
    'MOMENTUM_BURST_3D': 'Momentum Burst (3-Day)',
    'MOMENTUM_BURST_5D': 'Momentum Burst (5-Day)'
}
