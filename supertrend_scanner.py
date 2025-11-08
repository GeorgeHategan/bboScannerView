"""
SuperTrend Scanner
Detects bullish SuperTrend flips on daily timeframe

SuperTrend Indicator:
- Period: 10 (ATR calculation)
- Multiplier: 3.0 (standard setting)
- Signal: Price crosses above SuperTrend line

Entry Criteria:
1. SuperTrend flipped bullish within last 1-3 days
2. Price still above SuperTrend line
3. Volume confirmation (optional)
"""

import pandas as pd
import numpy as np


def calculate_atr(df, period=14):
    """Calculate Average True Range (ATR)"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    # True Range is the maximum of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR is the moving average of True Range
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_supertrend(df, period=10, multiplier=3.0):
    """
    Calculate SuperTrend indicator.
    
    Returns DataFrame with additional columns:
    - supertrend: The SuperTrend line value
    - supertrend_direction: 1 for bullish, -1 for bearish
    - trend_change: True when trend direction changes
    """
    df = df.copy()
    
    # Calculate ATR
    atr = calculate_atr(df, period)
    
    # Calculate basic bands
    hl_avg = (df['High'] + df['Low']) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    # Initialize SuperTrend
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    # First valid index (after ATR warmup)
    first_valid = period
    
    # Initialize first values
    if len(df) > first_valid:
        if df['Close'].iloc[first_valid] <= upper_band.iloc[first_valid]:
            supertrend.iloc[first_valid] = upper_band.iloc[first_valid]
            direction.iloc[first_valid] = -1  # Bearish
        else:
            supertrend.iloc[first_valid] = lower_band.iloc[first_valid]
            direction.iloc[first_valid] = 1   # Bullish
    
    # Calculate SuperTrend for remaining periods
    for i in range(first_valid + 1, len(df)):
        # Update bands based on previous SuperTrend
        if direction.iloc[i-1] == 1:  # Was bullish
            # Lower band can only go up or stay same
            if lower_band.iloc[i] > supertrend.iloc[i-1]:
                final_lower = lower_band.iloc[i]
            else:
                final_lower = supertrend.iloc[i-1]
                
            # Check if trend flips to bearish
            if df['Close'].iloc[i] <= final_lower:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = final_lower
                direction.iloc[i] = 1
                
        else:  # Was bearish
            # Upper band can only go down or stay same
            if upper_band.iloc[i] < supertrend.iloc[i-1]:
                final_upper = upper_band.iloc[i]
            else:
                final_upper = supertrend.iloc[i-1]
                
            # Check if trend flips to bullish
            if df['Close'].iloc[i] >= final_upper:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = final_upper
                direction.iloc[i] = -1
    
    # Add to dataframe
    df['supertrend'] = supertrend
    df['supertrend_direction'] = direction
    
    # Detect trend changes
    df['trend_change'] = df['supertrend_direction'].diff() != 0
    
    return df


def detect_supertrend_bullish(df, max_days_ago=3):
    """
    Detect recent bullish SuperTrend flip.
    
    Args:
        df: DataFrame with OHLCV data
        max_days_ago: Maximum days since flip (default: 3)
    
    Returns:
        'bullish' if recent bullish flip detected, None otherwise
    """
    if df is None or len(df) < 50:
        return None
    
    try:
        # Calculate SuperTrend
        df_st = calculate_supertrend(df, period=10, multiplier=3.0)
        
        # Get last few days
        recent = df_st.tail(max_days_ago + 1)
        
        # Find bullish flips
        for i in range(len(recent)):
            if (recent['trend_change'].iloc[i] and 
                recent['supertrend_direction'].iloc[i] == 1):
                
                # Check if still bullish today
                if df_st['supertrend_direction'].iloc[-1] != 1:
                    continue
                
                # Found a recent bullish flip that's still active
                return 'bullish'
        
        return None
        
    except Exception as e:
        print(f'Error in SuperTrend scanner: {e}')
        return None


def detect_supertrend_fresh(df):
    """SuperTrend flip within 1 day (freshest signals)"""
    return detect_supertrend_bullish(df, max_days_ago=1)


def detect_supertrend_recent(df):
    """SuperTrend flip within 2 days"""
    return detect_supertrend_bullish(df, max_days_ago=2)


# Pattern definitions for integration
supertrend_patterns = {
    'SUPERTREND_BULLISH': 'SuperTrend Bullish (1-3 Days)',
    'SUPERTREND_FRESH': 'SuperTrend Fresh (1 Day)',
    'SUPERTREND_RECENT': 'SuperTrend Recent (2 Days)'
}
