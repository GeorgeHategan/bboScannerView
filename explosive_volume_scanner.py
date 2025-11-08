"""
Explosive Volume Scanner
Identifies stocks with abnormal volume spikes that may indicate institutional activity,
breaking news, or potential big moves.

Signal Types:
- explosive_volume_3x: Volume >= 3x average (Major spike)
- explosive_volume_5x: Volume >= 5x average (Extreme spike)
- explosive_volume_10x: Volume >= 10x average (Massive institutional activity)
"""

import pandas as pd


def detect_explosive_volume_3x(df, lookback=20):
    """
    Detect 3x+ volume spike.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Days to calculate average volume (default: 20)
    
    Returns:
        'bullish' if volume spike detected, None otherwise
    """
    if df is None or len(df) < lookback + 1:
        return None
    
    try:
        recent = df.tail(lookback + 1).copy()
        
        if 'Volume' not in recent.columns:
            return None
        
        # Get today's volume
        today_volume = recent['Volume'].iloc[-1]
        
        # Calculate average volume (excluding today)
        avg_volume = recent['Volume'].iloc[:-1].mean()
        
        if avg_volume == 0:
            return None
        
        volume_ratio = today_volume / avg_volume
        
        # 3x volume spike
        if volume_ratio >= 3.0:
            return 'bullish'
        
        return None
        
    except Exception as e:
        print(f"Error in explosive_volume_3x: {e}")
        return None


def detect_explosive_volume_5x(df, lookback=20):
    """
    Detect 5x+ volume spike (extreme).
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Days to calculate average volume (default: 20)
    
    Returns:
        'bullish' if volume spike detected, None otherwise
    """
    if df is None or len(df) < lookback + 1:
        return None
    
    try:
        recent = df.tail(lookback + 1).copy()
        
        if 'Volume' not in recent.columns:
            return None
        
        # Get today's volume
        today_volume = recent['Volume'].iloc[-1]
        
        # Calculate average volume (excluding today)
        avg_volume = recent['Volume'].iloc[:-1].mean()
        
        if avg_volume == 0:
            return None
        
        volume_ratio = today_volume / avg_volume
        
        # 5x volume spike
        if volume_ratio >= 5.0:
            return 'bullish'
        
        return None
        
    except Exception as e:
        print(f"Error in explosive_volume_5x: {e}")
        return None


def detect_explosive_volume_10x(df, lookback=20):
    """
    Detect 10x+ volume spike (massive institutional activity).
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Days to calculate average volume (default: 20)
    
    Returns:
        'bullish' if volume spike detected, None otherwise
    """
    if df is None or len(df) < lookback + 1:
        return None
    
    try:
        recent = df.tail(lookback + 1).copy()
        
        if 'Volume' not in recent.columns:
            return None
        
        # Get today's volume
        today_volume = recent['Volume'].iloc[-1]
        
        # Calculate average volume (excluding today)
        avg_volume = recent['Volume'].iloc[:-1].mean()
        
        if avg_volume == 0:
            return None
        
        volume_ratio = today_volume / avg_volume
        
        # 10x volume spike
        if volume_ratio >= 10.0:
            return 'bullish'
        
        return None
        
    except Exception as e:
        print(f"Error in explosive_volume_10x: {e}")
        return None


def detect_volume_surge_with_price(df, lookback=20, min_price_change=2.0):
    """
    Detect volume surge (3x+) combined with significant price movement.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Days to calculate average volume (default: 20)
        min_price_change: Minimum price change % (default: 2%)
    
    Returns:
        'bullish' if surge detected with price up, 'bearish' if down, None otherwise
    """
    if df is None or len(df) < lookback + 1:
        return None
    
    try:
        recent = df.tail(lookback + 1).copy()
        
        if 'Volume' not in recent.columns or 'Close' not in recent.columns:
            return None
        
        # Get today's data
        today_volume = recent['Volume'].iloc[-1]
        today_close = recent['Close'].iloc[-1]
        today_open = recent['Open'].iloc[-1]
        
        # Calculate average volume (excluding today)
        avg_volume = recent['Volume'].iloc[:-1].mean()
        
        if avg_volume == 0:
            return None
        
        volume_ratio = today_volume / avg_volume
        
        # Calculate price change
        price_change = ((today_close - today_open) / today_open) * 100
        
        # Volume surge + significant price movement
        if volume_ratio >= 3.0 and abs(price_change) >= min_price_change:
            if price_change > 0:
                return 'bullish'
            else:
                return 'bearish'
        
        return None
        
    except Exception as e:
        print(f"Error in volume_surge_with_price: {e}")
        return None


# Pattern definitions for the UI
explosive_volume_patterns = {
    'EXPLOSIVE_VOLUME_3X': 'Explosive Volume (3x Average)',
    'EXPLOSIVE_VOLUME_5X': 'Explosive Volume (5x Average)',
    'EXPLOSIVE_VOLUME_10X': 'Explosive Volume (10x Average)',
    'VOLUME_SURGE_WITH_PRICE': 'Volume Surge + Price Move (3x + 2%)'
}
