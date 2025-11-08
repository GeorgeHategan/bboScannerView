import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def detect_cup_and_handle(df, cup_depth_min=0.12, cup_depth_max=0.33, 
                          handle_depth_max=0.5, window=5):
    """
    Detect Cup and Handle pattern in stock data.
    
    Parameters:
    - df: DataFrame with OHLC data
    - cup_depth_min: Minimum cup depth as percentage (default 12%)
    - cup_depth_max: Maximum cup depth as percentage (default 33%)
    - handle_depth_max: Maximum handle depth relative to cup (default 50%)
    - window: Window for finding local maxima/minima
    
    Returns:
    - 'bullish' if pattern detected, None otherwise
    """
    if len(df) < 60:  # Need at least 60 days for cup and handle
        return None
    
    try:
        # Use only recent data (last 120 days for pattern detection)
        recent_df = df.tail(120).copy()
        close = recent_df['Close'].values
        
        # Find local maxima and minima
        maxima_idx = argrelextrema(close, np.greater, order=window)[0]
        minima_idx = argrelextrema(close, np.less, order=window)[0]
        
        if len(maxima_idx) < 2 or len(minima_idx) < 1:
            return None
        
        # Look for cup pattern: high -> low -> high
        for i in range(len(maxima_idx) - 1):
            left_rim = maxima_idx[i]
            right_rim = maxima_idx[i + 1]
            
            # Find the lowest point between the two rims (cup bottom)
            cup_bottom_idx = minima_idx[(minima_idx > left_rim) & (minima_idx < right_rim)]
            if len(cup_bottom_idx) == 0:
                continue
            
            cup_bottom = np.argmin(close[left_rim:right_rim]) + left_rim
            
            # Check cup depth
            left_price = close[left_rim]
            bottom_price = close[cup_bottom]
            right_price = close[right_rim]
            
            cup_depth = (left_price - bottom_price) / left_price
            
            if cup_depth < cup_depth_min or cup_depth > cup_depth_max:
                continue
            
            # Check if right rim is close to left rim height (within 5%)
            if abs(right_price - left_price) / left_price > 0.05:
                continue
            
            # Look for handle: pullback after right rim
            if right_rim >= len(close) - 5:
                continue
            
            handle_section = close[right_rim:]
            if len(handle_section) < 5:
                continue
            
            handle_low = np.min(handle_section)
            handle_depth = (right_price - handle_low) / right_price
            
            # Handle should be shallower than cup
            if handle_depth > cup_depth * handle_depth_max:
                continue
            
            # Check if currently near breakout (within 3% of right rim)
            current_price = close[-1]
            if current_price >= right_price * 0.97:
                return 'bullish'
        
        return None
        
    except Exception as e:
        print(f'Error in cup and handle detection: {e}')
        return None


def detect_ascending_triangle(df, window=5):
    """
    Detect Ascending Triangle pattern.
    Returns 'bullish' if pattern detected.
    """
    if len(df) < 40:
        return None
    
    try:
        recent_df = df.tail(60).copy()
        high = recent_df['High'].values
        low = recent_df['Low'].values
        
        # Find resistance level (flat top)
        maxima_idx = argrelextrema(high, np.greater, order=window)[0]
        if len(maxima_idx) < 2:
            return None
        
        # Check if highs are relatively flat
        recent_highs = high[maxima_idx[-3:]] if len(maxima_idx) >= 3 else high[maxima_idx[-2:]]
        if np.std(recent_highs) / np.mean(recent_highs) < 0.02:  # Less than 2% variation
            # Check if lows are rising
            minima_idx = argrelextrema(low, np.less, order=window)[0]
            if len(minima_idx) >= 2:
                recent_lows = low[minima_idx[-2:]]
                if recent_lows[-1] > recent_lows[0]:
                    return 'bullish'
        
        return None
    except Exception as e:
        print(f'Error in ascending triangle detection: {e}')
        return None


def detect_double_bottom(df, window=5):
    """
    Detect Double Bottom pattern.
    Returns 'bullish' if pattern detected.
    """
    if len(df) < 40:
        return None
    
    try:
        recent_df = df.tail(80).copy()
        low = recent_df['Low'].values
        
        # Find local minima
        minima_idx = argrelextrema(low, np.less, order=window)[0]
        if len(minima_idx) < 2:
            return None
        
        # Check last two bottoms
        bottom1_idx = minima_idx[-2]
        bottom2_idx = minima_idx[-1]
        
        bottom1_price = low[bottom1_idx]
        bottom2_price = low[bottom2_idx]
        
        # Bottoms should be at similar levels (within 3%)
        if abs(bottom1_price - bottom2_price) / bottom1_price < 0.03:
            # Check if there's a peak between them
            if bottom2_idx > bottom1_idx + 5:
                middle_section = low[bottom1_idx:bottom2_idx]
                middle_high = np.max(middle_section)
                if middle_high > bottom1_price * 1.05:
                    return 'bullish'
        
        return None
    except Exception as e:
        print(f'Error in double bottom detection: {e}')
        return None


def detect_bull_flag(df, pole_min_gain=0.10, flag_max_decline=0.5, window=3):
    """
    Detect Bull Flag pattern.
    Bull Flag = Strong upward move (pole) + consolidation/pullback (flag)
    Returns 'bullish' if pattern detected.
    """
    if len(df) < 30:
        return None
    
    try:
        recent_df = df.tail(50).copy()
        close = recent_df['Close'].values
        high = recent_df['High'].values
        
        # Look for the pole (strong upward movement)
        for i in range(10, len(close) - 10):
            pole_start = close[i - 10]
            pole_end = close[i]
            
            # Skip if invalid data
            if pole_start <= 0 or np.isnan(pole_start) or np.isnan(pole_end):
                continue
            
            pole_gain = (pole_end - pole_start) / pole_start
            
            # Check if pole meets minimum gain requirement
            if pole_gain < pole_min_gain:
                continue
            
            # Look for flag (consolidation/pullback after pole)
            flag_section = close[i:i + 10] if i + 10 < len(close) else close[i:]
            if len(flag_section) < 5:
                continue
            
            flag_high = np.max(flag_section)
            flag_low = np.min(flag_section)
            flag_decline = (flag_high - flag_low) / flag_high
            
            # Flag should be a small consolidation (decline less than pole gain)
            if flag_decline < pole_gain * flag_max_decline:
                # Check if currently at flag support or breaking out
                current_price = close[-1]
                if current_price >= flag_low * 0.98:
                    return 'bullish'
        
        return None
    except Exception as e:
        print(f'Error in bull flag detection: {e}')
        return None


def detect_bear_flag(df, pole_min_decline=0.10, flag_max_gain=0.5, window=3):
    """
    Detect Bear Flag pattern.
    Bear Flag = Strong downward move (pole) + consolidation/bounce (flag)
    Returns 'bearish' if pattern detected.
    """
    if len(df) < 30:
        return None
    
    try:
        recent_df = df.tail(50).copy()
        close = recent_df['Close'].values
        low = recent_df['Low'].values
        
        # Look for the pole (strong downward movement)
        for i in range(10, len(close) - 10):
            pole_start = close[i - 10]
            pole_end = close[i]
            
            # Skip if invalid data
            if pole_start <= 0 or np.isnan(pole_start) or np.isnan(pole_end):
                continue
            
            pole_decline = (pole_start - pole_end) / pole_start
            
            # Check if pole meets minimum decline requirement
            if pole_decline < pole_min_decline:
                continue
            
            # Look for flag (consolidation/bounce after pole)
            flag_section = close[i:i + 10] if i + 10 < len(close) else close[i:]
            if len(flag_section) < 5:
                continue
            
            flag_high = np.max(flag_section)
            flag_low = np.min(flag_section)
            
            # Skip if invalid data
            if flag_low <= 0 or np.isnan(flag_low) or np.isnan(flag_high):
                continue
            
            flag_bounce = (flag_high - flag_low) / flag_low
            
            # Flag should be a small consolidation (bounce less than pole decline)
            if flag_bounce < pole_decline * flag_max_gain:
                # Check if currently at flag resistance or breaking down
                current_price = close[-1]
                if current_price <= flag_high * 1.02:
                    return 'bearish'
        
        return None
    except Exception as e:
        print(f'Error in bear flag detection: {e}')
        return None


# Dictionary of custom patterns
custom_chart_patterns = {
    'CUP_AND_HANDLE': 'Cup and Handle',
    'ASCENDING_TRIANGLE': 'Ascending Triangle',
    'DOUBLE_BOTTOM': 'Double Bottom',
    'BULL_FLAG': 'Bull Flag',
    'BEAR_FLAG': 'Bear Flag'
}
