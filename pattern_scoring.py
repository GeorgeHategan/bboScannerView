"""
Pattern strength/quality scoring system.
Adds confidence scores to pattern detections based on various factors.
"""

import numpy as np
import talib


def calculate_pattern_strength(df, pattern_result, pattern_name):
    """
    Calculate pattern strength/quality score (0-100).
    
    Factors considered:
    - Volume confirmation
    - Trend context
    - Pattern consistency
    - Support/resistance levels
    """
    if pattern_result is None:
        return 0
    
    score = 50  # Base score
    
    try:
        # Get recent data
        recent = df.tail(20)
        close = recent['Close'].values
        volume = recent['Volume'].values
        high = recent['High'].values
        low = recent['Low'].values
        
        # 1. Volume Confirmation (±20 points)
        avg_volume = np.mean(volume[:-1])
        recent_volume = volume[-1]
        if recent_volume > avg_volume * 1.5:
            score += 20  # Strong volume
        elif recent_volume > avg_volume * 1.2:
            score += 10  # Good volume
        elif recent_volume < avg_volume * 0.5:
            score -= 20  # Weak volume
        
        # 2. Trend Strength (±15 points)
        sma_20 = np.mean(close)
        current_price = close[-1]
        if pattern_result == 'bullish':
            if current_price > sma_20 * 1.02:
                score += 15  # Already in uptrend
            elif current_price < sma_20 * 0.98:
                score += 5   # Reversal potential
        elif pattern_result == 'bearish':
            if current_price < sma_20 * 0.98:
                score += 15  # Already in downtrend
            elif current_price > sma_20 * 1.02:
                score += 5   # Reversal potential
        
        # 3. RSI Context (±10 points)
        if len(df) >= 14:
            rsi = talib.RSI(df['Close'].values, timeperiod=14)[-1]
            if pattern_result == 'bullish':
                if 30 < rsi < 50:
                    score += 10  # Oversold, good for bullish
                elif rsi > 70:
                    score -= 10  # Overbought, risky
            elif pattern_result == 'bearish':
                if 50 < rsi < 70:
                    score += 10  # Overbought, good for bearish
                elif rsi < 30:
                    score -= 10  # Oversold, risky
        
        # 4. Volatility (±5 points)
        atr = talib.ATR(high, low, close, timeperiod=14)[-1]
        if current_price > 0 and not np.isnan(atr):
            atr_percent = (atr / current_price) * 100
            if 1 < atr_percent < 3:
                score += 5   # Good volatility
            elif atr_percent > 5:
                score -= 5   # Too volatile
        
        # 5. Price consolidation (±10 points)
        if current_price > 0:
            price_range = (np.max(close[-5:]) - np.min(close[-5:])) / current_price
            if price_range < 0.03:
                score += 10  # Tight consolidation
            elif price_range > 0.10:
                score -= 5   # Too much noise
        
    except Exception as e:
        print(f'Error calculating strength: {e}')
        return 50  # Return neutral score on error
    
    # Clamp score between 0-100
    return max(0, min(100, score))


def get_signal_quality(score):
    """Convert numeric score to quality label."""
    if score >= 80:
        return 'strong'
    elif score >= 60:
        return 'good'
    elif score >= 40:
        return 'moderate'
    elif score >= 20:
        return 'weak'
    else:
        return 'very_weak'


def add_pattern_metadata(df, pattern_result, pattern_name):
    """
    Add metadata about pattern quality.
    Returns dict with pattern result, strength score, and quality label.
    """
    if pattern_result is None or pattern_result == 'None':
        return None
    
    strength = calculate_pattern_strength(df, pattern_result, pattern_name)
    quality = get_signal_quality(strength)
    
    return {
        'signal': pattern_result,
        'strength': strength,
        'quality': quality
    }
