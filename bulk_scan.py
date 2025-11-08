"""
Bulk pattern scanner - scans all patterns at once and exports top signals.
"""

import os
import csv
import talib
import pandas
from patterns import candlestick_patterns
from custom_patterns import (custom_chart_patterns, detect_cup_and_handle,
                              detect_ascending_triangle, detect_double_bottom,
                              detect_bull_flag, detect_bear_flag)
from pattern_scoring import calculate_pattern_strength, get_signal_quality


def scan_all_patterns():
    """Scan all patterns for all stocks and return results sorted by strength."""
    
    results = []
    
    # Combine all patterns
    all_patterns = {**candlestick_patterns, **custom_chart_patterns}
    
    # Custom pattern functions
    custom_functions = {
        'CUP_AND_HANDLE': detect_cup_and_handle,
        'ASCENDING_TRIANGLE': detect_ascending_triangle,
        'DOUBLE_BOTTOM': detect_double_bottom,
        'BULL_FLAG': detect_bull_flag,
        'BEAR_FLAG': detect_bear_flag
    }
    
    # Read symbols
    stocks = {}
    with open('datasets/symbols.csv') as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                stocks[row[0]] = row[1]
    
    print(f'Scanning {len(stocks)} stocks for {len(all_patterns)} patterns...\n')
    
    # Scan each stock
    for idx, (symbol, company) in enumerate(stocks.items(), 1):
        csv_file = f'datasets/daily/{symbol}.csv'
        
        if not os.path.exists(csv_file):
            continue
        
        try:
            df = pandas.read_csv(csv_file)
            
            # Sort by date
            date_col = 'date' if 'date' in df.columns else 'Date' if 'Date' in df.columns else df.columns[0]
            if date_col in df.columns:
                df[date_col] = pandas.to_datetime(df[date_col])
                df = df.sort_values(date_col)
                
                # Skip outdated data
                latest_date = df[date_col].iloc[-1]
                days_old = (pandas.Timestamp.now() - latest_date).days
                if days_old > 30:
                    continue
            
            # Test each pattern
            for pattern_key, pattern_name in all_patterns.items():
                try:
                    signal = None
                    
                    if pattern_key in custom_functions:
                        # Custom pattern
                        signal = custom_functions[pattern_key](df)
                    else:
                        # TA-Lib pattern
                        pattern_function = getattr(talib, pattern_key)
                        pattern_results = pattern_function(df['Open'], df['High'], df['Low'], df['Close'])
                        last = pattern_results.tail(1).values[0]
                        
                        if last > 0:
                            signal = 'bullish'
                        elif last < 0:
                            signal = 'bearish'
                    
                    # Calculate strength if signal found
                    if signal and signal != 'None':
                        strength = calculate_pattern_strength(df, signal, pattern_key)
                        quality = get_signal_quality(strength)
                        
                        results.append({
                            'symbol': symbol,
                            'company': company,
                            'pattern': pattern_name,
                            'pattern_key': pattern_key,
                            'signal': signal,
                            'strength': strength,
                            'quality': quality
                        })
                        
                except Exception as e:
                    continue
            
            if idx % 50 == 0:
                print(f'Processed {idx}/{len(stocks)} stocks...')
                
        except Exception as e:
            print(f'Error processing {symbol}: {e}')
            continue
    
    return results


def export_top_signals(results, top_n=30, output_file='top_signals.csv'):
    """Export top N signals sorted by strength."""
    
    # Sort by strength (descending)
    sorted_results = sorted(results, key=lambda x: x['strength'], reverse=True)
    
    # Take top N
    top_results = sorted_results[:top_n]
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['rank', 'symbol', 'company', 'pattern', 'signal', 'strength', 'quality']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for idx, result in enumerate(top_results, 1):
            writer.writerow({
                'rank': idx,
                'symbol': result['symbol'],
                'company': result['company'],
                'pattern': result['pattern'],
                'signal': result['signal'],
                'strength': result['strength'],
                'quality': result['quality']
            })
    
    print(f'\n✅ Exported top {len(top_results)} signals to {output_file}')
    return top_results


def print_summary(results):
    """Print summary statistics."""
    if not results:
        print('No signals found!')
        return
    
    print(f'\n{"="*60}')
    print(f'SCAN SUMMARY')
    print(f'{"="*60}')
    print(f'Total signals found: {len(results)}')
    
    # Count by signal type
    bullish = sum(1 for r in results if r['signal'] == 'bullish')
    bearish = sum(1 for r in results if r['signal'] == 'bearish')
    print(f'Bullish signals: {bullish}')
    print(f'Bearish signals: {bearish}')
    
    # Count by quality
    quality_counts = {}
    for r in results:
        q = r['quality']
        quality_counts[q] = quality_counts.get(q, 0) + 1
    
    print(f'\nBy Quality:')
    for quality in ['strong', 'good', 'moderate', 'weak', 'very_weak']:
        count = quality_counts.get(quality, 0)
        if count > 0:
            print(f'  {quality.title()}: {count}')
    
    # Average strength
    avg_strength = sum(r['strength'] for r in results) / len(results)
    print(f'\nAverage strength: {avg_strength:.1f}')


if __name__ == '__main__':
    print('Starting bulk pattern scan...\n')
    
    # Run scan
    results = scan_all_patterns()
    
    # Print summary
    print_summary(results)
    
    # Export top 30
    top_signals = export_top_signals(results, top_n=30)
    
    # Print top 10
    print(f'\n{"="*60}')
    print(f'TOP 10 STRONGEST SIGNALS')
    print(f'{"="*60}')
    print(f'{"Rank":<5} {"Symbol":<8} {"Pattern":<30} {"Signal":<8} {"Strength":<8}')
    print(f'{"-"*60}')
    
    for idx, signal in enumerate(top_signals[:10], 1):
        print(f'{idx:<5} {signal["symbol"]:<8} {signal["pattern"]:<30} {signal["signal"]:<8} {signal["strength"]:<8}')
    
    print(f'\n✅ Full results saved to top_signals.csv')
