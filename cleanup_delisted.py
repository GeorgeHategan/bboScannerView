#!/usr/bin/env python3
"""
Clean up delisted or inactive stocks from the screener.
Removes stocks that haven't had trading data in the last 30 days.
"""

import os
import pandas
from datetime import datetime, timedelta

def cleanup_delisted_stocks():
    """Remove delisted stocks from CSV files and symbols list."""
    
    daily_dir = 'datasets/daily'
    symbols_file = 'datasets/symbols.csv'
    
    # Track valid and invalid symbols
    valid_symbols = []
    removed_symbols = []
    
    # Read current symbols
    with open(symbols_file, 'r') as f:
        symbols_data = [line.strip().split(',', 1) for line in f if ',' in line]
    
    # Check each stock's data
    for symbol_data in symbols_data:
        if len(symbol_data) != 2:
            continue
        symbol, company = symbol_data
        csv_file = os.path.join(daily_dir, f'{symbol}.csv')
        
        if not os.path.exists(csv_file):
            print(f'❌ {symbol}: CSV file not found')
            removed_symbols.append((symbol, company, 'No CSV file'))
            continue
        
        try:
            df = pandas.read_csv(csv_file)
            
            # Find date column
            date_col = 'date' if 'date' in df.columns else 'Date' if 'Date' in df.columns else df.columns[0]
            
            if date_col not in df.columns or len(df) == 0:
                print(f'❌ {symbol}: Invalid or empty data')
                removed_symbols.append((symbol, company, 'Invalid data'))
                os.remove(csv_file)
                continue
            
            # Check latest date
            df[date_col] = pandas.to_datetime(df[date_col])
            latest_date = df[date_col].max()
            days_old = (datetime.now() - latest_date).days
            
            if days_old > 30:
                print(f'❌ {symbol}: Data is {days_old} days old (last: {latest_date.date()})')
                removed_symbols.append((symbol, company, f'{days_old} days old'))
                os.remove(csv_file)
            else:
                print(f'✅ {symbol}: Active (last: {latest_date.date()})')
                valid_symbols.append((symbol, company))
                
        except Exception as e:
            print(f'❌ {symbol}: Error reading data - {e}')
            removed_symbols.append((symbol, company, str(e)))
            if os.path.exists(csv_file):
                os.remove(csv_file)
    
    # Write cleaned symbols list
    with open(symbols_file, 'w') as f:
        for symbol, company in valid_symbols:
            f.write(f'{symbol},{company}\n')
    
    # Print summary
    print('\n' + '='*60)
    print(f'✅ Valid symbols: {len(valid_symbols)}')
    print(f'❌ Removed symbols: {len(removed_symbols)}')
    print('='*60)
    
    if removed_symbols:
        print('\nRemoved stocks:')
        for symbol, company, reason in removed_symbols:
            print(f'  {symbol} ({company}) - {reason}')

if __name__ == '__main__':
    cleanup_delisted_stocks()
