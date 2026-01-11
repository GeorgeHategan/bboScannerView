#!/usr/bin/env python3
"""Test the support_put_wall scanner queries to debug 502 errors."""

import duckdb
import os
from datetime import datetime, timedelta
import json

token = os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN', '')
options_token = os.environ.get('options_motherduck_token') or os.environ.get('OPTIONS_MOTHERDUCK_TOKEN', '')

if not token:
    print("ERROR: motherduck_token not found")
    exit(1)

DUCKDB_PATH = f'md:scanner_results?motherduck_token={token}'
SCANNER_DATA_PATH = f'md:scanner_data?motherduck_token={token}'
OPTIONS_DUCKDB_PATH = f'md:options_data?motherduck_token={options_token}' if options_token else None

conn = duckdb.connect(DUCKDB_PATH)

pattern = 'support_put_wall'
selected_scan_date = '2026-01-10'

print('Step 1: Scanner query...')
date_obj = datetime.strptime(selected_scan_date, '%Y-%m-%d')
next_day = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')

scanner_query = '''
    SELECT symbol, signal_type, COALESCE(signal_strength, 75), 
           COALESCE(setup_stage, 'N/A'), entry_price, picked_by_scanners,
           setup_stage, scan_date, news_sentiment, news_sentiment_label,
           news_relevance, news_headline, news_published, news_url, metadata
    FROM scanner_results
    WHERE scanner_name = ?
    AND scan_date >= CAST(? AS TIMESTAMP) 
    AND scan_date < CAST(? AS TIMESTAMP)
    ORDER BY signal_strength DESC LIMIT 25
'''
results = conn.execute(scanner_query, [pattern, selected_scan_date, next_day]).fetchall()
print(f'  Found {len(results)} results')
symbols_list = [r[0] for r in results]

print('Step 2: Volume query...')
placeholders = ','.join(['?' for _ in symbols_list])
vol_query = f'''
    SELECT dc.symbol, dc.volume, dc.avg_volume_20
    FROM scanner_data.main.daily_cache dc
    INNER JOIN (
        SELECT symbol, MAX(date) as max_date
        FROM scanner_data.main.daily_cache
        WHERE symbol IN ({placeholders})
        GROUP BY symbol
    ) latest ON dc.symbol = latest.symbol AND dc.date = latest.max_date
'''
vol_results = conn.execute(vol_query, symbols_list).fetchall()
print(f'  Found volume for {len(vol_results)} symbols')

print('Step 3: All scanners query...')
all_scanners_query = f'''
    SELECT symbol, scanner_name
    FROM scanner_results
    WHERE symbol IN ({placeholders})
    AND scan_date >= CAST(? AS TIMESTAMP)
    AND scan_date < CAST((CAST(? AS TIMESTAMP) + INTERVAL 1 DAY) AS TIMESTAMP)
    ORDER BY symbol, scanner_name
'''
all_scanner_results = conn.execute(all_scanners_query, symbols_list + [selected_scan_date, selected_scan_date]).fetchall()
print(f'  Found {len(all_scanner_results)} scanner entries')

print('Step 4: Confirmations query...')
confirmations_query = f'''
    SELECT symbol, scanner_name, scan_date, signal_strength
    FROM scanner_results
    WHERE symbol IN ({placeholders})
    AND scanner_name != ?
    AND scan_date >= CURRENT_DATE - INTERVAL 30 DAY
    ORDER BY symbol, scan_date DESC, scanner_name
    LIMIT 200
'''
conf_results = conn.execute(confirmations_query, symbols_list + [pattern]).fetchall()
print(f'  Found {len(conf_results)} confirmations')

if OPTIONS_DUCKDB_PATH:
    options_conn = duckdb.connect(OPTIONS_DUCKDB_PATH)
    
    print('Step 5: Options signals query...')
    options_query = f'''
        SELECT underlying_symbol, signal_date, signal_type, 
               signal_strength, confidence_score, strike, dte,
               premium_spent, notes, direction
        FROM accumulation_signals
        WHERE underlying_symbol IN ({placeholders})
        ORDER BY underlying_symbol, signal_date DESC
    '''
    options_results = options_conn.execute(options_query, symbols_list).fetchall()
    print(f'  Found {len(options_results)} options signals')
    
    print('Step 6: Darkpool signals query...')
    dp_query = f'''
        SELECT ticker, signal_date, signal_type, 
               signal_strength, confidence_score, direction,
               dp_volume, dp_premium, avg_price,
               sell_volume, buy_volume, buy_sell_ratio,
               block_count, avg_block_size, consecutive_days, notes
        FROM darkpool_signals
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, signal_date DESC
    '''
    dp_results = options_conn.execute(dp_query, symbols_list).fetchall()
    print(f'  Found {len(dp_results)} darkpool signals')
    
    print('Step 7: Options walls query...')
    walls_query = f'''
        SELECT underlying_symbol, scan_date, stock_price,
               call_wall_strike, call_wall_oi,
               call_wall_2_strike, call_wall_2_oi,
               call_wall_3_strike, call_wall_3_oi,
               put_wall_strike, put_wall_oi,
               put_wall_2_strike, put_wall_2_oi,
               put_wall_3_strike, put_wall_3_oi,
               total_call_oi, total_put_oi, put_call_ratio
        FROM options_walls
        WHERE underlying_symbol IN ({placeholders})
        AND CAST(scan_date AS DATE) < CAST(? AS DATE)
        QUALIFY ROW_NUMBER() OVER (PARTITION BY underlying_symbol ORDER BY scan_date DESC) = 1
    '''
    walls_results = options_conn.execute(walls_query, symbols_list + [selected_scan_date]).fetchall()
    print(f'  Found {len(walls_results)} options walls')

print('Step 8: Fundamental quality query...')
fund_query = f'''
    SELECT symbol, fund_score, bar_blocks, bar_bucket, dot_state,
           score_components, computed_at
    FROM scanner_data.main.fundamental_quality_scores
    WHERE symbol IN ({placeholders})
'''
fund_results = conn.execute(fund_query, symbols_list).fetchall()
print(f'  Found fund quality for {len(fund_results)} symbols')

print()
print('SUCCESS: All queries completed!')
