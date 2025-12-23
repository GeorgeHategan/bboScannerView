#!/usr/bin/env python3
"""
Pre-calculate scanner performance metrics and store in scanner_data.performance_tracking table.
Run this daily/weekly to update performance stats.
"""

import duckdb
import os
from datetime import datetime

# Get tokens
motherduck_token = os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN', '')

# Connect to both databases
results_conn = duckdb.connect(f'md:scanner_results?motherduck_token={motherduck_token}')
data_conn = duckdb.connect(f'md:scanner_data?motherduck_token={motherduck_token}')

print(f"Starting performance calculation at {datetime.now()}")

# Drop and recreate performance_tracking table with aggregated schema
data_conn.execute("DROP TABLE IF EXISTS performance_tracking")
data_conn.execute("""
    CREATE TABLE performance_tracking (
        scanner_name VARCHAR PRIMARY KEY,
        total_picks INTEGER,
        avg_max_gain DOUBLE,
        avg_drawdown DOUBLE,
        avg_current_pnl DOUBLE,
        win_rate DOUBLE,
        best_symbol VARCHAR,
        best_gain DOUBLE,
        worst_symbol VARCHAR,
        worst_drawdown DOUBLE,
        calculated_at TIMESTAMP
    )
""")

# Get all scanners
scanners = results_conn.execute("""
    SELECT DISTINCT scanner_name 
    FROM scanner_results 
    ORDER BY scanner_name
""").fetchall()

print(f"Processing {len(scanners)} scanners...")

for (scanner_name,) in scanners:
    print(f"\n  Processing {scanner_name}...")
    
    # Get all picks (no limit - process all historical data)
    picks = results_conn.execute("""
        SELECT symbol, scan_date, entry_price
        FROM scanner_results
        WHERE scanner_name = ?
        AND entry_price IS NOT NULL
        AND scan_date < CURRENT_DATE
        ORDER BY scan_date DESC
    """, [scanner_name]).fetchall()
    
    if not picks:
        print(f"    No picks found")
        continue
    
    print(f"    Analyzing {len(picks)} picks...")
    
    gains = []
    drawdowns = []
    current_pnls = []
    best_pick = None
    worst_pick = None
    max_best_gain = -999999
    max_worst_dd = 0
    
    for symbol, scan_date, entry_price in picks:
        if not entry_price or entry_price <= 0:
            continue
        
        # Get price history
        prices = data_conn.execute("""
            SELECT close, high, low
            FROM main.daily_cache
            WHERE symbol = ?
            AND date > CAST(? AS VARCHAR)
            ORDER BY date
            LIMIT 60
        """, [symbol, str(scan_date)]).fetchall()
        
        if not prices:
            continue
        
        # Calculate metrics
        current_price = prices[-1][0]
        max_gain = max(((p[1] - entry_price) / entry_price * 100) for p in prices)
        
        peak = entry_price
        max_dd = 0
        for close, high, low in prices:
            peak = max(peak, high)
            if peak > entry_price:
                dd = ((low - peak) / peak * 100)
                max_dd = min(max_dd, dd)
        
        current_pnl = ((current_price - entry_price) / entry_price * 100)
        
        gains.append(max_gain)
        drawdowns.append(max_dd)
        current_pnls.append(current_pnl)
        
        if max_gain > max_best_gain:
            max_best_gain = max_gain
            best_pick = symbol
        
        if max_dd < max_worst_dd:
            max_worst_dd = max_dd
            worst_pick = symbol
    
    if not gains:
        print(f"    No valid performance data")
        continue
    
    # Calculate aggregates
    total_picks = len(gains)
    avg_max_gain = sum(gains) / total_picks
    avg_drawdown = sum(drawdowns) / total_picks
    avg_current_pnl = sum(current_pnls) / total_picks
    win_rate = (len([p for p in current_pnls if p > 0]) / total_picks * 100)
    
    print(f"    Stats: {total_picks} picks, {avg_max_gain:.1f}% avg gain, {win_rate:.1f}% win rate")
    
    # Insert into performance_tracking table in scanner_data
    data_conn.execute("""
        INSERT OR REPLACE INTO performance_tracking VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, [
        scanner_name,
        total_picks,
        avg_max_gain,
        avg_drawdown,
        avg_current_pnl,
        win_rate,
        best_pick,
        max_best_gain,
        worst_pick,
        max_worst_dd
    ])

results_conn.close()
data_conn.close()

print(f"\n✅ Performance calculation complete at {datetime.now()}")
