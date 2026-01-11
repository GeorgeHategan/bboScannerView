#!/usr/bin/env python3
"""
Debug why scanner performance isn't showing data.
"""

import duckdb
import os

motherduck_token = os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN', '')

# Connect to scanner_results
print("Connecting to scanner_results database...")
results_conn = duckdb.connect(f'md:scanner_results?motherduck_token={motherduck_token}')

# Get one scanner to test
print("\n=== TESTING ONE SCANNER ===")
scanner_name = results_conn.execute("""
    SELECT scanner_name, COUNT(*) as cnt
    FROM scanner_results
    WHERE scan_date < CURRENT_DATE
    GROUP BY scanner_name
    ORDER BY cnt DESC
    LIMIT 1
""").fetchone()

print(f"Scanner: {scanner_name[0]}")
print(f"Total picks (excluding today): {scanner_name[1]}")

# Get sample picks
print("\n=== SAMPLE PICKS ===")
picks = results_conn.execute("""
    SELECT symbol, scan_date, entry_price
    FROM scanner_results
    WHERE scanner_name = ?
    AND scan_date < CURRENT_DATE
    AND entry_price IS NOT NULL
    LIMIT 5
""", [scanner_name[0]]).fetchall()

for symbol, scan_date, entry_price in picks:
    print(f"{symbol}: {scan_date} @ ${entry_price:.2f}")
    
    # Check if there's price data after this scan_date
    data_conn = duckdb.connect(f'md:scanner_data?motherduck_token={motherduck_token}')
    
    price_count = data_conn.execute("""
        SELECT COUNT(*)
        FROM scanner_data.main.daily_cache
        WHERE symbol = ?
        AND date > ?
    """, [symbol, scan_date]).fetchone()[0]
    
    print(f"  → Price data rows after {scan_date}: {price_count}")
    
    if price_count > 0:
        sample_prices = data_conn.execute("""
            SELECT date, close
            FROM scanner_data.main.daily_cache
            WHERE symbol = ?
            AND date > ?
            ORDER BY date
            LIMIT 3
        """, [symbol, scan_date]).fetchall()
        
        for date, close in sample_prices:
            print(f"     {date}: ${close:.2f}")
    
    data_conn.close()

results_conn.close()
