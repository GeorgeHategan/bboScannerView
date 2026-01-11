#!/usr/bin/env python3
"""
Check scan date distribution in scanner_results.
"""

import duckdb
import os
from datetime import datetime

motherduck_token = os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN', '')
conn = duckdb.connect(f'md:scanner_results?motherduck_token={motherduck_token}')

print("=== SCAN DATE ANALYSIS ===\n")

# Get date range
date_range = conn.execute("""
    SELECT 
        MIN(scan_date) as earliest,
        MAX(scan_date) as latest,
        COUNT(DISTINCT scan_date) as unique_dates
    FROM scanner_results
""").fetchone()

print(f"Earliest scan: {date_range[0]}")
print(f"Latest scan: {date_range[1]}")
print(f"Unique scan dates: {date_range[2]}")
print(f"Today: {datetime.now().date()}")

# Check distribution
print("\n=== SCAN DATE DISTRIBUTION ===")
dist = conn.execute("""
    SELECT 
        scan_date,
        COUNT(*) as count
    FROM scanner_results
    GROUP BY scan_date
    ORDER BY scan_date DESC
    LIMIT 10
""").fetchall()

for date, count in dist:
    print(f"{date}: {count} picks")

# Check if there's price data after recent scans
print("\n=== CHECKING PRICE DATA AVAILABILITY ===")
recent_scan = conn.execute("SELECT MAX(scan_date) FROM scanner_results").fetchone()[0]
print(f"Most recent scan date: {recent_scan}")

# Switch to scanner_data database
data_conn = duckdb.connect(f'md:scanner_data?motherduck_token={motherduck_token}')

latest_price_date = data_conn.execute("SELECT MAX(date) FROM scanner_data.main.daily_cache").fetchone()[0]
print(f"Latest price data: {latest_price_date}")

if latest_price_date:
    days_diff = (latest_price_date - recent_scan).days if isinstance(latest_price_date, type(recent_scan)) else 0
    print(f"Days of price history after scan: {days_diff}")

data_conn.close()
conn.close()
