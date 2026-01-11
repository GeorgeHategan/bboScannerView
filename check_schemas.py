#!/usr/bin/env python3
"""
Check schema types.
"""

import duckdb
import os

motherduck_token = os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN', '')

print("=== SCANNER_RESULTS SCHEMA ===")
results_conn = duckdb.connect(f'md:scanner_results?motherduck_token={motherduck_token}')
schema = results_conn.execute("DESCRIBE scanner_results").fetchall()
for col in schema:
    if col[0] in ['scan_date', 'symbol', 'entry_price']:
        print(f"{col[0]}: {col[1]}")
results_conn.close()

print("\n=== DAILY_CACHE SCHEMA ===")
data_conn = duckdb.connect(f'md:scanner_data?motherduck_token={motherduck_token}')
schema = data_conn.execute("DESCRIBE scanner_data.main.daily_cache").fetchall()
for col in schema:
    if col[0] in ['date', 'symbol', 'close']:
        print(f"{col[0]}: {col[1]}")
data_conn.close()
