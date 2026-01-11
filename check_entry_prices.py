#!/usr/bin/env python3
"""
Check entry_price data in scanner_results.
"""

import duckdb
import os

motherduck_token = os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN', '')
conn = duckdb.connect(f'md:scanner_results?motherduck_token={motherduck_token}')

print("Checking scanner_results table...")

# Check schema
print("\n=== TABLE SCHEMA ===")
schema = conn.execute("DESCRIBE scanner_results").fetchall()
for col in schema:
    print(f"{col[0]}: {col[1]}")

# Check entry_price statistics
print("\n=== ENTRY_PRICE STATISTICS ===")
stats = conn.execute("""
    SELECT 
        COUNT(*) as total_records,
        COUNT(entry_price) as non_null_count,
        COUNT(*) - COUNT(entry_price) as null_count,
        MIN(entry_price) as min_price,
        MAX(entry_price) as max_price,
        AVG(entry_price) as avg_price
    FROM scanner_results
""").fetchone()

print(f"Total records: {stats[0]}")
print(f"Non-NULL entry_price: {stats[1]}")
print(f"NULL entry_price: {stats[2]}")
print(f"Min price: ${stats[3]:.2f}" if stats[3] else "N/A")
print(f"Max price: ${stats[4]:.2f}" if stats[4] else "N/A")
print(f"Avg price: ${stats[5]:.2f}" if stats[5] else "N/A")

# Sample data
print("\n=== SAMPLE DATA (5 records) ===")
samples = conn.execute("""
    SELECT symbol, scanner_name, scan_date, entry_price
    FROM scanner_results
    LIMIT 5
""").fetchall()

for symbol, scanner, date, price in samples:
    print(f"{symbol:8} | {scanner:30} | {date} | ${price if price else 'NULL'}")

conn.close()
