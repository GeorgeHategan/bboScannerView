#!/usr/bin/env python3
"""
Upload scanner results from local database to MotherDuck.
Run this after fixing the MotherDuck schema.
"""
import duckdb

# Your MotherDuck token (read-write needed)
MOTHERDUCK_TOKEN = "YOUR_READWRITE_TOKEN_HERE"

print("Connecting to local database...")
local_conn = duckdb.connect('/Users/george/scannerPOC/breakoutScannersPOCs/scanner_data.duckdb', read_only=True)

print("Fetching scanner results from local database...")
results = local_conn.execute("""
    SELECT symbol, scanner_name, signal, strength, quality, scan_date
    FROM scanner_data.scanner_results
    ORDER BY scan_date DESC, symbol
""").fetchall()

print(f"Found {len(results)} scanner results")
local_conn.close()

if len(results) == 0:
    print("No data to upload!")
    exit(0)

print(f"\nConnecting to MotherDuck...")
motherduck_conn = duckdb.connect(f'md:scanner_data?motherduck_token={MOTHERDUCK_TOKEN}')

print("Clearing old data from MotherDuck...")
motherduck_conn.execute("DELETE FROM scanner_data.scanner_results")

print(f"Uploading {len(results)} records to MotherDuck...")
motherduck_conn.executemany("""
    INSERT INTO scanner_data.scanner_results 
    (symbol, scanner_name, signal, strength, quality, scan_date)
    VALUES (?, ?, ?, ?, ?, ?)
""", results)

print("✅ Upload complete!")

# Verify
count = motherduck_conn.execute("SELECT COUNT(*) FROM scanner_data.scanner_results").fetchone()[0]
print(f"\nVerification: {count} records in MotherDuck scanner_results table")

# Show sample
print("\nSample data from MotherDuck:")
sample = motherduck_conn.execute("""
    SELECT scanner_name, COUNT(*) as count 
    FROM scanner_data.scanner_results 
    GROUP BY scanner_name 
    ORDER BY count DESC
""").fetchall()
for scanner, cnt in sample:
    print(f"  {scanner}: {cnt} signals")

motherduck_conn.close()
print("\n✅ Done! MotherDuck is now the source of truth.")
