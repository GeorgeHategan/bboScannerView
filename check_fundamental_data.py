#!/usr/bin/env python3
"""
Check fundamental_cache table in MotherDuck to diagnose missing data.
"""

import duckdb
import os

MOTHERDUCK_TOKEN = os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN', '')

if not MOTHERDUCK_TOKEN:
    print("ERROR: MOTHERDUCK_TOKEN not set!")
    exit(1)

print("Connecting to MotherDuck scanner_data...")
conn = duckdb.connect(f'md:scanner_data?motherduck_token={MOTHERDUCK_TOKEN}')

# Check if table exists
print("\n=== Checking if fundamental_cache table exists ===")
try:
    tables = conn.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'scanner_data' AND table_name = 'fundamental_cache'
    """).fetchall()
    
    if tables:
        print("✅ fundamental_cache table exists")
    else:
        print("❌ fundamental_cache table NOT found!")
        print("\nAvailable tables in scanner_data schema:")
        all_tables = conn.execute("""
            SELECT table_name FROM information_schema.tables WHERE table_schema = 'scanner_data'
        """).fetchall()
        for t in all_tables:
            print(f"  - {t[0]}")
        exit(1)
except Exception as e:
    print(f"Error checking tables: {e}")
    exit(1)

# Check table structure
print("\n=== Table structure ===")
try:
    cols = conn.execute("DESCRIBE scanner_data.fundamental_cache").fetchall()
    for col in cols:
        print(f"  {col[0]}: {col[1]}")
except Exception as e:
    print(f"Error: {e}")

# Check row count
print("\n=== Row count ===")
count = conn.execute("SELECT COUNT(*) FROM scanner_data.fundamental_cache").fetchone()[0]
print(f"Total records: {count}")

# Check if BXMT exists
print("\n=== Checking BXMT specifically ===")
bxmt = conn.execute("""
    SELECT symbol, name, market_cap, sector, industry 
    FROM scanner_data.fundamental_cache 
    WHERE symbol = 'BXMT'
""").fetchone()

if bxmt:
    print(f"✅ BXMT found:")
    print(f"  Symbol: {bxmt[0]}")
    print(f"  Name: {bxmt[1]}")
    print(f"  Market Cap: {bxmt[2]}")
    print(f"  Sector: {bxmt[3]}")
    print(f"  Industry: {bxmt[4]}")
else:
    print("❌ BXMT NOT found in fundamental_cache")

# Show sample records
print("\n=== Sample records (first 10) ===")
samples = conn.execute("""
    SELECT symbol, name, sector, industry, market_cap 
    FROM scanner_data.fundamental_cache 
    ORDER BY symbol 
    LIMIT 10
""").fetchall()

for s in samples:
    print(f"  {s[0]}: {s[1]} | Sector: {s[2]} | Industry: {s[3]} | Market Cap: {s[4]}")

# Check for NULL values
print("\n=== Checking for NULL values ===")
null_counts = conn.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN name IS NULL THEN 1 ELSE 0 END) as null_name,
        SUM(CASE WHEN sector IS NULL THEN 1 ELSE 0 END) as null_sector,
        SUM(CASE WHEN industry IS NULL THEN 1 ELSE 0 END) as null_industry,
        SUM(CASE WHEN market_cap IS NULL THEN 1 ELSE 0 END) as null_market_cap
    FROM scanner_data.fundamental_cache
""").fetchone()

print(f"  Total records: {null_counts[0]}")
print(f"  NULL name: {null_counts[1]}")
print(f"  NULL sector: {null_counts[2]}")
print(f"  NULL industry: {null_counts[3]}")
print(f"  NULL market_cap: {null_counts[4]}")

# Show symbols with NULL sector
print("\n=== Symbols with NULL sector (first 20) ===")
null_sectors = conn.execute("""
    SELECT symbol, name 
    FROM scanner_data.fundamental_cache 
    WHERE sector IS NULL 
    LIMIT 20
""").fetchall()

for ns in null_sectors:
    print(f"  {ns[0]}: {ns[1]}")

conn.close()
print("\n✅ Analysis complete!")
