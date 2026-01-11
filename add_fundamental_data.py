#!/usr/bin/env python3
"""
Add fundamental data for symbols to MotherDuck fundamental_cache table.
This will populate sector, industry, and market cap data.
"""

import duckdb
import os

# Your MotherDuck token
MOTHERDUCK_TOKEN = os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN', '')

if not MOTHERDUCK_TOKEN:
    print("ERROR: MOTHERDUCK_TOKEN environment variable not set!")
    exit(1)

# Example fundamental data - you would typically fetch this from an API
# For BXMT (Blackstone Mortgage Trust)
FUNDAMENTAL_DATA = [
    # (symbol, name, market_cap, sector, industry)
    ('BXMT', 'Blackstone Mortgage Trust Inc.', 4_500_000_000, 'Real Estate', 'REIT - Mortgage'),
    # Add more symbols here as needed
]

print("Connecting to MotherDuck scanner_data database...")
conn = duckdb.connect(f'md:scanner_data?motherduck_token={MOTHERDUCK_TOKEN}')

# Check if fundamental_cache table exists
try:
    result = conn.execute("SELECT COUNT(*) FROM scanner_data.fundamental_cache").fetchone()
    print(f"fundamental_cache table exists with {result[0]} records")
except Exception as e:
    print(f"Error: fundamental_cache table may not exist: {e}")
    print("\nCreating fundamental_cache table...")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scanner_data.fundamental_cache (
            symbol VARCHAR PRIMARY KEY,
            name VARCHAR,
            market_cap BIGINT,
            sector VARCHAR,
            industry VARCHAR,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✅ Table created!")

# Insert or update fundamental data
print(f"\nInserting/updating {len(FUNDAMENTAL_DATA)} symbols...")
for symbol, name, market_cap, sector, industry in FUNDAMENTAL_DATA:
    try:
        # Delete existing record if it exists
        conn.execute("DELETE FROM scanner_data.fundamental_cache WHERE symbol = ?", [symbol])
        
        # Insert new record
        conn.execute("""
            INSERT INTO scanner_data.fundamental_cache 
            (symbol, name, market_cap, sector, industry)
            VALUES (?, ?, ?, ?, ?)
        """, [symbol, name, market_cap, sector, industry])
        
        print(f"✅ {symbol}: {name} | Sector: {sector} | Industry: {industry} | Market Cap: ${market_cap:,}")
    except Exception as e:
        print(f"❌ Error inserting {symbol}: {e}")

print("\n✅ Done!")

# Verify
print("\nVerification - Fetching BXMT data:")
result = conn.execute("""
    SELECT symbol, name, market_cap, sector, industry 
    FROM scanner_data.fundamental_cache 
    WHERE symbol = 'BXMT'
""").fetchone()

if result:
    print(f"Symbol: {result[0]}")
    print(f"Name: {result[1]}")
    print(f"Market Cap: ${result[2]:,}")
    print(f"Sector: {result[3]}")
    print(f"Industry: {result[4]}")
else:
    print("BXMT not found in fundamental_cache")

conn.close()
