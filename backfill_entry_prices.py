#!/usr/bin/env python3
"""
Backfill entry_price for existing scanner_results records.
Looks up closing price from daily_cache on the scan_date.
"""

import duckdb
import os

# Get MotherDuck token (try both lowercase and uppercase)
motherduck_token = os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN', '')
if not motherduck_token:
    print("Error: motherduck_token or MOTHERDUCK_TOKEN environment variable not set")
    exit(1)

print("Connecting to MotherDuck scanner_results database...")
conn = duckdb.connect(f'md:scanner_results?motherduck_token={motherduck_token}')

try:
    # First, add entry_price column if it doesn't exist
    print("\nEnsuring entry_price column exists...")
    try:
        conn.execute("""
            ALTER TABLE scanner_results 
            ADD COLUMN IF NOT EXISTS entry_price DOUBLE
        """)
        print("✅ Column added/verified")
    except Exception as e:
        print(f"Column may already exist: {e}")
    
    # Count records needing backfill
    print("\nChecking records that need entry_price...")
    count = conn.execute("""
        SELECT COUNT(*) 
        FROM scanner_results 
        WHERE entry_price IS NULL
    """).fetchone()[0]
    
    print(f"Found {count} records with NULL entry_price")
    
    if count == 0:
        print("✅ Nothing to backfill!")
        conn.close()
        exit(0)
    
    # Backfill using a JOIN with daily_cache
    print("\nBackfilling entry_price from daily_cache...")
    
    updated = conn.execute("""
        UPDATE scanner_results sr
        SET entry_price = (
            SELECT dc.close
            FROM scanner_data.main.daily_cache dc
            WHERE dc.symbol = sr.symbol
            AND dc.date = sr.scan_date
            LIMIT 1
        )
        WHERE sr.entry_price IS NULL
    """)
    
    print(f"✅ Update complete!")
    
    # Verify results
    print("\nVerifying backfill...")
    still_null = conn.execute("""
        SELECT COUNT(*) 
        FROM scanner_results 
        WHERE entry_price IS NULL
    """).fetchone()[0]
    
    filled = conn.execute("""
        SELECT COUNT(*) 
        FROM scanner_results 
        WHERE entry_price IS NOT NULL
    """).fetchone()[0]
    
    print(f"Records with entry_price: {filled}")
    print(f"Records still NULL: {still_null}")
    
    if still_null > 0:
        print(f"\n⚠️  Warning: {still_null} records couldn't be backfilled (no matching price data)")
        
        # Show sample of failed records
        print("\nSample of records that failed:")
        failed = conn.execute("""
            SELECT symbol, scan_date, scanner_name
            FROM scanner_results
            WHERE entry_price IS NULL
            LIMIT 10
        """).fetchall()
        
        for symbol, scan_date, scanner in failed:
            print(f"  {symbol} - {scan_date} - {scanner}")
    
    print("\n✅ Backfill complete!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    conn.close()
