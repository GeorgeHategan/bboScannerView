#!/usr/bin/env python3
"""
Fix MotherDuck remote database schema.
This should be run with DUCKDB_PATH environment variable set to MotherDuck connection string.
"""
import duckdb
import os
import sys

# Get MotherDuck connection string from environment
motherduck_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Imdlb3JnZS5oYXRlZ2FuQGdtYWlsLmNvbSIsInNlc3Npb24iOiJnZW9yZ2UuaGF0ZWdhbi5nbWFpbC5jb20iLCJwYXQiOiJzRjdZeGV0Zzd4czRfQjJWS2NtVlRuenB4d1VZUWFWUUVwQmZsOVdfYU1nIiwidXNlcklkIjoiMDEzY2E5ZDAtNTU3Yi00YzkzLWIyYWYtNDFjM2MwYWIxMTAyIiwiaXNzIjoibWRfcGF0IiwiaWF0IjoxNzMwNjY2NjkyfQ.0qUEAR0C5XbRDlI6OXWvvVZm8TmF8o-4KjKJ2XL_xxw"

connection_string = f'md:scanner_data?motherduck_token={motherduck_token}'

print(f"Connecting to MotherDuck...")
try:
    conn = duckdb.connect(connection_string)
    
    print("✅ Connected to MotherDuck")
    
    # Check current schema
    print("\nChecking scanner_results table...")
    try:
        schema = conn.execute("DESCRIBE scanner_data.scanner_results").fetchall()
        print("Current schema:")
        for col in schema:
            print(f"  {col[0]}: {col[1]}")
        
        # Check if we need to fix it
        column_names = [col[0] for col in schema]
        
        if 'signal_type' in column_names or 'signal_strength' in column_names or 'news_headline' in column_names:
            print("\n⚠️  WRONG SCHEMA! This is an old or different table.")
            print("Dropping and recreating with correct schema...")
            
            # Drop old table
            conn.execute("DROP TABLE IF EXISTS scanner_data.scanner_results")
            print("✅ Dropped old table")
        else:
            print("\n✅ Schema looks correct, but let's verify columns match...")
            expected_cols = ['symbol', 'scanner_name', 'signal', 'strength', 'quality', 'scan_date']
            if column_names != expected_cols:
                print(f"⚠️  Column mismatch! Expected: {expected_cols}, Got: {column_names}")
                print("Dropping and recreating...")
                conn.execute("DROP TABLE IF EXISTS scanner_data.scanner_results")
                print("✅ Dropped old table")
            else:
                print("✅ Schema is perfect! No changes needed.")
                conn.close()
                sys.exit(0)
    
    except Exception as e:
        if "does not exist" in str(e) or "not found" in str(e):
            print("Table doesn't exist yet. Will create it.")
        else:
            print(f"Error checking table: {e}")
    
    # Create table with correct schema
    print("\nCreating scanner_results table with correct schema...")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scanner_data.scanner_results (
            symbol VARCHAR,
            scanner_name VARCHAR,
            signal VARCHAR,
            strength DOUBLE,
            quality VARCHAR,
            scan_date DATE,
            PRIMARY KEY (symbol, scanner_name, scan_date)
        )
    """)
    print("✅ Created table")
    
    # Verify
    schema = conn.execute("DESCRIBE scanner_data.scanner_results").fetchall()
    print("\nVerified new schema:")
    for col in schema:
        print(f"  {col[0]}: {col[1]}")
    
    print("\n✅ MotherDuck schema fixed!")
    print("\nNext steps:")
    print("1. Run save_scanner_results_to_db.py with DUCKDB_PATH set to MotherDuck")
    print("2. Redeploy on Render")
    
    conn.close()
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
