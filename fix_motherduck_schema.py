#!/usr/bin/env python3
"""
Fix MotherDuck scanner_results table schema.
Drop old table if exists and recreate with correct schema.
"""
import duckdb
import os

# Get MotherDuck connection string from environment
DUCKDB_PATH = os.environ.get('DUCKDB_PATH', '/Users/george/scannerPOC/breakoutScannersPOCs/scanner_data.duckdb')

print(f"Connecting to: {DUCKDB_PATH}")
conn = duckdb.connect(DUCKDB_PATH)

try:
    # Check if scanner_results exists
    tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'scanner_data'").fetchall()
    print(f"\nTables in scanner_data schema:")
    for table in tables:
        print(f"  - {table[0]}")
    
    # Check current schema if table exists
    try:
        schema = conn.execute("DESCRIBE scanner_data.scanner_results").fetchall()
        print(f"\nCurrent scanner_results schema:")
        for col in schema:
            print(f"  {col[0]}: {col[1]}")
        
        # Check if schema is wrong
        column_names = [col[0] for col in schema]
        if 'signal_type' in column_names or 'signal_strength' in column_names:
            print("\n⚠️  OLD SCHEMA DETECTED! Dropping and recreating table...")
            
            # Drop old table
            conn.execute("DROP TABLE IF EXISTS scanner_data.scanner_results")
            print("✅ Dropped old table")
            
            # Create new table with correct schema
            conn.execute("""
                CREATE TABLE scanner_data.scanner_results (
                    symbol VARCHAR,
                    scanner_name VARCHAR,
                    signal VARCHAR,
                    strength DOUBLE,
                    quality VARCHAR,
                    scan_date DATE,
                    PRIMARY KEY (symbol, scanner_name, scan_date)
                )
            """)
            print("✅ Created new table with correct schema")
            
            # Verify new schema
            new_schema = conn.execute("DESCRIBE scanner_data.scanner_results").fetchall()
            print("\nNew schema:")
            for col in new_schema:
                print(f"  {col[0]}: {col[1]}")
        else:
            print("\n✅ Schema is correct!")
            
    except Exception as e:
        if "does not exist" in str(e) or "not found" in str(e):
            print("\nTable doesn't exist. Creating it...")
            
            # Create table
            conn.execute("""
                CREATE TABLE scanner_data.scanner_results (
                    symbol VARCHAR,
                    scanner_name VARCHAR,
                    signal VARCHAR,
                    strength DOUBLE,
                    quality VARCHAR,
                    scan_date DATE,
                    PRIMARY KEY (symbol, scanner_name, scan_date)
                )
            """)
            print("✅ Created scanner_results table")
        else:
            raise
    
    print("\n✅ Schema fix complete!")
    print("\nNext steps:")
    print("1. Run save_scanner_results_to_db.py to populate the table")
    print("2. Deploy to Render")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    conn.close()
