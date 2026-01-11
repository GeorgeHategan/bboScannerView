#!/usr/bin/env python3
"""
List all tables in scanner_data database to see what's available.
"""

import duckdb
import os

MOTHERDUCK_TOKEN = os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN', '')

if not MOTHERDUCK_TOKEN:
    print("ERROR: MOTHERDUCK_TOKEN not set!")
    exit(1)

print("Connecting to MotherDuck scanner_data...")
conn = duckdb.connect(f'md:scanner_data?motherduck_token={MOTHERDUCK_TOKEN}')

print("\n=== All tables in scanner_data database ===")
tables = conn.execute("""
    SELECT table_schema, table_name 
    FROM information_schema.tables 
    WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
    ORDER BY table_schema, table_name
""").fetchall()

for schema, table in tables:
    print(f"  {schema}.{table}")
    
    # Show row count for each table
    try:
        count = conn.execute(f"SELECT COUNT(*) FROM {schema}.{table}").fetchone()[0]
        print(f"    → {count:,} rows")
    except:
        print(f"    → (unable to count)")

print("\n=== Checking for 'fundamental' in table/column names ===")
# Search for any table or column with 'fundamental' in the name
all_columns = conn.execute("""
    SELECT table_schema, table_name, column_name 
    FROM information_schema.columns 
    WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
    AND (table_name LIKE '%fundamental%' OR column_name LIKE '%fundamental%')
""").fetchall()

if all_columns:
    for schema, table, col in all_columns:
        print(f"  {schema}.{table}.{col}")
else:
    print("  None found")

print("\n=== Checking for columns like 'sector', 'industry', 'market_cap' ===")
relevant_cols = conn.execute("""
    SELECT DISTINCT table_schema, table_name, column_name 
    FROM information_schema.columns 
    WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
    AND (column_name IN ('sector', 'industry', 'market_cap', 'name', 'company'))
    ORDER BY table_schema, table_name, column_name
""").fetchall()

if relevant_cols:
    for schema, table, col in relevant_cols:
        print(f"  {schema}.{table}.{col}")
else:
    print("  None found")

conn.close()
