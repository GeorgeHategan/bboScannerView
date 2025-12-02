#!/usr/bin/env python3
"""
Upload historical VIX data from CSV to MotherDuck options_data database.
"""

import duckdb
import pandas as pd
import os
from datetime import datetime

# MotherDuck token for options database
OPTIONS_MOTHERDUCK_TOKEN = os.environ.get('options_motherduck_token') or os.environ.get('OPTIONS_MOTHERDUCK_TOKEN', '')

if not OPTIONS_MOTHERDUCK_TOKEN:
    print("❌ ERROR: options_motherduck_token environment variable not set!")
    print("Set it with: export options_motherduck_token='your_token_here'")
    exit(1)

# CSV file path
CSV_PATH = os.path.join(os.path.dirname(__file__), 'CBOE_VIX, 60_fbf8b.csv')

print(f"📂 Loading VIX data from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

print(f"📊 Found {len(df)} rows in CSV")
print(f"📋 Columns: {list(df.columns)}")

# Extract relevant columns
# time, VIX spot, VX30 synthetic contract (second occurrence = .1 in pandas)
vix_df = df[['time', 'VIX spot', 'VX30 synthetic contract.1']].copy()
vix_df.columns = ['timestamp', 'vix', 'vx30']

# Convert timestamp to proper format
vix_df['timestamp'] = pd.to_datetime(vix_df['timestamp'])

# Drop rows with NaN values in vix or vx30
initial_count = len(vix_df)
vix_df = vix_df.dropna(subset=['vix', 'vx30'])
print(f"📉 Dropped {initial_count - len(vix_df)} rows with NaN values")
print(f"✅ {len(vix_df)} valid rows to upload")

# Show sample data
print("\n📋 Sample data:")
print(vix_df.head())
print(f"\n📅 Date range: {vix_df['timestamp'].min()} to {vix_df['timestamp'].max()}")

# Connect to MotherDuck
print(f"\n🔌 Connecting to MotherDuck options_data database...")
conn = duckdb.connect(f'md:options_data?motherduck_token={OPTIONS_MOTHERDUCK_TOKEN}')

# Check if vix_data table exists
try:
    existing = conn.execute("SELECT COUNT(*) FROM vix_data").fetchone()[0]
    print(f"📊 Current vix_data table has {existing} rows")
except:
    print("📊 vix_data table doesn't exist or is empty, will create it")
    # Create the table if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vix_data (
            id INTEGER PRIMARY KEY,
            timestamp TIMESTAMP,
            vix DOUBLE,
            vx30 DOUBLE,
            source VARCHAR DEFAULT 'csv_import',
            notes VARCHAR
        )
    """)

# Ask for confirmation
print("\n⚠️  This will INSERT new VIX data into the database.")
response = input("Continue? (y/n): ").strip().lower()

if response != 'y':
    print("❌ Aborted.")
    conn.close()
    exit(0)

# Clear existing data (optional - comment out if you want to append)
print("\n🗑️  Clearing existing vix_data...")
conn.execute("DELETE FROM vix_data")

# Insert data with IDs
print(f"📤 Uploading {len(vix_df)} records...")

# Add id column and source
vix_df['id'] = range(1, len(vix_df) + 1)
vix_df['source'] = 'csv_import'
vix_df['notes'] = 'Historical data from BATS_NOK CSV'

# Insert using DuckDB's efficient method
conn.execute("""
    INSERT INTO vix_data (id, timestamp, vix, vx30, source, notes)
    SELECT id, timestamp, vix, vx30, source, notes FROM vix_df
""")

print("✅ Upload complete!")

# Verify
count = conn.execute("SELECT COUNT(*) FROM vix_data").fetchone()[0]
print(f"\n📊 Verification: {count} records in vix_data table")

# Show sample from database
print("\n📋 Sample data from MotherDuck:")
sample = conn.execute("""
    SELECT timestamp, vix, vx30, source 
    FROM vix_data 
    ORDER BY timestamp DESC 
    LIMIT 5
""").fetchall()

for row in sample:
    print(f"  {row[0]} | VIX: {row[1]:.2f} | VX30: {row[2]:.2f} | {row[3]}")

# Show date range
date_range = conn.execute("""
    SELECT MIN(timestamp), MAX(timestamp) FROM vix_data
""").fetchone()
print(f"\n📅 Date range in DB: {date_range[0]} to {date_range[1]}")

conn.close()
print("\n✅ Done! VIX data is now in MotherDuck.")
