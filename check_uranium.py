#!/usr/bin/env python3
"""Check if URANIUM industry exists in the data."""
import duckdb
import os

SCANNER_DATA_PATH = os.getenv('SCANNER_DATA_PATH', 'md:scanner_data')

conn = duckdb.connect(SCANNER_DATA_PATH)

# Check for URANIUM in RRG data
print("Checking for URANIUM in v_rrg_latest...")
result = conn.execute("""
    SELECT group_name, group_type, rs_ratio, rs_momentum, members_count
    FROM v_rrg_latest
    WHERE UPPER(group_name) LIKE '%URANIUM%'
    OR UPPER(group_name) LIKE '%URAN%'
""").fetchall()

if result:
    print("Found URANIUM-related entries:")
    for row in result:
        print(f"  {row}")
else:
    print("No URANIUM found in v_rrg_latest")

# Check fundamental_cache for URANIUM industry
print("\nChecking fundamental_cache for URANIUM industry...")
result = conn.execute("""
    SELECT DISTINCT industry, sector, COUNT(*) as count
    FROM scanner_data.main.fundamental_cache
    WHERE UPPER(industry) LIKE '%URANIUM%'
    OR UPPER(industry) LIKE '%URAN%'
    GROUP BY industry, sector
""").fetchall()

if result:
    print("Found URANIUM in fundamental_cache:")
    for row in result:
        print(f"  Industry: {row[0]}, Sector: {row[1]}, Count: {row[2]}")
else:
    print("No URANIUM found in fundamental_cache")

# Check all unique industries
print("\nAll industries in v_rrg_latest (type=industry):")
result = conn.execute("""
    SELECT group_name, members_count
    FROM v_rrg_latest
    WHERE group_type = 'industry'
    ORDER BY group_name
""").fetchall()

print(f"Total industries: {len(result)}")
for row in result:
    if 'URAN' in row[0].upper() or 'URANI' in row[0].upper():
        print(f"  *** {row[0]} ({row[1]} members)")

conn.close()
