#!/usr/bin/env python3
"""
Fix yoy_eps_growth values in fundamental_quality_scores table.
Divide by 100 to convert percentage to decimal to match other fields.
"""
import duckdb
import json
import os

# Connect to MotherDuck
conn = duckdb.connect('md:scanner_data')

print("Checking current yoy_eps_growth values...")

# Check a few samples before fix
sample_query = """
    SELECT symbol, score_components 
    FROM main.fundamental_quality_scores 
    WHERE score_version = 'fq_v2' 
    AND json_extract(score_components, '$.raw_inputs.yoy_eps_growth') IS NOT NULL
    LIMIT 5
"""
samples_before = conn.execute(sample_query).fetchall()

print("\nBEFORE FIX - Sample values:")
for row in samples_before:
    symbol = row[0]
    components = json.loads(row[1])
    yoy = components['raw_inputs'].get('yoy_eps_growth')
    print(f"  {symbol}: yoy_eps_growth = {yoy}")

# Count total rows to update
count_query = """
    SELECT COUNT(*) 
    FROM main.fundamental_quality_scores 
    WHERE score_version = 'fq_v2'
    AND json_extract(score_components, '$.raw_inputs.yoy_eps_growth') IS NOT NULL
"""
count = conn.execute(count_query).fetchone()[0]
print(f"\nTotal rows to update: {count}")

# Perform the update by fetching, modifying, and updating each row
print("\nUpdating yoy_eps_growth values (dividing by 100)...")

fetch_query = """
    SELECT symbol, score_components 
    FROM main.fundamental_quality_scores 
    WHERE score_version = 'fq_v2'
    AND json_extract(score_components, '$.raw_inputs.yoy_eps_growth') IS NOT NULL
"""

rows = conn.execute(fetch_query).fetchall()
updated_count = 0

for row in rows:
    symbol = row[0]
    components = json.loads(row[1])
    
    # Fix yoy_eps_growth by dividing by 100
    if 'raw_inputs' in components and 'yoy_eps_growth' in components['raw_inputs']:
        old_value = components['raw_inputs']['yoy_eps_growth']
        if old_value is not None:
            components['raw_inputs']['yoy_eps_growth'] = old_value / 100.0
            
            # Update the row
            update_query = """
                UPDATE main.fundamental_quality_scores
                SET score_components = ?
                WHERE symbol = ? AND score_version = 'fq_v2'
            """
            conn.execute(update_query, [json.dumps(components), symbol])
            updated_count += 1
            
            if updated_count % 100 == 0:
                print(f"  Updated {updated_count}/{len(rows)} rows...")

print(f"Update complete! Updated {updated_count} rows.")

# Verify the fix
samples_after = conn.execute(sample_query).fetchall()

print("\nAFTER FIX - Sample values:")
for row in samples_after:
    symbol = row[0]
    components = json.loads(row[1])
    yoy = components['raw_inputs'].get('yoy_eps_growth')
    qeg = components['raw_inputs'].get('quarterly_earnings_growth')
    print(f"  {symbol}: yoy_eps_growth = {yoy:.4f}, quarterly_earnings_growth = {qeg:.4f if qeg else None}")

# Verify no values are still in percentage format (> 100)
verify_query = """
    SELECT symbol, json_extract(score_components, '$.raw_inputs.yoy_eps_growth') as yoy
    FROM main.fundamental_quality_scores 
    WHERE score_version = 'fq_v2'
    AND json_extract(score_components, '$.raw_inputs.yoy_eps_growth') IS NOT NULL
    AND ABS(json_extract(score_components, '$.raw_inputs.yoy_eps_growth')) > 10
"""
large_values = conn.execute(verify_query).fetchall()

if large_values:
    print(f"\n⚠️  WARNING: Found {len(large_values)} symbols with |yoy_eps_growth| > 10 (may still be percentages):")
    for row in large_values[:10]:
        print(f"  {row[0]}: {row[1]}")
else:
    print("\n✅ All yoy_eps_growth values are now in decimal format!")

conn.close()
print("\nDone!")
