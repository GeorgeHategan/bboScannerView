import duckdb
import os

# Connect to the database
conn = duckdb.connect('md:scanner_data?motherduck_token=' + os.getenv('MOTHERDUCK_TOKEN'))

# Check if SQM appears in scanner results
results = conn.execute("""
    SELECT scanner_name, scan_date
    FROM scanner_results.scanner_results
    WHERE symbol = 'SQM'
    AND scan_date >= CURRENT_DATE - INTERVAL 30 DAY
    ORDER BY scan_date DESC, scanner_name
""").fetchall()

print("SQM Scanner Results (last 30 days):")
print("=" * 60)
if results:
    for scanner, date in results:
        print(f"{date}: {scanner}")
    unique_scanners = set(r[0] for r in results)
    print(f"\nTotal unique scanners: {len(unique_scanners)}")
    print(f"Scanners: {', '.join(unique_scanners)}")
else:
    print("❌ No results found for SQM in the last 30 days")

# Check the most recent scan date in the database
latest = conn.execute("""
    SELECT MAX(scan_date) as latest_date
    FROM scanner_results.scanner_results
""").fetchone()
print(f"\nLatest scan_date in database: {latest[0]}")

conn.close()
