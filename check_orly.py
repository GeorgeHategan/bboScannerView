import duckdb

conn = duckdb.connect('md:scanner_data')

# Count ORLY occurrences
result = conn.execute("""
    SELECT COUNT(*) as count
    FROM scanner_results.scanner_results
    WHERE symbol = 'ORLY'
""").fetchone()
print(f'Total ORLY results: {result[0]}')

# By scanner
scanner_dist = conn.execute("""
    SELECT scanner_name, COUNT(*) as count
    FROM scanner_results.scanner_results
    WHERE symbol = 'ORLY'
    GROUP BY scanner_name
    ORDER BY count DESC
""").fetchall()
print('\nBy scanner:')
for scanner, count in scanner_dist:
    print(f'  {scanner}: {count}')

# Date range
date_range = conn.execute("""
    SELECT MIN(scan_date), MAX(scan_date), COUNT(DISTINCT scan_date)
    FROM scanner_results.scanner_results
    WHERE symbol = 'ORLY'
""").fetchone()
print(f'\nDate range: {date_range[0]} to {date_range[1]} ({date_range[2]} unique dates)')

# Recent detections
recent = conn.execute("""
    SELECT scan_date, scanner_name, scan_timestamp
    FROM scanner_results.scanner_results
    WHERE symbol = 'ORLY'
    ORDER BY scan_date DESC
    LIMIT 20
""").fetchall()
print('\nRecent 20 detections:')
for date, scanner, timestamp in recent:
    print(f'  {date} - {scanner} ({timestamp})')
