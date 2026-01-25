import duckdb

conn = duckdb.connect('md:scanner_data')

# Check if GLXY exists in fundamental_cache
result = conn.execute("""
    SELECT symbol, sector, industry, market_cap, beta, last_updated
    FROM main.fundamental_cache
    WHERE symbol = 'GLXY'
""").fetchone()

if result:
    print(f'GLXY in fundamental_cache:')
    print(f'  Symbol: {result[0]}')
    print(f'  Sector: {result[1]}')
    print(f'  Industry: {result[2]}')
    print(f'  Market Cap: {result[3]}')
    print(f'  Beta: {result[4]}')
    print(f'  Last Updated: {result[5]}')
else:
    print('GLXY NOT FOUND in fundamental_cache')

# Check rotation_metrics
if result and result[2]:  # if industry exists
    result2 = conn.execute("""
        SELECT industry, rs_ratio, rs_momentum, last_updated
        FROM main.rotation_metrics
        WHERE industry = ?
    """, [result[2]]).fetchone()
    
    if result2:
        print(f'\nIndustry rotation_metrics:')
        print(f'  Industry: {result2[0]}')
        print(f'  RS Ratio: {result2[1]}')
        print(f'  RS Momentum: {result2[2]}')
    else:
        print(f'\nNo rotation_metrics found for industry: {result[2]}')

conn.close()
