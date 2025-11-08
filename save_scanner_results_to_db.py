"""
Save scanner results to DuckDB database for fast web app access.
Run this periodically (daily/hourly) to update scanner results.
"""

import os
import duckdb
import pandas as pd
from datetime import datetime
from qullamaggie_scanner import detect_qullamaggie_breakout
from momentum_burst_scanner import (detect_momentum_burst_1d,
                                     detect_momentum_burst_3d,
                                     detect_momentum_burst_5d)
from supertrend_scanner import (detect_supertrend_bullish,
                                 detect_supertrend_fresh,
                                 detect_supertrend_recent)
from explosive_volume_scanner import (detect_explosive_volume_3x,
                                       detect_explosive_volume_5x,
                                       detect_explosive_volume_10x,
                                       detect_volume_surge_with_price)
from pattern_scoring import calculate_pattern_strength, get_signal_quality

# Database path
DB_PATH = os.environ.get('DUCKDB_PATH', '/Users/george/scannerPOC/breakoutScannersPOCs/scanner_data.duckdb')

# Scanner configurations
SCANNERS = {
    'QULLAMAGGIE_BREAKOUT': detect_qullamaggie_breakout,
    'MOMENTUM_BURST_1D': detect_momentum_burst_1d,
    'MOMENTUM_BURST_3D': detect_momentum_burst_3d,
    'MOMENTUM_BURST_5D': detect_momentum_burst_5d,
    'SUPERTREND_BULLISH': detect_supertrend_bullish,
    'SUPERTREND_FRESH': detect_supertrend_fresh,
    'SUPERTREND_RECENT': detect_supertrend_recent,
    'EXPLOSIVE_VOLUME_3X': detect_explosive_volume_3x,
    'EXPLOSIVE_VOLUME_5X': detect_explosive_volume_5x,
    'EXPLOSIVE_VOLUME_10X': detect_explosive_volume_10x,
    'VOLUME_SURGE_WITH_PRICE': detect_volume_surge_with_price
}


def create_scanner_results_table(conn):
    """Create scanner_results table if it doesn't exist."""
    
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
    print("✅ scanner_results table created/verified")


def run_scanners_and_save():
    """Run all scanners and save results to database."""
    
    print(f"Connecting to database: {DB_PATH}")
    conn = duckdb.connect(DB_PATH, read_only=False)
    
    # Create table
    create_scanner_results_table(conn)
    
    # Get all symbols
    symbols_query = """
        SELECT DISTINCT symbol 
        FROM scanner_data.daily_cache
        ORDER BY symbol
    """
    symbols = [row[0] for row in conn.execute(symbols_query).fetchall()]
    
    print(f"\n📊 Scanning {len(symbols)} symbols with {len(SCANNERS)} scanners...")
    
    results = []
    today = datetime.now().date()
    
    for idx, symbol in enumerate(symbols, 1):
        try:
            # Get data from database
            query = """
                SELECT date, open, high, low, close, volume,
                       rsi_14, sma_20, sma_50, sma_200, atr_14, rvol
                FROM scanner_data.daily_cache
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT 252
            """
            df = conn.execute(query, [symbol]).df()
            
            if df.empty:
                continue
            
            # Sort oldest to newest
            df = df.sort_values('date')
            
            # Skip outdated data
            latest_date = df['date'].iloc[-1]
            days_old = (pd.Timestamp.now() - latest_date).days
            if days_old > 30:
                continue
            
            # Rename columns for scanners
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            # Run each scanner
            for scanner_name, scanner_func in SCANNERS.items():
                try:
                    result = scanner_func(df)
                    
                    if result and result != 'None':
                        # Calculate strength
                        strength = calculate_pattern_strength(df, result, scanner_name)
                        quality = get_signal_quality(strength)
                        
                        results.append({
                            'symbol': symbol,
                            'scanner_name': scanner_name,
                            'signal': result,
                            'strength': strength,
                            'quality': quality,
                            'scan_date': today
                        })
                        
                except Exception as e:
                    print(f"  Error in {scanner_name} for {symbol}: {e}")
                    continue
            
            if idx % 50 == 0:
                print(f"  Processed {idx}/{len(symbols)} symbols... ({len(results)} signals found)")
                
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
            continue
    
    print(f"\n✅ Scan complete! Found {len(results)} signals")
    
    # Save results to database
    if results:
        print("\n💾 Saving results to database...")
        
        # Delete today's results first
        conn.execute("""
            DELETE FROM scanner_data.scanner_results
            WHERE scan_date = ?
        """, [today])
        
        # Insert new results
        df_results = pd.DataFrame(results)
        conn.execute("""
            INSERT INTO scanner_data.scanner_results
            SELECT * FROM df_results
        """)
        
        print(f"✅ Saved {len(results)} scanner results to database")
        
        # Show summary
        print("\n📈 Summary by scanner:")
        for scanner_name in SCANNERS.keys():
            count = sum(1 for r in results if r['scanner_name'] == scanner_name)
            if count > 0:
                print(f"  {scanner_name}: {count} signals")
    
    conn.close()
    print("\n✨ Done!")


if __name__ == '__main__':
    run_scanners_and_save()
