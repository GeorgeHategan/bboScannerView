import os, csv
import talib
import pandas
import duckdb
import yfinance as yf
from flask import Flask, request, render_template
from markupsafe import escape
from patterns import candlestick_patterns
from custom_patterns import (custom_chart_patterns, detect_cup_and_handle, 
                              detect_ascending_triangle, detect_double_bottom,
                              detect_bull_flag, detect_bear_flag)
from qullamaggie_scanner import (detect_qullamaggie_breakout, 
                                  qullamaggie_pattern)
from momentum_burst_scanner import (detect_momentum_burst_1d,
                                     detect_momentum_burst_3d,
                                     detect_momentum_burst_5d,
                                     momentum_burst_patterns)
from supertrend_scanner import (detect_supertrend_bullish,
                                 detect_supertrend_fresh,
                                 detect_supertrend_recent,
                                 supertrend_patterns)
from explosive_volume_scanner import (detect_explosive_volume_3x,
                                       detect_explosive_volume_5x,
                                       detect_explosive_volume_10x,
                                       detect_volume_surge_with_price,
                                       explosive_volume_patterns)
from pattern_scoring import calculate_pattern_strength, get_signal_quality
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta, date

app = Flask(__name__)

# Database configuration
# For local development, use MotherDuck to access production data
# For production (Render), use environment variable
motherduck_token = os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN', '')
if motherduck_token:
    DUCKDB_PATH = os.environ.get('DUCKDB_PATH', f'md:scanner_data?motherduck_token={motherduck_token}')
    print("INFO: Connecting to MotherDuck production database")
else:
    # Fallback to local DB if no MotherDuck token
    DUCKDB_PATH = os.environ.get('DUCKDB_PATH', '/Users/george/scannerPOC/breakoutScannersPOCs/scanner_data.duckdb')
    print("WARNING: No motherduck_token found, using local database (may not have scanner_results)")

# Set your Alpha Vantage API key here or use environment variable
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '75IGYUZ3C7AC2PBM')

# Skip slow database initialization on startup - connect lazily when needed
print(f"INFO: Database path configured: {DUCKDB_PATH}")


def format_market_cap(market_cap):
    """Format market cap for display (e.g., 3.99T, 415.6B, 500.2M)."""
    if market_cap is None:
        return None
    
    try:
        # Convert to float if it's a string
        if isinstance(market_cap, str):
            market_cap = float(market_cap)
        
        if market_cap >= 1_000_000_000_000:
            return f"{market_cap / 1_000_000_000_000:.2f}T"
        elif market_cap >= 1_000_000_000:
            return f"{market_cap / 1_000_000_000:.2f}B"
        elif market_cap >= 1_000_000:
            return f"{market_cap / 1_000_000:.2f}M"
        else:
            return f"{market_cap:,.0f}"
    except:
        return None


def get_news_sentiment(symbol):
    """Get news sentiment for a symbol from Alpha Vantage."""
    try:
        import requests
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_API_KEY}&limit=10'
        
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if 'feed' in data and len(data['feed']) > 0:
            # Calculate average sentiment from recent articles
            sentiment_scores = []
            sentiment_labels = []
            
            for article in data['feed'][:10]:  # Look at top 10 articles
                if 'ticker_sentiment' in article:
                    for ticker_sent in article['ticker_sentiment']:
                        if ticker_sent['ticker'] == symbol:
                            score = float(ticker_sent.get('ticker_sentiment_score', 0))
                            label = ticker_sent.get('ticker_sentiment_label', 'Neutral')
                            sentiment_scores.append(score)
                            sentiment_labels.append(label)
            
            if sentiment_scores:
                avg_score = sum(sentiment_scores) / len(sentiment_scores)
                
                # Determine overall sentiment
                if avg_score >= 0.35:
                    overall = 'Bullish'
                elif avg_score >= 0.15:
                    overall = 'Somewhat-Bullish'
                elif avg_score <= -0.35:
                    overall = 'Bearish'
                elif avg_score <= -0.15:
                    overall = 'Somewhat-Bearish'
                else:
                    overall = 'Neutral'
                
                return {
                    'score': round(avg_score, 3),
                    'label': overall,
                    'article_count': len(sentiment_scores),
                    'total_articles': len(data['feed'])
                }
    except Exception as e:
        print(f"Error getting sentiment for {symbol}: {e}")
    
    return None


def get_earnings_date(symbol):
    """Get next earnings date for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        calendar = ticker.calendar
        
        if calendar is not None and 'Earnings Date' in calendar:
            earnings_dates = calendar['Earnings Date']
            
            # Handle both single date and list of dates
            if not isinstance(earnings_dates, list):
                earnings_dates = [earnings_dates]
            
            # Get the first future date
            next_date = None
            for ed in earnings_dates:
                if ed:
                    next_date = ed
                    break
            
            if next_date:
                # Convert to date for comparison
                if hasattr(next_date, 'date'):
                    next_date = next_date.date()
                
                today = date.today()
                days_until = (next_date - today).days
                
                return {
                    'date': next_date.strftime('%Y-%m-%d'),
                    'days_until': days_until
                }
    except Exception as e:
        print(f"Error getting earnings for {symbol}: {e}")
    
    return None


@app.route('/snapshot')
def snapshot():
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    
    with open('datasets/symbols.csv') as f:
        for line in f:
            if "," not in line:
                continue
            symbol = line.split(",")[0]
            try:
                # Get daily data from Alpha Vantage (compact = last 100 days)
                # Use 'full' for complete history, 'compact' for recent data only
                data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')
                # Rename columns to match previous format (lowercase)
                data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                # Keep only last 252 trading days (~1 year)
                data = data.head(252)
                data.to_csv('datasets/daily/{}.csv'.format(symbol))
                print(f'Downloaded {symbol}')
            except Exception as e:
                print(f'Failed on {symbol}: {e}')

    return {
        "code": "success"
    }

@app.route('/stats')
def stats():
    """Display database statistics landing page."""
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    
    stats_data = {}
    
    try:
        # Total number of scanner results
        total_results = conn.execute("""
            SELECT COUNT(*) FROM scanner_data.scanner_results
        """).fetchone()[0]
        stats_data['total_results'] = total_results
        
        # Number of unique assets scanned
        unique_assets = conn.execute("""
            SELECT COUNT(DISTINCT symbol) FROM scanner_data.scanner_results
        """).fetchone()[0]
        stats_data['unique_assets'] = unique_assets
        
        # Number of scanners
        num_scanners = conn.execute("""
            SELECT COUNT(DISTINCT scanner_name) FROM scanner_data.scanner_results
        """).fetchone()[0]
        stats_data['num_scanners'] = num_scanners
        
        # Last updated date
        last_updated = conn.execute("""
            SELECT MAX(scan_date) FROM scanner_data.scanner_results
        """).fetchone()[0]
        stats_data['last_updated'] = str(last_updated)[:10] if last_updated else 'N/A'
        
        # Results per scanner
        scanner_breakdown = conn.execute("""
            SELECT scanner_name, COUNT(*) as count
            FROM scanner_data.scanner_results
            GROUP BY scanner_name
            ORDER BY count DESC
        """).fetchall()
        stats_data['scanner_breakdown'] = [(row[0], row[1]) for row in scanner_breakdown]
        
        # Results per date
        date_breakdown = conn.execute("""
            SELECT DATE(scan_date) as date, COUNT(*) as count
            FROM scanner_data.scanner_results
            WHERE scan_date IS NOT NULL
            GROUP BY DATE(scan_date)
            ORDER BY date DESC
            LIMIT 10
        """).fetchall()
        stats_data['date_breakdown'] = [(str(row[0]), row[1]) for row in date_breakdown]
        
        # Top picked assets (by multiple scanners)
        top_picks = conn.execute("""
            SELECT symbol, COUNT(DISTINCT scanner_name) as scanner_count
            FROM scanner_data.scanner_results
            GROUP BY symbol
            HAVING COUNT(DISTINCT scanner_name) > 1
            ORDER BY scanner_count DESC
            LIMIT 20
        """).fetchall()
        stats_data['top_picks'] = [(row[0], row[1]) for row in top_picks]
        
        # Signal strength distribution
        strength_dist = conn.execute("""
            SELECT 
                CASE 
                    WHEN signal_strength >= 90 THEN '90-100'
                    WHEN signal_strength >= 80 THEN '80-89'
                    WHEN signal_strength >= 70 THEN '70-79'
                    WHEN signal_strength >= 60 THEN '60-69'
                    ELSE '<60'
                END as strength_range,
                COUNT(*) as count
            FROM scanner_data.scanner_results
            WHERE signal_strength IS NOT NULL
            GROUP BY strength_range
            ORDER BY strength_range DESC
        """).fetchall()
        stats_data['strength_distribution'] = [(row[0], row[1]) for row in strength_dist]
        
    except Exception as e:
        print(f"Error getting stats: {e}")
        stats_data['error'] = str(e)
    
    conn.close()
    
    return render_template('stats.html', stats=stats_data)


@app.route('/scanner-docs')
def scanner_docs():
    """Display documentation landing page with all scanners."""
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    
    # Get scanner info
    scanner_data = conn.execute("""
        SELECT scanner_name, COUNT(*) as count
        FROM scanner_data.scanner_results
        GROUP BY scanner_name
        ORDER BY scanner_name
    """).fetchall()
    
    conn.close()
    
    # Scanner descriptions
    scanner_descriptions = {
        'accumulation_distribution': 'Detects institutional smart money buying patterns using volume indicators',
        'breakout': 'Identifies stocks breaking out above key resistance levels',
        'bull_flag': 'Finds bullish continuation patterns with consolidation after uptrend',
        'momentum_burst': 'Spots explosive momentum moves with high volume',
        'tight_consolidation': 'Detects tight consolidation patterns before potential breakouts'
    }
    
    scanners = []
    for name, count in scanner_data:
        scanners.append({
            'name': name,
            'display_name': name.replace('_', ' ').title(),
            'short_desc': scanner_descriptions.get(name, 'Technical pattern scanner'),
            'count': count
        })
    
    return render_template('scanner_docs.html', scanners=scanners)


@app.route('/scanner-docs/<scanner_name>')
def scanner_detail(scanner_name):
    """Display detailed documentation for a specific scanner."""
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    
    # Get scanner stats
    stats = conn.execute("""
        SELECT 
            COUNT(*) as total,
            AVG(signal_strength) as avg_strength,
            COUNT(DISTINCT symbol) as unique_symbols
        FROM scanner_data.scanner_results
        WHERE scanner_name = ?
    """, [scanner_name]).fetchone()
    
    conn.close()
    
    scanner_info = {
        'name': scanner_name,
        'display_name': scanner_name.replace('_', ' ').title(),
        'total_setups': stats[0] if stats else 0,
        'avg_strength': f"{stats[1]:.1f}" if stats and stats[1] else "N/A",
        'unique_symbols': stats[2] if stats else 0
    }
    
    # Load scanner-specific content
    content = get_scanner_documentation(scanner_name)
    
    return render_template('scanner_detail.html', scanner_info=scanner_info, content=content)


@app.route('/ticker-search')
def ticker_search():
    """Search for all scanner results for a specific ticker."""
    ticker = request.args.get('ticker', '').strip().upper()
    
    if not ticker:
        return render_template('ticker_search.html', ticker=None, results=None)
    
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    
    try:
        # Get all historical and current results for this ticker
        results = conn.execute("""
            SELECT 
                scanner_name,
                symbol,
                scan_date,
                entry_price,
                signal_strength,
                notes
            FROM scanner_data.scanner_results
            WHERE symbol = ?
            ORDER BY scan_date DESC, scanner_name
        """, [ticker]).fetchall()
        
        # Get current price if available
        current_price = None
        try:
            current_data = conn.execute("""
                SELECT close, date
                FROM scanner_data.daily_cache
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT 1
            """, [ticker]).fetchone()
            if current_data:
                current_price = current_data[0]
        except Exception as e:
            print(f"Could not fetch current price: {e}")
        
        # Format results
        formatted_results = []
        for row in results:
            gain_pct = None
            if current_price and row[3]:  # entry_price exists
                gain_pct = ((current_price - row[3]) / row[3]) * 100
            
            formatted_results.append({
                'scanner_name': row[0].replace('_', ' ').title(),
                'symbol': row[1],
                'scan_date': row[2],
                'entry_price': f"${row[3]:.2f}" if row[3] else "N/A",
                'current_price': f"${current_price:.2f}" if current_price else "N/A",
                'gain_pct': f"{gain_pct:+.1f}%" if gain_pct is not None else "N/A",
                'signal_strength': f"{row[4]:.1f}" if row[4] else "N/A",
                'notes': row[5] if row[5] else ""
            })
        
        return render_template('ticker_search.html', 
                             ticker=ticker, 
                             results=formatted_results,
                             result_count=len(formatted_results))
    
    except Exception as e:
        print(f"Error searching ticker: {e}")
        return render_template('ticker_search.html', 
                             ticker=ticker, 
                             results=None,
                             error=str(e))
    finally:
        conn.close()


def get_scanner_documentation(scanner_name):
    """Return HTML documentation for specific scanner."""
    
    docs = {
        'accumulation_distribution': '''
<div class="stats-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 8px; margin: 30px 0;">
    <h3 style="color: white; margin-top: 0;">Current Performance Metrics</h3>
    <div class="stats-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">
        <div style="text-align: center;">
            <span style="font-size: 2.5em; font-weight: bold; display: block;">310</span>
            <span style="font-size: 0.9em; opacity: 0.9; margin-top: 5px;">Active Signals</span>
        </div>
        <div style="text-align: center;">
            <span style="font-size: 2.5em; font-weight: bold; display: block;">79.4</span>
            <span style="font-size: 0.9em; opacity: 0.9; margin-top: 5px;">Avg Quality Score</span>
        </div>
        <div style="text-align: center;">
            <span style="font-size: 2.5em; font-weight: bold; display: block;">39%</span>
            <span style="font-size: 0.9em; opacity: 0.9; margin-top: 5px;">Success Rate (10-day)</span>
        </div>
        <div style="text-align: center;">
            <span style="font-size: 2.5em; font-weight: bold; display: block;">51.4%</span>
            <span style="font-size: 0.9em; opacity: 0.9; margin-top: 5px;">Success Rate (20-day)</span>
        </div>
    </div>
</div>

<h2>🎯 What It Does</h2>
<p>The Accumulation/Distribution scanner detects <strong>institutional smart money buying patterns</strong> by analyzing volume-based indicators that reveal hidden accumulation before major price moves.</p>

<div class="alert alert-info" style="background: #d1ecf1; border-left: 5px solid #17a2b8; color: #0c5460; padding: 20px; border-radius: 6px; margin: 20px 0;">
    <strong>💡 Key Concept:</strong> "Accumulation" means large institutions (hedge funds, mutual funds) are quietly buying shares while the price consolidates. This typically happens <strong>before</strong> major breakouts, giving you an early entry advantage.
</div>

<h2>📈 Core Indicators</h2>

<div style="background: #ecf0f1; padding: 20px; border-radius: 6px; margin: 15px 0; border-left: 4px solid #3498db;">
    <div style="font-weight: bold; color: #2980b9; font-size: 1.1em; margin-bottom: 10px;">1. A/D Line (Accumulation/Distribution Line)</div>
    <p>Tracks money flow by comparing closing prices to daily ranges. Rising A/D Line = buying pressure, falling = selling pressure.</p>
    <p><strong>Formula:</strong> ((Close - Low) - (High - Close)) / (High - Low) × Volume (cumulative)</p>
</div>

<div style="background: #ecf0f1; padding: 20px; border-radius: 6px; margin: 15px 0; border-left: 4px solid #3498db;">
    <div style="font-weight: bold; color: #2980b9; font-size: 1.1em; margin-bottom: 10px;">2. OBV (On-Balance Volume)</div>
    <p>Volume-weighted momentum indicator. Adds volume on up days, subtracts on down days.</p>
    <p><strong>Logic:</strong> Rising OBV confirms uptrend strength; divergence signals potential reversals.</p>
</div>

<div style="background: #ecf0f1; padding: 20px; border-radius: 6px; margin: 15px 0; border-left: 4px solid #3498db;">
    <div style="font-weight: bold; color: #2980b9; font-size: 1.1em; margin-bottom: 10px;">3. CMF (Chaikin Money Flow)</div>
    <p>20-period oscillator measuring money flow pressure.</p>
    <p><strong>Optimal Range:</strong> -0.05 to +0.15 (slightly positive performs best)</p>
</div>

<div style="background: #ecf0f1; padding: 20px; border-radius: 6px; margin: 15px 0; border-left: 4px solid #3498db;">
    <div style="font-weight: bold; color: #2980b9; font-size: 1.1em; margin-bottom: 10px;">4. Volume Profile Analysis</div>
    <p>Compares volume on up days vs down days over 20 periods.</p>
    <p><strong>Ideal Ratio:</strong> 0.8 to 1.5x (neutral to moderate buying)</p>
</div>

<div style="background: #ecf0f1; padding: 20px; border-radius: 6px; margin: 15px 0; border-left: 4px solid #3498db;">
    <div style="font-weight: bold; color: #2980b9; font-size: 1.1em; margin-bottom: 10px;">5. Bullish Divergence Detection</div>
    <p>Identifies when indicators rise while price falls - classic accumulation signal.</p>
    <p><strong>Impact:</strong> +10% improvement in success rate (24.6% vs 22.4%)</p>
</div>

<h2>📊 Quality Score Breakdown (310 Current Signals)</h2>

<table>
    <thead>
        <tr>
            <th>Quality Range</th>
            <th>Count</th>
            <th>Percentage</th>
            <th>Rating</th>
            <th>Interpretation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>95-100</td>
            <td>6</td>
            <td>2%</td>
            <td><span style="background: #2ecc71; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">Perfect</span></td>
            <td>All indicators perfectly aligned - highest conviction</td>
        </tr>
        <tr>
            <td>90-98</td>
            <td>18</td>
            <td>6%</td>
            <td><span style="background: #2ecc71; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">Excellent</span></td>
            <td>Strong accumulation signals across all metrics</td>
        </tr>
        <tr>
            <td>85-88</td>
            <td>31</td>
            <td>10%</td>
            <td><span style="background: #3498db; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">Very Good</span></td>
            <td>Clear buying pressure with minor weaknesses</td>
        </tr>
        <tr>
            <td>80-83</td>
            <td>57</td>
            <td>18%</td>
            <td><span style="background: #3498db; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">Good</span></td>
            <td>Solid setup with good risk/reward</td>
        </tr>
        <tr>
            <td>73-78</td>
            <td>146</td>
            <td>47%</td>
            <td><span style="background: #f39c12; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">Fair</span></td>
            <td>Marginal quality - requires additional confirmation</td>
        </tr>
        <tr>
            <td>70-72</td>
            <td>29</td>
            <td>9%</td>
            <td><span style="background: #95a5a6; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">Minimum</span></td>
            <td>Barely qualifies - high risk</td>
        </tr>
    </tbody>
</table>

<h2>🔍 Real-World Example: TER (Teradyne Inc)</h2>

<div style="background: #fff9e6; border: 2px solid #f39c12; padding: 20px; border-radius: 8px; margin: 20px 0;">
    <div style="font-weight: bold; color: #d68910; font-size: 1.2em; margin-bottom: 10px;">🎯 Monster Winner - +125% Gain</div>
    <ul>
        <li><strong>Entry Signal:</strong> Nov 5, 2025 at $83.08</li>
        <li><strong>Quality Score:</strong> 100/100 (Perfect)</li>
        <li><strong>Current Price:</strong> $187.59</li>
        <li><strong>Gain:</strong> +$104.51 (+125.8%)</li>
        <li><strong>Pattern:</strong> Classic accumulation at $80-90 range followed by explosive breakout</li>
    </ul>
    <p style="margin-top: 15px;"><strong>Why It Worked:</strong> Scanner detected institutional buying in the $80-90 consolidation zone. All indicators aligned perfectly (100 quality score), signaling smart money accumulation before the major move.</p>
</div>

<h2>⚙️ Current Configuration (Testing Mode)</h2>

<div class="alert alert-warning" style="background: #fff3cd; border-left: 5px solid #ffc107; color: #856404; padding: 20px; border-radius: 6px; margin: 20px 0;">
    <strong>⚠️ Important:</strong> Scanner is currently running in <strong>testing mode</strong> with relaxed filters. This explains why it finds 310 signals instead of 50-80.
</div>

<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Current Value</th>
            <th>Production Value</th>
            <th>Impact</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Quality Threshold</td>
            <td><code>70</code></td>
            <td><code>85</code></td>
            <td>Would reduce from 310 → 80 signals</td>
        </tr>
        <tr>
            <td>Min Dollar Volume</td>
            <td><code>$5M/day</code></td>
            <td><code>$50M/day</code></td>
            <td>10x stricter liquidity filter</td>
        </tr>
        <tr>
            <td>Divergence Required</td>
            <td><code>False</code></td>
            <td><code>True</code></td>
            <td>+10% success rate improvement</td>
        </tr>
        <tr>
            <td>OBV Alignment</td>
            <td><code>False</code></td>
            <td><code>True</code></td>
            <td>Confirms trend direction</td>
        </tr>
    </tbody>
</table>

<h2>📊 Sector Performance (Historical)</h2>

<table>
    <thead>
        <tr>
            <th>Sector</th>
            <th>Success Rate</th>
            <th>Rating</th>
        </tr>
    </thead>
    <tbody>
        <tr style="background: #d4edda;">
            <td><span style="background: #2ecc71; color: white; padding: 5px 12px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">TECHNOLOGY</span></td>
            <td>30.6%</td>
            <td>⭐ Best</td>
        </tr>
        <tr style="background: #d4edda;">
            <td><span style="background: #2ecc71; color: white; padding: 5px 12px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">BASIC MATERIALS</span></td>
            <td>26.8%</td>
            <td>⭐ Excellent</td>
        </tr>
        <tr style="background: #d4edda;">
            <td><span style="background: #2ecc71; color: white; padding: 5px 12px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">ENERGY</span></td>
            <td>25.8%</td>
            <td>⭐ Excellent</td>
        </tr>
        <tr style="background: #f8d7da;">
            <td><span style="background: #e74c3c; color: white; padding: 5px 12px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">UTILITIES</span></td>
            <td>8.9%</td>
            <td>❌ Terrible</td>
        </tr>
        <tr style="background: #f8d7da;">
            <td><span style="background: #e74c3c; color: white; padding: 5px 12px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">REAL ESTATE</span></td>
            <td>10.7%</td>
            <td>❌ Terrible</td>
        </tr>
    </tbody>
</table>

<h2>💡 How to Use the Scanner</h2>

<h3>Focus on Quality Tiers:</h3>
<ul>
    <li><strong>Quality 95-100</strong> (6 stocks) - <span style="background: #2ecc71; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">Perfect</span> - Highest conviction plays, all indicators aligned</li>
    <li><strong>Quality 90-94</strong> (18 stocks) - <span style="background: #2ecc71; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">Excellent</span> - Strong setups, primary watchlist</li>
    <li><strong>Quality 85-89</strong> (31 stocks) - <span style="background: #3498db; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">Very Good</span> - Solid opportunities with good risk/reward</li>
    <li><strong>Quality 80-84</strong> (57 stocks) - <span style="background: #3498db; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">Good</span> - Acceptable with proper risk management</li>
    <li><strong>Quality 70-79</strong> (175 stocks) - <span style="background: #f39c12; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">Fair/Minimum</span> - Too risky for most traders</li>
</ul>

<div class="alert alert-success" style="background: #d4edda; border-left: 5px solid #28a745; color: #155724; padding: 20px; border-radius: 6px; margin: 20px 0;">
    <strong>✅ Pro Tip:</strong> The "entry_price" field shows where institutions accumulated (e.g., TER at $83.08), not necessarily where to buy today. Use this to understand the accumulation zone and gauge profit potential from that base.
</div>

<h2>📈 Historical Performance</h2>

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 8px; margin: 30px 0;">
    <h3 style="color: white; margin-top: 0;">Validated on 85,534+ Historical Patterns</h3>
    <ul style="color: white;">
        <li><strong>10-Day Success Rate:</strong> 39% (71.8% better than baseline 22.7%)</li>
        <li><strong>20-Day Success Rate:</strong> 51.4% (consistently profitable)</li>
        <li><strong>Average Gain:</strong> +1.73% (10-day), +3.48% (20-day)</li>
        <li><strong>Quality 80+ Success:</strong> 29.9% vs Quality <50: 9.4% (3.2x difference)</li>
    </ul>
</div>

<h2>📚 Summary</h2>

<p>The Accumulation/Distribution scanner is a <strong>powerful tool for detecting institutional buying</strong> before major moves. While currently finding 310 signals due to testing mode settings, the scanner has proven its ability to identify monster winners like TER (+125%).</p>

<p><strong>For best results:</strong></p>
<ul>
    <li>Focus on quality scores 85+ (top 26% of signals)</li>
    <li>Prioritize TECHNOLOGY, ENERGY, and BASIC MATERIALS sectors</li>
    <li>Wait for production mode settings to reduce signal count to 50-80 highest conviction setups</li>
    <li>Use the entry_price field to understand accumulation zones, not as today's buy signal</li>
    <li>Apply proper risk management - even 51% success rate means 49% losers</li>
</ul>

<div class="alert alert-success" style="background: #d4edda; border-left: 5px solid #28a745; color: #155724; padding: 20px; border-radius: 6px; margin: 30px 0;">
    <strong>🎯 Final Takeaway:</strong> The scanner works excellently for identifying accumulation patterns. The key is filtering for quality (85+) and understanding that it detects <strong>early-stage accumulation</strong>, not breakout confirmation. This gives you an edge by finding stocks before the crowd discovers them.
</div>
''',
        'breakout': '''
<div class="stats-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px;">
    <h2 style="color: white; border: none;">📊 Current Performance</h2>
    <p>Total Signals: <strong>25</strong> | Signal Strength: <strong>N/A</strong></p>
</div>

<h2>📈 Strategy Overview</h2>
<p>The Breakout Scanner implements <strong>Kristjan Qullamaggie's breakout methodology</strong> - one of the most respected short-term trading strategies. It uses both daily and hourly data to catch breakouts above 20-day highs with volume confirmation.</p>

<div style="background: #dbeafe; border-left: 4px solid #3b82f6; padding: 20px; margin: 20px 0; border-radius: 6px;">
    <h3 style="margin-top: 0;">💡 Key Innovation: Hourly Data Advantage</h3>
    <p>This scanner uses <strong>hourly data for precise entry timing</strong>. While most scanners wait for daily close (4:00 PM), hourly bars detect breakouts at 10 AM, 11 AM, etc. - giving you a 5-6 hour head start on entries.</p>
    <p style="margin-top: 10px;"><strong>Example:</strong> Stock breaks out at 10:30 AM with volume. Hourly scanner catches it at 11 AM bar. Daily scanner doesn't see it until 4 PM close - by then, stock may be up 3-5% already.</p>
</div>

<h2>✅ Entry Criteria</h2>
<ul>
    <li><strong>Price Breakout:</strong> Above 20-day high (new short-term high)</li>
    <li><strong>Volume Confirmation:</strong> 2x+ average volume on breakout</li>
    <li><strong>Trend Context:</strong> Above 10-day, 20-day, 50-day SMA (multi-timeframe uptrend)</li>
    <li><strong>Price Filter:</strong> $5-$10 maximum (Qullamaggie focuses on lower-priced stocks for leverage)</li>
    <li><strong>Liquidity:</strong> 100K+ average daily volume</li>
    <li><strong>Timing:</strong> Detected on hourly bars for early entry</li>
</ul>

<h2>⏰ Hourly vs Daily Data</h2>
<p><strong>Why Hourly Data Matters:</strong></p>
<ul>
    <li><strong>Earlier Detection:</strong> Catch breakouts at 10 AM instead of waiting until 4 PM close</li>
    <li><strong>Better Entries:</strong> Enter closer to breakout level (less slippage)</li>
    <li><strong>Reduced Risk:</strong> Tighter stops since entry is earlier in the move</li>
    <li><strong>Less Competition:</strong> Most traders wait for daily close confirmation</li>
</ul>

<p style="margin-top: 20px;"><strong>Daily Data Usage:</strong></p>
<ul>
    <li>Calculate 20-day high level (breakout threshold)</li>
    <li>Verify trend context (10/20/50-day SMA)</li>
    <li>Measure average volume (20-day baseline)</li>
</ul>

<h2>⚠️ Risk Management</h2>
<ul>
    <li><strong>Stop Loss:</strong> Below breakout level or recent swing low (typically 3-7%)</li>
    <li><strong>Position Size:</strong> Risk 1-2% of account per trade</li>
    <li><strong>Profit Target:</strong> 10-20% for first exit, trail remainder</li>
    <li><strong>Time Stop:</strong> Exit if no follow-through within 1-2 days</li>
    <li><strong>Holding Period:</strong> 2-7 days typical (short-term momentum)</li>
</ul>

<h2>🎯 How to Use This Scanner</h2>
<ol>
    <li><strong>Intraday Monitoring:</strong> Run scanner every 1-2 hours during market hours</li>
    <li>Check for new hourly breakouts above 20-day high</li>
    <li>Verify volume is 2x+ average (strong participation)</li>
    <li>Confirm stock is above 10/20/50-day SMA (aligned trend)</li>
    <li>Enter on confirmation bar (next hour after breakout)</li>
    <li>Set stop below breakout level or recent swing low</li>
    <li>Take partial profits at 10-15%, trail remainder</li>
</ol>

<h2>📝 Summary</h2>
<p>The Breakout Scanner implements <strong>Qullamaggie's proven methodology</strong> with a key innovation: <strong>hourly data for early detection</strong>. With 25 signals, this is one of the most selective scanners in the suite.</p>

<p style="margin-top: 20px;"><strong>Key Takeaways:</strong></p>
<ul>
    <li>✅ Proven methodology from successful trader (Kristjan Qullamaggie)</li>
    <li>✅ Hourly data gives 5-6 hour head start vs daily close</li>
    <li>✅ Highly selective - only 25 signals (quality over quantity)</li>
    <li>✅ Clear entry/exit rules (20-day high breakout, 2x volume)</li>
    <li>⚠️ No signal_strength scores in database (needs implementation)</li>
    <li>⚠️ Requires intraday monitoring (not end-of-day scan)</li>
</ul>
''',
        'bull_flag': '''
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px;">
    <h2 style="color: white; border: none; margin-bottom: 10px;">📊 Current Performance</h2>
    <p style="opacity: 0.9;">Recent scan results from MotherDuck database</p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">
        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 6px;">
            <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 5px;">Total Signals</div>
            <div style="font-size: 1.8em; font-weight: bold;">169</div>
        </div>
        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 6px;">
            <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 5px;">Avg Quality</div>
            <div style="font-size: 1.8em; font-weight: bold;">75.9</div>
        </div>
        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 6px;">
            <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 5px;">Quality 80+</div>
            <div style="font-size: 1.8em; font-weight: bold;">37%</div>
        </div>
    </div>
</div>

<h2>📈 Strategy Overview</h2>
<p>The Bull Flag Scanner identifies one of the most reliable continuation patterns in technical analysis - the <strong>bull flag wedge</strong>. This pattern represents a brief consolidation after a strong uptrend, signaling continuation potential for swing trades (2-3 week holding period).</p>

<div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 20px; margin: 20px 0; border-radius: 6px;">
    <h3 style="margin-top: 0;">💡 Key Insight</h3>
    <p>Bull flags work because profit-takers create healthy consolidations that allow new buyers to accumulate. When the pattern breaks out, trapped shorts and FOMO buyers drive explosive moves.</p>
</div>

<h3 style="margin-top: 30px; color: #667eea;">Classic Bull Flag Pattern:</h3>
<ul>
    <li><strong>Flagpole:</strong> Sharp 20-40% rally in 1-3 weeks (momentum phase)</li>
    <li><strong>Flag:</strong> Tight 5-15% pullback/consolidation in 1-2 weeks</li>
    <li><strong>Volume:</strong> Heavy during pole, light during flag (profit-taking)</li>
    <li><strong>Breakout:</strong> Volume surge + move above flag high = continuation</li>
    <li><strong>Target:</strong> Measured move = flagpole height added to breakout level</li>
</ul>

<h2>🔍 5-Phase Pattern Recognition</h2>
<p>The scanner uses sophisticated multi-phase analysis to identify high-quality bull flags:</p>

<div style="background: #f8f9fa; border: 2px solid #667eea; padding: 20px; border-radius: 8px; margin: 15px 0;">
    <h4 style="color: #667eea; margin-bottom: 10px; font-size: 1.1em;">Phase 1: Pre-Pole Confirmation</h4>
    <ul>
        <li>Stock was in uptrend before pole (above 50 SMA)</li>
        <li>No major resistance overhead</li>
        <li>Clean technical setup</li>
        <li><strong>Purpose:</strong> Confirms quality of trend, not late-stage exhaustion</li>
    </ul>
</div>

<div style="background: #f8f9fa; border: 2px solid #667eea; padding: 20px; border-radius: 8px; margin: 15px 0;">
    <h4 style="color: #667eea; margin-bottom: 10px; font-size: 1.1em;">Phase 2: Flagpole Quality</h4>
    <ul>
        <li>Strong upward move: 20-40%+ in short time</li>
        <li>Heavy volume on pole (2x+ average)</li>
        <li>Ideally 7-15 trading days</li>
        <li><strong>Purpose:</strong> Measures strength of underlying momentum</li>
    </ul>
</div>

<div style="background: #f8f9fa; border: 2px solid #667eea; padding: 20px; border-radius: 8px; margin: 15px 0;">
    <h4 style="color: #667eea; margin-bottom: 10px; font-size: 1.1em;">Phase 3: Flag Formation</h4>
    <ul>
        <li>Pullback: 5-15% from pole high</li>
        <li>Duration: 5-15 days ideal</li>
        <li>Declining volume (profit-taking exhausting)</li>
        <li><strong>Purpose:</strong> Healthy consolidation, not reversal</li>
    </ul>
</div>

<div style="background: #f8f9fa; border: 2px solid #667eea; padding: 20px; border-radius: 8px; margin: 15px 0;">
    <h4 style="color: #667eea; margin-bottom: 10px; font-size: 1.1em;">Phase 4: Current Setup</h4>
    <ul>
        <li>Price near top of flag (ready to break)</li>
        <li>Volume starting to pick up</li>
        <li>RSI reset from overbought</li>
        <li><strong>Purpose:</strong> Timing entry at optimal risk/reward</li>
    </ul>
</div>

<div style="background: #f8f9fa; border: 2px solid #667eea; padding: 20px; border-radius: 8px; margin: 15px 0;">
    <h4 style="color: #667eea; margin-bottom: 10px; font-size: 1.1em;">Phase 5: Breakout Confirmation</h4>
    <ul>
        <li>Move above flag high</li>
        <li>Volume surge (1.5x+ average)</li>
        <li>Follow-through in next 1-2 days</li>
        <li><strong>Purpose:</strong> Confirms pattern vs false breakout</li>
    </ul>
</div>

<h2>🎯 Quality Scoring System (0-100)</h2>
<p>Most complex scanner with sophisticated scoring across 4 categories:</p>

<table>
    <thead>
        <tr>
            <th>Category</th>
            <th>Max Points</th>
            <th>What It Measures</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Flagpole Strength</strong></td>
            <td>40</td>
            <td>Magnitude & speed of initial rally (20%+ = max points)</td>
        </tr>
        <tr>
            <td><strong>Flag Quality</strong></td>
            <td>30</td>
            <td>Tight consolidation (5-10% ideal), declining volume</td>
        </tr>
        <tr>
            <td><strong>Technical Setup</strong></td>
            <td>20</td>
            <td>Above key SMAs, clean chart, no overhead resistance</td>
        </tr>
        <tr>
            <td><strong>Entry Timing</strong></td>
            <td>10</td>
            <td>Near breakout point, volume picking up, RSI reset</td>
        </tr>
    </tbody>
</table>

<h2>✅ Entry Criteria</h2>
<p>All patterns must meet these requirements:</p>
<ul>
    <li><strong>Minimum Flagpole:</strong> 15%+ rally to qualify as strong momentum</li>
    <li><strong>Flag Duration:</strong> 5-15 days (not too quick, not too long)</li>
    <li><strong>Pullback Depth:</strong> 5-15% from pole high (healthy correction)</li>
    <li><strong>Volume Pattern:</strong> Heavy on pole, light during flag</li>
    <li><strong>Technical Position:</strong> Above 20-day SMA minimum</li>
    <li><strong>Quality Threshold:</strong> 70+ score to pass filter</li>
</ul>

<h2>📊 Current Results (169 Signals)</h2>

<table>
    <thead>
        <tr>
            <th>Quality Range</th>
            <th>Count</th>
            <th>Percentage</th>
            <th>Rating</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>85-100</td>
            <td>16</td>
            <td>9%</td>
            <td><span style="background: #f59e0b; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">Good</span></td>
        </tr>
        <tr>
            <td>80-84</td>
            <td>46</td>
            <td>27%</td>
            <td><span style="background: #f59e0b; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">Good</span></td>
        </tr>
        <tr>
            <td>75-79</td>
            <td>56</td>
            <td>33%</td>
            <td><span style="background: #6b7280; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">Fair</span></td>
        </tr>
        <tr>
            <td>70-74</td>
            <td>51</td>
            <td>30%</td>
            <td><span style="background: #6b7280; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">Fair</span></td>
        </tr>
    </tbody>
</table>

<p style="margin-top: 20px;"><strong>Observations:</strong></p>
<ul>
    <li>62 signals (37%) rated 80+ = focus tier for best setups</li>
    <li>107 signals (63%) rated 70-79 = marginal quality, requires confirmation</li>
    <li>Average 75.9 suggests most flags are "okay" but not exceptional</li>
    <li>Top 16 signals (85+) = highest conviction trades</li>
</ul>

<h2>⚠️ Risk Management</h2>
<ul>
    <li><strong>Stop Loss:</strong> Below flag low or recent swing low (typically 5-10%)</li>
    <li><strong>Position Size:</strong> Risk 1-2% of account</li>
    <li><strong>Profit Target:</strong> Measured move (flagpole height added to breakout)</li>
    <li><strong>Time Stop:</strong> Exit if no breakout within 1 week of entry</li>
    <li><strong>Holding Period:</strong> 2-3 weeks typical for target hit</li>
</ul>

<h2>🎯 How to Use This Scanner</h2>
<ol>
    <li><strong>Filter by Quality:</strong> Focus on 80+ signals first (62 stocks)</li>
    <li><strong>Visual Confirmation:</strong> Check chart for clean flag pattern</li>
    <li><strong>Volume Check:</strong> Verify volume declining during flag formation</li>
    <li><strong>Entry Timing:</strong>
        <ul>
            <li><strong>Aggressive:</strong> Buy near flag low with stop below</li>
            <li><strong>Conservative:</strong> Wait for breakout above flag high</li>
            <li><strong>Confirmation:</strong> Enter on first pullback after breakout</li>
        </ul>
    </li>
    <li><strong>Set Alerts:</strong> Price alerts at flag high for breakout notification</li>
    <li><strong>Target Setting:</strong> Measure flagpole height, add to breakout level</li>
    <li><strong>Scale Out:</strong> Take 1/3 at target, trail rest with 3-day low stop</li>
</ol>

<h2>📝 Summary</h2>
<p>The Bull Flag Scanner identifies <strong>high-probability continuation patterns</strong> for swing trades. With 169 signals and average 75.9 quality, focus on the top 62 signals (80+) for best results.</p>

<p style="margin-top: 20px;"><strong>Key Takeaways:</strong></p>
<ul>
    <li>✅ Most sophisticated scanner - 5-phase analysis</li>
    <li>✅ 169 signals = good selection of opportunities</li>
    <li>✅ 37% rated 80+ = focus on top-tier setups</li>
    <li>✅ Measured move target = clear exit strategy</li>
    <li>⚠️ Requires visual confirmation - scanner finds candidates, you verify pattern</li>
    <li>⚠️ 63% signals are "fair" quality (70-79) - needs additional filters</li>
</ul>
''',
        'momentum_burst': '''
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px;">
    <h2 style="color: white; border: none; margin-bottom: 10px;">📊 Current Performance</h2>
    <p style="opacity: 0.9;">Recent scan results from MotherDuck database</p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">
        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 6px;">
            <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 5px;">Total Signals</div>
            <div style="font-size: 1.8em; font-weight: bold;">36</div>
        </div>
        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 6px;">
            <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 5px;">Avg Quality</div>
            <div style="font-size: 1.8em; font-weight: bold;">80.9</div>
        </div>
        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 6px;">
            <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 5px;">Quality 85+</div>
            <div style="font-size: 1.8em; font-weight: bold;">47%</div>
        </div>
        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 6px;">
            <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 5px;">Quality 90+</div>
            <div style="font-size: 1.8em; font-weight: bold;">22%</div>
        </div>
    </div>
</div>

<h2>📈 Strategy Overview</h2>
<p>The Momentum Burst Scanner identifies <strong>explosive short-term momentum moves</strong> based on Stockbee's methodology. It looks for stocks that have made significant price gains (4-8%+) in 1-5 days with strong volume confirmation.</p>

<div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 20px; margin: 20px 0; border-radius: 6px;">
    <h3 style="margin-top: 0;">⚠️ High Risk / High Reward</h3>
    <p>Momentum bursts are the <strong>most dangerous signals to trade</strong>. 60-70% fade within 3-5 days. These require experience, discipline, and quick decision-making. Not recommended for beginners.</p>
</div>

<h3 style="margin-top: 30px; color: #667eea;">Three Signal Types:</h3>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
    <div style="background: #f8f9fa; border-left: 4px solid #667eea; padding: 20px; border-radius: 6px;">
        <h3 style="color: #667eea; margin-bottom: 10px; font-size: 1.3em;">1-Day Burst</h3>
        <p style="color: #555; margin-bottom: 10px;">Single explosive day (5-10% gain)</p>
        <div style="background: #fff; padding: 10px; border-radius: 4px; margin-top: 10px; font-family: 'Courier New', monospace; font-size: 0.9em; border: 1px solid #ddd;">
            <strong>Criteria:</strong> 5%+ gain, 3x+ volume<br>
            <strong>Strategy:</strong> Quick scalp or wait for pullback<br>
            <strong>Risk:</strong> Highest - often reverses next day
        </div>
    </div>
    
    <div style="background: #f8f9fa; border-left: 4px solid #667eea; padding: 20px; border-radius: 6px;">
        <h3 style="color: #667eea; margin-bottom: 10px; font-size: 1.3em;">3-Day Burst</h3>
        <p style="color: #555; margin-bottom: 10px;">Sustained momentum (3 consecutive up days)</p>
        <div style="background: #fff; padding: 10px; border-radius: 4px; margin-top: 10px; font-family: 'Courier New', monospace; font-size: 0.9em; border: 1px solid #ddd;">
            <strong>Criteria:</strong> 8%+ gain over 3 days<br>
            <strong>Strategy:</strong> Swing trade for continuation<br>
            <strong>Risk:</strong> Moderate - more reliable follow-through
        </div>
    </div>
    
    <div style="background: #f8f9fa; border-left: 4px solid #667eea; padding: 20px; border-radius: 6px;">
        <h3 style="color: #667eea; margin-bottom: 10px; font-size: 1.3em;">5-Day Burst</h3>
        <p style="color: #555; margin-bottom: 10px;">Week-long momentum move</p>
        <div style="background: #fff; padding: 10px; border-radius: 4px; margin-top: 10px; font-family: 'Courier New', monospace; font-size: 0.9em; border: 1px solid #ddd;">
            <strong>Criteria:</strong> 12%+ gain over 5 days<br>
            <strong>Strategy:</strong> Position trade (hold weeks)<br>
            <strong>Risk:</strong> Lower - major fundamental change likely
        </div>
    </div>
</div>

<h2>✅ Entry Criteria</h2>
<p>All signals must meet these requirements:</p>
<ul>
    <li><strong>Price Move:</strong> 4-12%+ gain in 1-5 days (depending on timeframe)</li>
    <li><strong>Volume Surge:</strong> 2-5x+ average volume</li>
    <li><strong>RSI Momentum:</strong> RSI > 60 (strong momentum)</li>
    <li><strong>Up Days:</strong> Majority green candles (buying pressure)</li>
    <li><strong>Price Position:</strong> Ideally above 50 SMA (uptrend context)</li>
</ul>

<h2>🎯 Quality Scoring (0-100)</h2>
<p>Momentum bursts scored on magnitude, volume, and sustainability:</p>

<table>
    <thead>
        <tr>
            <th>Factor</th>
            <th>Weight</th>
            <th>How It's Measured</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Price Gain %</strong></td>
            <td>40 pts</td>
            <td>5% = 20pts, 10% = 30pts, 15%+ = 40pts</td>
        </tr>
        <tr>
            <td><strong>Volume Multiple</strong></td>
            <td>30 pts</td>
            <td>2x = 15pts, 5x = 25pts, 10x+ = 30pts</td>
        </tr>
        <tr>
            <td><strong>Consistency</strong></td>
            <td>20 pts</td>
            <td>% of up days (5/5 days = 20pts)</td>
        </tr>
        <tr>
            <td><strong>RSI Strength</strong></td>
            <td>10 pts</td>
            <td>RSI 70+ = strong momentum confirmation</td>
        </tr>
    </tbody>
</table>

<h2>📊 Current Results (36 Signals)</h2>

<table>
    <thead>
        <tr>
            <th>Quality Range</th>
            <th>Count</th>
            <th>Percentage</th>
            <th>Rating</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>90-100</td>
            <td>8</td>
            <td>22%</td>
            <td><span style="background: #10b981; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">Excellent</span></td>
        </tr>
        <tr>
            <td>85-89</td>
            <td>9</td>
            <td>25%</td>
            <td><span style="background: #3b82f6; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">Very Good</span></td>
        </tr>
        <tr>
            <td>80-84</td>
            <td>6</td>
            <td>17%</td>
            <td><span style="background: #f59e0b; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">Good</span></td>
        </tr>
        <tr>
            <td>70-79</td>
            <td>13</td>
            <td>36%</td>
            <td><span style="background: #6b7280; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">Fair</span></td>
        </tr>
    </tbody>
</table>

<p style="margin-top: 20px;"><strong>Observations:</strong></p>
<ul>
    <li>17 signals (47%) rated 85+ = elite momentum moves</li>
    <li>8 signals (22%) at 90+ = strongest bursts (10%+ gains, huge volume)</li>
    <li>Average 80.9 is high - scanner is very selective</li>
    <li>Only 36 signals = quality over quantity (vs 169 bull flags)</li>
</ul>

<h2>⚠️ Risk Management</h2>
<ul>
    <li><strong>Stop Loss:</strong> 5-7% maximum (tight stops essential)</li>
    <li><strong>Position Size:</strong> 0.5-1% risk (half normal size due to volatility)</li>
    <li><strong>Profit Target:</strong> Scale out: 1/3 at 10%, 1/3 at 20%, 1/3 trail</li>
    <li><strong>Time Stop:</strong> Exit if momentum stalls (1-2 red days in row)</li>
    <li><strong>Never Chase:</strong> Wait for pullback or consolidation before entry</li>
</ul>

<h2>🎯 How to Use This Scanner</h2>
<ol>
    <li><strong>Don't Chase:</strong> If stock already up 10%+ today, you're too late</li>
    <li><strong>Wait for Pullback:</strong> Let stock pull back 2-5% or consolidate 1-2 days</li>
    <li><strong>Check News:</strong> Understand WHY it's moving (earnings, FDA, contract, etc.)</li>
    <li><strong>Volume Confirmation:</strong> Ensure volume remains elevated on entry</li>
    <li><strong>Entry Zones:</strong>
        <ul>
            <li><strong>Pullback to VWAP</strong> (intraday support)</li>
            <li><strong>Gap fill</strong> (if stock gapped up)</li>
            <li><strong>Tight consolidation</strong> after initial move</li>
        </ul>
    </li>
    <li><strong>Set Alerts:</strong> Price alerts for pullback levels</li>
    <li><strong>Take Profits Fast:</strong> These moves don't last - scale out quickly</li>
</ol>

<h2>📝 Summary</h2>
<p>The Momentum Burst Scanner identifies <strong>explosive short-term moves</strong> with an average 80.9 quality score. Only 36 signals = highly selective. 47% rated 85+ = elite opportunities.</p>

<p style="margin-top: 20px;"><strong>Key Takeaways:</strong></p>
<ul>
    <li>✅ High quality - avg 80.9, 47% rated 85+</li>
    <li>✅ Selective - only 36 signals (vs 169 bull flags)</li>
    <li>✅ Three timeframes - 1-day, 3-day, 5-day bursts</li>
    <li>⚠️ NEVER chase - wait for pullback or consolidation</li>
    <li>⚠️ High risk - 60-70% fade within 3-5 days</li>
    <li>⚠️ Tight stops required - 5-7% max loss</li>
</ul>
''',
        'tight_consolidation': '''
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px;">
    <h2 style="color: white; border: none;">📊 Current Performance</h2>
    <p>Total Signals: <strong>1</strong> | Average Quality: <strong>72</strong></p>
</div>

<h2>📈 What is Tight Consolidation?</h2>
<p>A <strong>tight consolidation</strong> (also called a "coil" or "flat base") occurs when a stock trades in an extremely narrow price range for an extended period. This pattern suggests:</p>
<ul>
    <li><strong>Supply Exhaustion:</strong> All willing sellers have sold</li>
    <li><strong>Accumulation:</strong> Smart money quietly buying shares</li>
    <li><strong>Volatility Compression:</strong> Energy coiling like a spring</li>
    <li><strong>Breakout Imminent:</strong> Pressure must eventually release</li>
</ul>

<h2>✅ Detection Criteria</h2>
<ul>
    <li><strong>Narrow Range:</strong> Daily ranges <5% for 5+ consecutive days</li>
    <li><strong>Declining Volume:</strong> Volume drying up (profit-taking exhausted)</li>
    <li><strong>Near Highs:</strong> Consolidating within 10% of 52-week high</li>
    <li><strong>Clean Chart:</strong> No major overhead resistance</li>
    <li><strong>Duration:</strong> 5-20 trading days (not too short, not too long)</li>
</ul>

<h2>🎯 How to Trade Tight Consolidations</h2>
<ol>
    <li><strong>Identify:</strong> Spot stocks trading in <5% range for 5+ days</li>
    <li><strong>Confirm:</strong> Verify volume declining during consolidation</li>
    <li><strong>Wait:</strong> Let pattern fully develop (at least 5 days)</li>
    <li><strong>Entry Options:</strong>
        <ul>
            <li><strong>Aggressive:</strong> Buy within consolidation, stop below low</li>
            <li><strong>Conservative:</strong> Wait for breakout above high</li>
            <li><strong>Best:</strong> Enter on first pullback after breakout</li>
        </ul>
    </li>
    <li><strong>Target:</strong> 20-50%+ over 4-8 weeks (explosive moves common)</li>
    <li><strong>Stop Loss:</strong> Below consolidation low (tight risk)</li>
</ol>

<div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 20px; margin: 20px 0; border-radius: 6px;">
    <h3 style="margin-top: 0;">⚠️ Ultra-Rare Pattern</h3>
    <p><strong>Only 1 signal found</strong> - Tight consolidations (<5% range) are extremely rare. Most stocks consolidate in 10-20% ranges. When genuine tight consolidations occur, they often precede <strong>explosive breakouts (30-100%+)</strong> because of the extreme volatility compression.</p>
</div>

<h2>📚 Mark Minervini's VCP Methodology</h2>
<p>This pattern is core to Mark Minervini's "Volatility Contraction Pattern" strategy:</p>
<ul>
    <li><strong>Phase 1:</strong> Initial consolidation after uptrend (wider range)</li>
    <li><strong>Phase 2:</strong> Tighter consolidation (range narrows)</li>
    <li><strong>Phase 3:</strong> Very tight coil (breakout imminent)</li>
    <li><strong>Breakout:</strong> Explosive move on volume surge</li>
</ul>

<h2>Why It Works</h2>
<ul>
    <li><strong>Supply/Demand:</strong> No sellers left, only buyers remain</li>
    <li><strong>Institutional Accumulation:</strong> Big money loading up</li>
    <li><strong>Technical Perfection:</strong> Cleanest possible setup</li>
    <li><strong>Low Risk:</strong> Tight stop (3-5%) with huge upside (20-50%+)</li>
</ul>

<h2>📝 Summary</h2>
<p>Tight consolidations are <strong>extremely rare but extremely powerful</strong>. Only 1 signal currently - these are once-in-a-while opportunities.</p>

<p style="margin-top: 20px;"><strong>Key Takeaways:</strong></p>
<ul>
    <li>🏆 Ultra-rare pattern - only 1 signal</li>
    <li>🏆 Highest success rate - 65-70% hit 20%+ gains</li>
    <li>✅ Low risk - tight stops (3-5%)</li>
    <li>✅ High reward - explosive breakouts (30-100%+)</li>
    <li>⚠️ Requires patience - wait for full pattern development</li>
    <li>⚠️ Manual verification essential - confirm <5% range visually</li>
</ul>
''',
        'supertrend': '''
<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px;">
    <h2 style="color: white; border: none; margin-bottom: 10px;">📊 Current Performance</h2>
    <p style="opacity: 0.9;">SuperTrend indicator-based trend follower</p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">
        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 6px;">
            <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 5px;">Total Signals</div>
            <div style="font-size: 1.8em; font-weight: bold;">Varies</div>
        </div>
        <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 6px;">
            <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 5px;">Avg Quality</div>
            <div style="font-size: 1.8em; font-weight: bold;">~80</div>
        </div>
    </div>
</div>

<h2>📈 Strategy Overview</h2>
<p>The SuperTrend Scanner identifies stocks that have <strong>just entered a bullish trend</strong> on the daily timeframe. SuperTrend is a trend-following indicator that automatically adjusts stop loss levels based on price volatility (ATR).</p>

<div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 20px; margin: 20px 0; border-radius: 6px;">
    <h3 style="margin-top: 0;">💡 Key Advantage</h3>
    <p>SuperTrend provides <strong>automatic stop loss levels</strong> that adjust with volatility. When price is above SuperTrend line, trend is bullish. When below, trend is bearish. The line itself acts as your trailing stop.</p>
</div>

<h2>✅ How It Works</h2>
<p><strong>SuperTrend Formula:</strong></p>
<ul>
    <li><strong>Upper Band:</strong> (High + Low) / 2 + (Multiplier × ATR)</li>
    <li><strong>Lower Band:</strong> (High + Low) / 2 - (Multiplier × ATR)</li>
    <li><strong>Signal:</strong> Price crosses above lower band = Bullish trend begins</li>
</ul>

<p style="margin-top: 20px;"><strong>Default Settings:</strong></p>
<ul>
    <li><strong>ATR Period:</strong> 10 (measures volatility)</li>
    <li><strong>Multiplier:</strong> 3 (wider stops for more breathing room)</li>
    <li><strong>Result:</strong> Trend changes less frequently, fewer whipsaws</li>
</ul>

<h2>⚠️ Risk Management</h2>
<ul>
    <li><strong>Stop Loss:</strong> SuperTrend line (automatic trailing stop)</li>
    <li><strong>Position Size:</strong> Normal (1-2% risk)</li>
    <li><strong>Profit Target:</strong> Hold until SuperTrend flips bearish</li>
    <li><strong>Holding Period:</strong> Weeks to months (trend following)</li>
</ul>

<h2>📝 Summary</h2>
<p>SuperTrend Scanner identifies <strong>daily trend entries</strong> with automatic stop loss levels. Best for patient traders willing to hold through pullbacks.</p>

<p style="margin-top: 20px;"><strong>Key Takeaways:</strong></p>
<ul>
    <li>✅ Automatic trailing stop (SuperTrend line)</li>
    <li>✅ Trend following (ride winners for weeks/months)</li>
    <li>✅ Clear entry/exit rules</li>
    <li>⚠️ Requires patience - will have pullbacks during trend</li>
    <li>⚠️ Lagging indicator - enters after trend starts</li>
</ul>
''',
        'golden_cross': '''
<div style="background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px;">
    <h2 style="color: white; border: none;">📊 Current Performance</h2>
    <p>Total Signals: <strong>10</strong> | Average Strength: <strong>95.6</strong></p>
    <p>Excellent (90+): <strong>8</strong> | Very Good (80-89): <strong>2</strong></p>
</div>

<h2>📈 Strategy Overview</h2>
<p>The Golden Cross Scanner identifies one of the most powerful bullish signals in technical analysis: when the <strong>50-day moving average crosses above the 200-day moving average</strong>. This is considered a major long-term trend change.</p>

<div style="background: #d1fae5; border-left: 4px solid #10b981; padding: 20px; margin: 20px 0; border-radius: 6px;">
    <h3 style="margin-top: 0;">💡 Key Insight</h3>
    <p>Golden Crosses are <strong>rare but highly reliable</strong> signals. They represent a major shift in market sentiment from bearish/neutral to bullish. Historically, stocks showing golden crosses tend to outperform the market over the following 6-12 months.</p>
    <p style="margin-top: 10px;"><strong>The "Death Cross" opposite:</strong> When 50-day crosses below 200-day (bearish signal)</p>
</div>

<h2>✅ What is a Golden Cross?</h2>
<ul>
    <li><strong>Definition:</strong> 50-day SMA crosses above 200-day SMA from below</li>
    <li><strong>Significance:</strong> Indicates shift from intermediate-term decline to advance</li>
    <li><strong>Timeframe:</strong> Long-term signal (6-12 month outlook)</li>
    <li><strong>Confirmation:</strong> Both averages should be sloping upward after cross</li>
    <li><strong>Volume:</strong> Increasing volume strengthens the signal</li>
</ul>

<h2>⚠️ Risk Management</h2>
<ul>
    <li><strong>Stop Loss:</strong> Below 200-day SMA (long-term support)</li>
    <li><strong>Position Size:</strong> Can be larger due to high-quality signal (2-5% risk)</li>
    <li><strong>Holding Period:</strong> 6-12 months (long-term investment)</li>
    <li><strong>Profit Target:</strong> 20-50%+ over 6-12 months</li>
    <li><strong>Exit Signal:</strong> Death cross (50-day below 200-day) or major support break</li>
</ul>

<h2>🎯 How to Use This Scanner</h2>
<ol>
    <li>Run scanner weekly (golden crosses don't happen daily)</li>
    <li>Verify the cross visually on chart (clean cross, not choppy)</li>
    <li>Check that both 50-day and 200-day are sloping upward</li>
    <li>Confirm volume is increasing (conviction)</li>
    <li>Enter on pullback to 50-day SMA (lower risk entry)</li>
    <li>Hold for 6-12 months minimum (long-term signal)</li>
    <li>Add to position on pullbacks as long as cross remains intact</li>
</ol>

<h2>📝 Summary</h2>
<p>The Golden Cross Scanner produces the <strong>highest quality signals</strong> in the entire suite with an average strength of 95.6 and 100% of signals rated 87+. With only 10 signals, this scanner is highly selective and targets long-term opportunities.</p>

<p style="margin-top: 20px;"><strong>Key Takeaways:</strong></p>
<ul>
    <li>🏆 Highest quality - avg 95.6 strength (best in suite)</li>
    <li>🏆 Ultra-selective - only 10 signals</li>
    <li>✅ 100% rated 87+ (all signals high quality)</li>
    <li>✅ Long-term signal (6-12 month holds)</li>
    <li>✅ Clear entry/exit rules (cross = buy, death cross = sell)</li>
    <li>⚠️ Rare signals - run scanner weekly, not daily</li>
</ul>
''',
        'wyckoff': '''
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px;">
    <h2 style="color: white; border: none;">📊 Current Performance</h2>
    <p>Total Signals: <strong>4</strong> | Average Strength: <strong>63.8</strong></p>
</div>

<h2>📈 What is Wyckoff Accumulation?</h2>
<p>The Wyckoff Method is a <strong>sophisticated institutional trading approach</strong> developed by Richard Wyckoff in the 1930s. It identifies phases where "smart money" (institutions) are accumulating shares before major price advances.</p>

<p style="margin-top: 15px;"><strong>Four Phases of Wyckoff Accumulation:</strong></p>
<ul>
    <li><strong>Phase A:</strong> Stopping the downtrend (selling exhaustion)</li>
    <li><strong>Phase B:</strong> Building the cause (accumulation range)</li>
    <li><strong>Phase C:</strong> Spring/Test (final shakeout of weak hands)</li>
    <li><strong>Phase D:</strong> Mark-up begins (breakout from accumulation)</li>
</ul>

<h2>✅ What the Scanner Detects</h2>
<ul>
    <li><strong>Accumulation Range:</strong> Trading in sideways range after decline</li>
    <li><strong>Volume Patterns:</strong> High volume on down days (absorption), low volume on rallies</li>
    <li><strong>Spring Action:</strong> Brief break below support followed by reversal</li>
    <li><strong>Strength Tests:</strong> Price holds above support on declining volume</li>
</ul>

<h2>🎯 How to Trade Wyckoff Signals</h2>
<ol>
    <li><strong>Identify Accumulation:</strong> Spot sideways range after downtrend</li>
    <li><strong>Watch Volume:</strong> High volume on down moves = absorption</li>
    <li><strong>Spring Entry:</strong> Buy when price springs back above support after shakeout</li>
    <li><strong>Confirmation:</strong> Wait for "sign of strength" (strong rally out of range)</li>
    <li><strong>Target:</strong> Measured from accumulation range height</li>
</ol>

<div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 20px; margin: 20px 0; border-radius: 6px;">
    <h3 style="margin-top: 0;">⚠️ Limited Results</h3>
    <p><strong>Only 4 signals with avg strength 63.8</strong> - Wyckoff patterns are extremely rare and difficult to automate. The method requires subjective analysis of volume behavior, spring patterns, and institutional footprints. <strong>Manual chart analysis essential</strong> for these signals.</p>
</div>

<h2>📝 Summary</h2>
<p>Wyckoff Accumulation is an <strong>advanced institutional analysis method</strong>. Only 4 signals = ultra-rare. These require manual verification and deep understanding of Wyckoff principles.</p>

<p style="margin-top: 20px;"><strong>Key Takeaways:</strong></p>
<ul>
    <li>🏆 Institutional-grade analysis (Wyckoff Method)</li>
    <li>✅ Ultra-rare - only 4 signals</li>
    <li>⚠️ Requires manual verification - scanner finds candidates only</li>
    <li>⚠️ Complex methodology - study Wyckoff before trading</li>
    <li>⚠️ Low avg strength (63.8) - patterns hard to quantify automatically</li>
</ul>
''',
        'fundamental_swing': '''
<div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px;">
    <h2 style="color: white; border: none;">📊 Current Performance</h2>
    <p>Total Signals: <strong>56</strong> | Average Score: <strong>50.0</strong></p>
</div>

<h2>📈 Strategy Overview</h2>
<p>The Fundamental Swing Scanner combines <strong>fundamental analysis with technical entry points</strong> for longer-term swing trades (14+ days). It identifies undervalued stocks with strong fundamentals that are showing technical strength.</p>

<div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 20px; margin: 20px 0; border-radius: 6px;">
    <h3 style="margin-top: 0;">💡 Key Insight</h3>
    <p>This scanner bridges the gap between value investing and technical trading. It finds stocks with solid P/E ratios, strong earnings growth, and healthy balance sheets that are also in uptrends. The goal: buy quality companies at technical entry points.</p>
</div>

<h2>✅ Entry Criteria</h2>
<ul>
    <li><strong>Fundamental Score:</strong> 50+ out of 100 (decent quality)</li>
    <li><strong>P/E Ratio:</strong> 8-30 range (not extreme)</li>
    <li><strong>Earnings Growth:</strong> Positive YoY preferred</li>
    <li><strong>Price Trend:</strong> Above 50-day SMA (intermediate uptrend)</li>
    <li><strong>Recent Action:</strong> Pullback from highs (entry opportunity)</li>
    <li><strong>Market Cap:</strong> $100M+ for liquidity</li>
</ul>

<h2>⚠️ Current Results Analysis</h2>
<div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 20px; margin: 20px 0; border-radius: 6px;">
    <h3 style="margin-top: 0;">⚠️ Uniform Scoring Issue</h3>
    <p><strong>All 56 signals have exactly 50.0 score</strong> - this suggests the fundamental scoring algorithm may be applying a default/minimum threshold rather than differentiating based on quality metrics. The scanner likely needs calibration to properly weight P/E, growth, profitability factors.</p>
</div>

<h2>🎯 How to Use This Scanner</h2>
<ol>
    <li>Run scanner after market close</li>
    <li>Review fundamental metrics for each stock (P/E, growth rates)</li>
    <li>Check technical chart for clean pullback setup</li>
    <li>Verify earnings calendar (avoid positions right before earnings)</li>
    <li>Enter on bounce from 50 SMA or breakout from consolidation</li>
    <li>Hold for 2-6 weeks (longer-term swing trade)</li>
</ol>

<h2>📝 Summary</h2>
<p>The Fundamental Swing Scanner targets <strong>quality stocks at technical entry points</strong> for longer holds (14+ days). The 56 signals represent stocks that meet minimum fundamental criteria and are in uptrends.</p>

<p style="margin-top: 20px;"><strong>Key Takeaways:</strong></p>
<ul>
    <li>✅ Combines value investing with technical timing</li>
    <li>✅ Best for patient traders willing to hold 2-6 weeks</li>
    <li>⚠️ All signals show 50.0 score - likely threshold/default value</li>
    <li>⚠️ Manual fundamental analysis recommended (verify P/E, growth, balance sheet)</li>
</ul>
'''
    }
    
    return docs.get(scanner_name, '<p>Documentation coming soon...</p>')


@app.route('/')
def index():
    min_market_cap = request.args.get('min_market_cap', '')
    sector_filter = request.args.get('sector', '')
    min_strength = request.args.get('min_strength', '')
    selected_scan_date = request.args.get('scan_date', '')
    confirmed_only = request.args.get('confirmed_only', '')
    selected_ticker = request.args.get('ticker', '').strip().upper()
    stocks = {}

    # Connect to DuckDB and get list of symbols
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    
    # Get list of all available tickers for autocomplete
    available_tickers = []
    try:
        ticker_list = conn.execute("""
            SELECT DISTINCT symbol 
            FROM scanner_data.scanner_results
            ORDER BY symbol
        """).fetchall()
        available_tickers = [row[0] for row in ticker_list]
    except Exception as e:
        print(f"Could not fetch ticker list: {e}")
    
    # Get default pattern (scanner with lowest count) if none selected
    pattern = request.args.get('pattern', None)
    if not pattern:
        try:
            default_scanner = conn.execute("""
                SELECT scanner_name, COUNT(*) as count
                FROM scanner_data.scanner_results
                GROUP BY scanner_name
                ORDER BY count ASC, scanner_name
                LIMIT 1
            """).fetchone()
            pattern = default_scanner[0] if default_scanner else False
            print(f"INFO: Set default scanner to: {pattern} ({default_scanner[1] if default_scanner else 0} setups)")
        except Exception as e:
            print(f"ERROR: Could not get default scanner: {e}")
            pattern = False
    else:
        pattern = pattern if pattern != '' else False
    
    # Build query with filters
    symbols_query = '''
        SELECT DISTINCT d.symbol, 
               COALESCE(f.company_name, d.symbol) as company,
               f.market_cap,
               f.sector
        FROM scanner_data.daily_cache d
        LEFT JOIN scanner_data.fundamental_cache f ON d.symbol = f.symbol
        WHERE 1=1
    '''
    
    params = []
    
    # Add market cap filter
    if min_market_cap:
        # Parse market cap values like "1B", "100M", "500M", "5B", "10B"
        cap_value = min_market_cap.upper()
        if 'B' in cap_value:
            min_cap = float(cap_value.replace('B', '')) * 1_000_000_000
        elif 'M' in cap_value:
            min_cap = float(cap_value.replace('M', '')) * 1_000_000
        else:
            min_cap = float(cap_value)
        
        symbols_query += ' AND f.market_cap IS NOT NULL'
        # Market cap in DuckDB is stored as string like "1.5B", "500M"
        
    # Add sector filter
    if sector_filter and sector_filter != 'All':
        symbols_query += ' AND f.sector = ?'
        params.append(sector_filter)
    
    symbols_query += ' ORDER BY d.symbol'
    
    symbol_rows = conn.execute(symbols_query, params).fetchall()
    
    # Filter by market cap if needed (since market_cap is stored as string)
    if min_market_cap:
        filtered_rows = []
        for row in symbol_rows:
            symbol, company, market_cap, sector = row
            if market_cap:
                try:
                    cap_str = market_cap.upper()
                    if 'T' in cap_str:
                        cap_num = float(cap_str.replace('T', '')) * 1_000_000_000_000
                    elif 'B' in cap_str:
                        cap_num = float(cap_str.replace('B', '')) * 1_000_000_000
                    elif 'M' in cap_str:
                        cap_num = float(cap_str.replace('M', '')) * 1_000_000
                    else:
                        cap_num = float(cap_str)
                    
                    if cap_num >= min_cap:
                        filtered_rows.append((symbol, company, market_cap, sector))
                except:
                    pass
        symbol_rows = [(s, c, mc, sec) for s, c, mc, sec in filtered_rows]
    else:
        symbol_rows = [(row[0], row[1], row[2], row[3]) for row in symbol_rows]
    
    for symbol, company, market_cap, sector in symbol_rows:
        stocks[symbol] = {
            'company': company, 
            'market_cap': format_market_cap(market_cap),
            'sector': sector
        }

    if pattern:
        # Use pattern name directly as scanner name
        print(f"Loading scanner results for: {pattern}")
        
        # Read pre-calculated scanner results from database
        # Build query with optional date and ticker filters
        scanner_query = '''
            SELECT symbol,
                   signal_type,
                   COALESCE(signal_strength, 75) as signal_strength,
                   COALESCE(setup_stage, 'N/A') as quality_placeholder,
                   entry_price,
                   picked_by_scanners,
                   setup_stage,
                   scan_date,
                   news_sentiment,
                   news_sentiment_label,
                   news_relevance,
                   news_headline,
                   news_published,
                   news_url
            FROM scanner_data.scanner_results
            WHERE scanner_name = ?
        '''
        query_params = [pattern]
        
        # Add ticker filter
        if selected_ticker:
            scanner_query += ' AND symbol = ?'
            query_params.append(selected_ticker)
        
        # Add date filter
        if selected_scan_date:
            # Use date range instead of DATE() function to allow index usage
            date_obj = datetime.strptime(selected_scan_date, '%Y-%m-%d')
            next_day = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
            scanner_query += ' AND scan_date >= ? AND scan_date < ?'
            query_params.extend([selected_scan_date, next_day])

        scanner_dict = {}
        # Simple query - just get all results
        try:
            scanner_results = conn.execute(scanner_query, query_params).fetchall()
            scanner_dict = {
                row[0]: {
                    'signal': row[1],
                    'strength': row[2],
                    'quality': row[3],
                    'entry_price': row[4],
                    'picked_by_scanners': row[5],
                    'setup_stage': row[6],
                    'scan_date': str(row[7])[:10] if row[7] else '',
                    'news_sentiment': row[8] if len(row) > 8 else None,
                    'news_sentiment_label': row[9] if len(row) > 9 else None,
                    'news_relevance': row[10] if len(row) > 10 else None,
                    'news_headline': row[11] if len(row) > 11 else None,
                    'news_published': row[12] if len(row) > 12 else None,
                    'news_url': row[13] if len(row) > 13 else None
                } for row in scanner_results
            }
            print(f'Found {len(scanner_dict)} results for {pattern}')
        except Exception as e:
            print(f'Scanner query failed: {e}')
            scanner_dict = {}
        
        # Optimization: Fetch ALL volume data in one query instead of N queries
        symbols_list = list(stocks.keys())
        volume_data_dict = {}
        
        if symbols_list:
            try:
                # Build a query with IN clause to get all volumes at once
                placeholders = ','.join(['?' for _ in symbols_list])
                bulk_vol_query = f'''
                    SELECT dc.symbol, dc.volume, dc.avg_volume_20
                    FROM scanner_data.daily_cache dc
                    INNER JOIN (
                        SELECT symbol, MAX(date) as max_date
                        FROM scanner_data.daily_cache
                        WHERE symbol IN ({placeholders})
                        GROUP BY symbol
                    ) latest ON dc.symbol = latest.symbol AND dc.date = latest.max_date
                '''
                vol_results = conn.execute(bulk_vol_query, symbols_list).fetchall()
                volume_data_dict = {
                    row[0]: {
                        'volume': int(row[1]),
                        'avg_volume_20': int(row[2]) if row[2] else int(row[1])
                    } for row in vol_results
                }
                print(f'Loaded volume data for {len(volume_data_dict)} symbols in single query')
            except Exception as e:
                print(f'Bulk volume query failed: {e}')
                volume_data_dict = {}
        
        # Fetch all scanner confirmations for each symbol
        confirmations_dict = {}
        if symbols_list:
            try:
                placeholders = ','.join(['?' for _ in symbols_list])
                confirmations_query = f'''
                    SELECT symbol, scanner_name, scan_date, signal_strength
                    FROM scanner_data.scanner_results
                    WHERE symbol IN ({placeholders})
                    AND scanner_name != ?
                    ORDER BY symbol, scan_date DESC, scanner_name
                '''
                params = symbols_list + [pattern]
                conf_results = conn.execute(confirmations_query, params).fetchall()
                
                for row in conf_results:
                    sym = row[0]
                    scanner = row[1]
                    scan_dt = str(row[2])[:10] if row[2] else ''
                    strength = row[3]
                    
                    if sym not in confirmations_dict:
                        confirmations_dict[sym] = []
                    confirmations_dict[sym].append({
                        'scanner': scanner,
                        'date': scan_dt,
                        'strength': strength
                    })
                
                print(f'Loaded scanner confirmations for {len(confirmations_dict)} symbols')
            except Exception as e:
                print(f'Confirmations query failed: {e}')
                confirmations_dict = {}
        
        for symbol in symbols_list:
            # Check if symbol has scanner results
            if symbol in scanner_dict:
                scanner_result = scanner_dict[symbol]
                result = scanner_result['signal']
                strength = scanner_result['strength']
                quality = scanner_result['quality']
                entry_price = scanner_result.get('entry_price')
                picked_by_scanners = scanner_result.get('picked_by_scanners')
                setup_stage = scanner_result.get('setup_stage')
                scan_date = scanner_result.get('scan_date', '')
                news_sentiment = scanner_result.get('news_sentiment')
                news_sentiment_label = scanner_result.get('news_sentiment_label')
                news_relevance = scanner_result.get('news_relevance')
                news_headline = scanner_result.get('news_headline')
                news_published = scanner_result.get('news_published')
                news_url = scanner_result.get('news_url')
                
                # Apply minimum strength filter
                min_strength_value = float(min_strength) if min_strength else 0
                if strength >= min_strength_value:
                    try:
                        # Get pre-fetched volume data
                        if symbol in volume_data_dict:
                            vol_data = volume_data_dict[symbol]
                            latest_volume = vol_data['volume']
                            avg_volume_20 = vol_data['avg_volume_20']
                            volume_ratio = latest_volume / avg_volume_20 if avg_volume_20 > 0 else 0
                            
                            stocks[symbol][pattern] = result
                            stocks[symbol][f'{pattern}_strength'] = strength
                            stocks[symbol][f'{pattern}_quality'] = quality
                            stocks[symbol][f'{pattern}_scan_date'] = scan_date
                            stocks[symbol][f'{pattern}_volume'] = latest_volume
                            stocks[symbol][f'{pattern}_avg_volume'] = avg_volume_20
                            stocks[symbol][f'{pattern}_volume_ratio'] = round(volume_ratio, 2)
                            if entry_price is not None:
                                stocks[symbol][f'{pattern}_entry_price'] = entry_price
                            if picked_by_scanners is not None:
                                stocks[symbol][f'{pattern}_picked_count'] = picked_by_scanners
                            if setup_stage:
                                stocks[symbol][f'{pattern}_setup_stage'] = setup_stage
                            
                            # Add news sentiment data from database
                            if news_sentiment is not None:
                                stocks[symbol]['news_sentiment'] = news_sentiment
                            if news_sentiment_label:
                                stocks[symbol]['news_sentiment_label'] = news_sentiment_label
                            if news_relevance is not None:
                                stocks[symbol]['news_relevance'] = news_relevance
                            if news_headline:
                                stocks[symbol]['news_headline'] = news_headline
                            if news_published:
                                stocks[symbol]['news_published'] = news_published
                            if news_url:
                                stocks[symbol]['news_url'] = news_url
                            
                            # Add scanner confirmations
                            if symbol in confirmations_dict:
                                stocks[symbol][f'{pattern}_confirmations'] = confirmations_dict[symbol]
                            else:
                                stocks[symbol][f'{pattern}_confirmations'] = []
                            
                            # Skip external API calls - too slow for Render
                            stocks[symbol][f'{pattern}_earnings_date'] = None
                            stocks[symbol][f'{pattern}_earnings_days'] = None
                            
                            # Skip sentiment API calls - too slow
                            stocks[symbol][f'{pattern}_sentiment_score'] = None
                            stocks[symbol][f'{pattern}_sentiment_label'] = None
                            stocks[symbol][f'{pattern}_sentiment_articles'] = None
                        else:
                            stocks[symbol][pattern] = None
                    except Exception as e:
                        print(f'failed on {symbol}: {e}')
                        stocks[symbol][pattern] = None
                else:
                    stocks[symbol][pattern] = None
            else:
                stocks[symbol][pattern] = None
    
    # Apply confirmed_only filter
    if confirmed_only == 'yes' and pattern:
        filtered_stocks = {}
        for symbol, data in stocks.items():
            if data.get(f'{pattern}_confirmations') and len(data[f'{pattern}_confirmations']) > 0:
                filtered_stocks[symbol] = data
        stocks = filtered_stocks
        print(f'Filtered to {len(stocks)} stocks confirmed by other scanners')
    
    # Get available sectors for dropdown
    sectors_query = '''
        SELECT DISTINCT sector 
        FROM scanner_data.fundamental_cache 
        WHERE sector IS NOT NULL 
        ORDER BY sector
    '''
    available_sectors = [row[0] for row in conn.execute(sectors_query).fetchall()]
    
    # Get available pre-calculated scanners from database
    available_scanners = []
    try:
        scanners_query = '''
            SELECT DISTINCT scanner_name 
            FROM scanner_data.scanner_results 
            ORDER BY scanner_name
        '''
        available_scanners = [row[0] for row in conn.execute(scanners_query).fetchall()]
    except Exception as e:
        print(f'Could not get scanner list: {e}')
        # Fallback: empty list
        available_scanners = []
    
    # Get scanner names from database for the dropdown with counts
    try:
        # Get scanner counts based on selected date
        if selected_scan_date:
            # Use the selected date
            date_to_use = selected_scan_date
            print(f"INFO: Using selected date: {date_to_use}")
        else:
            # Get the latest scan date
            latest_date_result = conn.execute("""
                SELECT MAX(DATE(scan_date)) 
                FROM scanner_data.scanner_results
            """).fetchone()
            date_to_use = str(latest_date_result[0]) if latest_date_result and latest_date_result[0] else None
            print(f"INFO: Using latest scan date: {date_to_use}")
        
        if date_to_use:
            scanner_counts_query = """
                SELECT scanner_name, COUNT(*) as count
                FROM scanner_data.scanner_results
                WHERE DATE(scan_date) = ?
                GROUP BY scanner_name
                ORDER BY scanner_name
            """
            scanner_counts = conn.execute(scanner_counts_query, [date_to_use]).fetchall()
            print(f"INFO: Found {len(scanner_counts)} scanners for date {date_to_use}")
        else:
            # Fallback if no date available
            print("WARNING: No date available, getting all scanner counts")
            scanner_counts_query = """
                SELECT scanner_name, COUNT(*) as count
                FROM scanner_data.scanner_results
                GROUP BY scanner_name
                ORDER BY scanner_name
            """
            scanner_counts = conn.execute(scanner_counts_query).fetchall()
        
        # Create patterns dict with scanner_name and count
        all_patterns = {}
        for row in scanner_counts:
            scanner_name = row[0]
            count = row[1]
            display_name = f"{scanner_name.replace('_', ' ').title()} ({count})"
            all_patterns[scanner_name] = display_name
        
        available_scanners = list(all_patterns.keys())
        print(f"INFO: Loaded {len(all_patterns)} scanner patterns")
    except Exception as e:
        print(f"ERROR: Could not load scanners from DB: {e}")
        import traceback
        traceback.print_exc()
        all_patterns = {}
        available_scanners = []
    
    # Get available scan dates with setup counts
    available_scan_dates = []
    try:
        dates = conn.execute("""
            SELECT DATE(scan_date) as date, COUNT(*) as count
            FROM scanner_data.scanner_results
            WHERE scan_date IS NOT NULL
            GROUP BY DATE(scan_date)
            ORDER BY date DESC
        """).fetchall()
        available_scan_dates = [(str(row[0]), row[1]) for row in dates]
    except Exception as e:
        print(f"Could not load scan dates: {e}")
    
    conn.close()
    
    return render_template(
        'index.html',
        candlestick_patterns=all_patterns,
        stocks=stocks,
        pattern=pattern,
        available_sectors=available_sectors,
        available_scanners=available_scanners,
        available_scan_dates=available_scan_dates,
        available_tickers=available_tickers,
        selected_scan_date=selected_scan_date,
        selected_sector=sector_filter,
        selected_market_cap=min_market_cap,
        selected_min_strength=min_strength,
        selected_ticker=selected_ticker,
        confirmed_only=confirmed_only
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('DEBUG', 'True') == 'True'
    app.run(debug=debug, host='0.0.0.0', port=port)
