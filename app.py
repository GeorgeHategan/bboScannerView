import os
from dotenv import load_dotenv
import duckdb
from fastapi import FastAPI, Request, Query, Form, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from datetime import datetime, timedelta
from typing import Optional
from authlib.integrations.starlette_client import OAuth, OAuthError

# Load environment variables from .env file
load_dotenv()

# Initialize OAuth
oauth = OAuth()

# HARDCODED SCANNER COLOR MAPPING - Single source of truth
# This must match the JavaScript SCANNER_COLORS in index.html
SCANNER_COLORS = {
    'accumulation_distribution': '#F2AE7F',  # salmon
    'breakout':                  '#96FFBD',  # florescent green
    'bull_flag':                 '#DC3E26',  # bloodWeist
    'candlestick_bullish':       '#CEE4B3',  # mint Green
    'candlestick_continuation':  '#F9EC7E',  # blondeYello
    'cup_and_handle':            '#D5CAE4',  # liiac
    'fundamental_swing':         '#80C4B7',  # teal
    'golden_cross':              '#9EE8E1',  # neonBlue
    'momentum_burst':            '#EC6D67',  # colral
    'supertrend':                '#4AAFD5',  # bkue
    'tight_consolidation':       '#E59462',  # orangfe
    'volatility_macd':           '#E3CCB2',  # beige
    'wyckoff':                   '#D8905A',  # Muted Orange
    'wyckoff_accumulation':      '#6BAA6B'   # Muted Green
}

# Candlestick pattern strength weights (out of 10)
PATTERN_WEIGHTS = {
    # Strong reversal patterns (8-10)
    'CDLMORNINGSTAR': 9.5, 'CDLEVENINGSTAR': 9.5,
    'CDL3WHITESOLDIERS': 9.0, 'CDL3BLACKCROWS': 9.0,
    'CDLABANDONEDBABY': 10.0,
    'CDLENGULFING': 8.5, 'CDLPIERCING': 8.0,
    'CDLHAMMER': 8.0, 'CDLHANGINGMAN': 8.0,
    'CDLINVERTEDHAMMER': 7.5, 'CDLSHOOTINGSTAR': 7.5,
    
    # Moderate reversal patterns (6-7.5)
    'CDLHARAMI': 7.0, 'CDLHARAMICROSS': 7.5,
    'CDLDARKCLOUDCOVER': 7.5,
    'CDLMORNINGDOJISTAR': 8.0, 'CDLEVENINGDOJISTAR': 8.0,
    'CDL3INSIDE': 6.5, 'CDL3OUTSIDE': 7.0,
    'CDLKICKING': 8.5,
    
    # Continuation patterns (6-8)
    'CDLRISEFALL3METHODS': 7.5, 'CDLMATHOLD': 7.0,
    'CDLSEPARATINGLINES': 6.5, 'CDLIDENTICAL3CROWS': 7.5,
    'CDLBREAKAWAY': 7.0, 'CDLCONCEALBABYSWALL': 6.5,
    
    # Single candle patterns (5-7)
    'CDLMARUBOZU': 6.5, 'CDLCLOSINGMARUBOZU': 6.0,
    'CDLBELTHOLD': 6.5, 'CDLSPINNINGTOP': 5.0,
    'CDLDOJI': 6.0, 'CDLDRAGONFLYDOJI': 7.0,
    'CDLGRAVESTONEDOJI': 7.0, 'CDLLONGLEGGEDDOJI': 5.5,
    'CDLRICKSHAWMAN': 5.0,
    
    # Other patterns (4-6)
    'CDLHIGHWAVE': 5.5, 'CDLSHORTLINE': 4.0,
    'CDLTAKURI': 6.5, 'CDLHOMINGPIGEON': 6.0,
    'CDLLADDERBOTTOM': 7.5, 'CDLSTICKSANDWICH': 6.0,
    'CDLTRISTAR': 7.5, 'CDLUNIQUE3RIVER': 7.0,
    'CDL2CROWS': 6.5, 'CDL3LINESTRIKE': 7.0,
    'CDLADVANCEBLOCK': 6.5, 'CDLGAPSIDESIDEWHITE': 6.0,
}

app = FastAPI(title="BBO Scanner View", description="Stock Scanner Dashboard")
app.add_middleware(SessionMiddleware, secret_key=os.environ.get('SECRET_KEY', 'supersecret'))

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Favicon route
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.png", media_type="image/png")

# Configure Google OAuth
oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

# Helper: get allowed emails from env
def get_allowed_emails():
    return [e.strip() for e in os.environ.get('ALLOWED_EMAILS', '').split(',') if e.strip()]

# Dependency: require login and allowed email
def require_login(request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Not authenticated')
    
    email = user.get('email')
    allowed = get_allowed_emails()
    if not email or email not in allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Access denied - not in authorized user list')
    return email

# Login page - redirect to Google OAuth
@app.get('/login', response_class=HTMLResponse)
async def login_form(request: Request):
    # Check if user is already logged in
    user = request.session.get('user')
    if user:
        email = user.get('email')
        if email in get_allowed_emails():
            return RedirectResponse('/', status_code=302)
    
    # Show login page with Google button
    return templates.TemplateResponse('login.html', {'request': request, 'error': None})

# Google OAuth login redirect
@app.get('/auth/google')
async def google_login(request: Request):
    redirect_uri = request.url_for('google_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

# Google OAuth callback
@app.get('/auth/google/callback')
async def google_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        
        if not user_info:
            return templates.TemplateResponse('login.html', {
                'request': request, 
                'error': 'Failed to get user information from Google'
            })
        
        email = user_info.get('email')
        allowed = get_allowed_emails()
        
        print(f"DEBUG: Google OAuth callback - Email: {email}")
        print(f"DEBUG: Allowed emails: {allowed}")
        
        if email not in allowed:
            return templates.TemplateResponse('login.html', {
                'request': request,
                'error': f'Access denied. Email {email} is not authorized to access this application.'
            })
        
        # Store user info in session
        request.session['user'] = {
            'email': email,
            'name': user_info.get('name'),
            'picture': user_info.get('picture')
        }
        
        print(f"DEBUG: Login successful for {email}")
        return RedirectResponse('/', status_code=302)
        
    except OAuthError as e:
        print(f"ERROR: OAuth error: {e}")
        return templates.TemplateResponse('login.html', {
            'request': request,
            'error': f'Authentication failed: {str(e)}'
        })

# Logout
@app.get('/logout')
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse('/login', status_code=302)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Database configuration
# For local development, use MotherDuck to access production data
# For production (Render), use environment variable

# Scanner database (primary)
motherduck_token = os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN', '')
print(f"DEBUG: motherduck_token found: {bool(motherduck_token)}")
if motherduck_token:
    # Always use scanner_data database in MotherDuck
    DUCKDB_PATH = f'md:scanner_data?motherduck_token={motherduck_token}'
    print("INFO: Connecting to MotherDuck production database")
    print(f"INFO: Database path: md:scanner_data?motherduck_token=***")
else:
    # Fallback to local DB if no MotherDuck token - this will fail on Render
    DUCKDB_PATH = '/Users/george/scannerPOC/breakoutScannersPOCs/scanner_data.duckdb'
    print("WARNING: No motherduck_token found, using local database (may not have scanner_results)")
    print("ERROR: This will fail on Render - ensure MOTHERDUCK_TOKEN env var is set!")

# Options data database (secondary) - contains accumulation_signals and options_chain_changes tables
options_motherduck_token = os.environ.get('options_motherduck_token') or os.environ.get('OPTIONS_MOTHERDUCK_TOKEN', '')
print(f"DEBUG: options_motherduck_token found: {bool(options_motherduck_token)}")
if options_motherduck_token:
    OPTIONS_DUCKDB_PATH = f'md:options_data?motherduck_token={options_motherduck_token}'
    print("INFO: Options database configured: md:options_data?motherduck_token=***")
else:
    OPTIONS_DUCKDB_PATH = None
    print("INFO: No options database token found - options features disabled")

# Skip slow database initialization on startup - connect lazily when needed
print(f"INFO: Database path configured: {DUCKDB_PATH if not motherduck_token else 'md:scanner_data?motherduck_token=***'}")


def get_options_db_connection():
    """Get a connection to the options signals database."""
    if OPTIONS_DUCKDB_PATH:
        return duckdb.connect(OPTIONS_DUCKDB_PATH, read_only=True)
    return None


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



@app.get("/health")
async def health_check(request: Request, email: str = Depends(require_login)):
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "bbo-scanner-view"}


@app.get("/options-signals", response_class=HTMLResponse)
async def options_signals(
    request: Request,
    signal_type: Optional[str] = Query(None),
    signal_strength: Optional[str] = Query(None),
    sector: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    min_confidence: Optional[str] = Query(None),
    min_premium: Optional[str] = Query(None),
    asset_type: Optional[str] = Query(None),
    scan_date: Optional[str] = Query(None)
):
    """Display options accumulation signals from the options_data database."""
    
    # Convert string params to int (handle empty strings from form)
    min_confidence_val = int(min_confidence) if min_confidence else None
    min_premium_val = int(min_premium) if min_premium else None
    
    # Check if options database is configured
    if not OPTIONS_DUCKDB_PATH:
        return templates.TemplateResponse('options_signals.html', {
            'request': request,
            'error': 'Options database not configured. Please set OPTIONS_MOTHERDUCK_TOKEN.',
            'signals': [],
            'signal_types': [],
            'signal_strengths': [],
            'sectors': [],
            'stats': {}
        })
    
    try:
        conn = get_options_db_connection()
        if not conn:
            raise Exception("Could not connect to options database")
        
        # Get available filter options
        signal_types = [row[0] for row in conn.execute(
            "SELECT DISTINCT signal_type FROM accumulation_signals ORDER BY signal_type"
        ).fetchall()]
        
        signal_strengths = [row[0] for row in conn.execute(
            "SELECT DISTINCT signal_strength FROM accumulation_signals ORDER BY signal_strength"
        ).fetchall()]
        
        sectors = [row[0] for row in conn.execute(
            "SELECT DISTINCT sector FROM accumulation_signals WHERE sector IS NOT NULL ORDER BY sector"
        ).fetchall()]
        
        available_dates = [str(row[0]) for row in conn.execute(
            "SELECT DISTINCT signal_date FROM accumulation_signals ORDER BY signal_date DESC LIMIT 30"
        ).fetchall()]
        
        # Build the query with filters
        query = """
            SELECT 
                signal_id,
                signal_date,
                signal_type,
                underlying_symbol,
                sector,
                asset_type,
                option_symbol,
                strike,
                stock_price,
                dte,
                volume,
                curr_oi,
                oi_change_pct,
                premium_spent,
                confidence_score,
                signal_strength,
                notes
            FROM accumulation_signals
            WHERE 1=1
        """
        params = []
        
        if signal_type:
            query += " AND signal_type = ?"
            params.append(signal_type)
        
        if signal_strength:
            query += " AND signal_strength = ?"
            params.append(signal_strength)
        
        if sector:
            query += " AND sector = ?"
            params.append(sector)
        
        if symbol:
            query += " AND underlying_symbol = ?"
            params.append(symbol.upper())
        
        if min_confidence_val:
            query += " AND confidence_score >= ?"
            params.append(min_confidence_val)
        
        if min_premium_val:
            query += " AND premium_spent >= ?"
            params.append(min_premium_val)
        
        if asset_type:
            query += " AND asset_type = ?"
            params.append(asset_type)
        
        if scan_date:
            query += " AND signal_date = ?"
            params.append(scan_date)
        elif not symbol:
            # Default to latest date only if no specific symbol is requested
            # When viewing a specific symbol, show all dates
            latest_date = conn.execute(
                "SELECT MAX(signal_date) FROM accumulation_signals"
            ).fetchone()[0]
            if latest_date:
                query += " AND signal_date = ?"
                params.append(str(latest_date))
                scan_date = str(latest_date)
        
        query += " ORDER BY signal_date DESC, confidence_score DESC, premium_spent DESC LIMIT 200"
        
        results = conn.execute(query, params).fetchall()
        
        # Convert to list of dicts
        signals = []
        for row in results:
            signals.append({
                'signal_id': row[0],
                'signal_date': str(row[1]) if row[1] else '',
                'signal_type': row[2],
                'underlying_symbol': row[3],
                'sector': row[4] or 'ETF',
                'asset_type': row[5],
                'option_symbol': row[6],
                'strike': row[7],
                'stock_price': row[8],
                'dte': row[9],
                'volume': row[10],
                'curr_oi': row[11],
                'oi_change_pct': row[12],
                'premium_spent': row[13],
                'confidence_score': row[14],
                'signal_strength': row[15],
                'notes': row[16]
            })
        
        # Get stats for the dashboard
        stats_query = """
            SELECT 
                COUNT(*) as total_signals,
                COUNT(DISTINCT underlying_symbol) as unique_symbols,
                SUM(premium_spent) as total_premium,
                AVG(confidence_score) as avg_confidence,
                SUM(CASE WHEN signal_strength = 'EXTREME' THEN 1 ELSE 0 END) as extreme_count,
                SUM(CASE WHEN signal_strength = 'VERY_HIGH' THEN 1 ELSE 0 END) as very_high_count,
                SUM(CASE WHEN signal_strength = 'HIGH' THEN 1 ELSE 0 END) as high_count
            FROM accumulation_signals
        """
        if scan_date:
            stats_query += " WHERE signal_date = ?"
            stats_result = conn.execute(stats_query, [scan_date]).fetchone()
        elif symbol:
            stats_query += " WHERE underlying_symbol = ?"
            stats_result = conn.execute(stats_query, [symbol.upper()]).fetchone()
        else:
            stats_result = conn.execute(stats_query).fetchone()
        
        stats = {
            'total_signals': stats_result[0],
            'unique_symbols': stats_result[1],
            'total_premium': stats_result[2],
            'avg_confidence': round(stats_result[3], 1) if stats_result[3] else 0,
            'extreme_count': stats_result[4],
            'very_high_count': stats_result[5],
            'high_count': stats_result[6]
        }
        
        # Get distribution by signal type for chart
        if scan_date:
            type_distribution = conn.execute("""
                SELECT signal_type, COUNT(*) as count
                FROM accumulation_signals
                WHERE signal_date = ?
                GROUP BY signal_type
                ORDER BY count DESC
            """, [scan_date]).fetchall()
        elif symbol:
            type_distribution = conn.execute("""
                SELECT signal_type, COUNT(*) as count
                FROM accumulation_signals
                WHERE underlying_symbol = ?
                GROUP BY signal_type
                ORDER BY count DESC
            """, [symbol.upper()]).fetchall()
        else:
            type_distribution = []
        
        # Get distribution by strength for chart
        if scan_date:
            strength_distribution = conn.execute("""
                SELECT signal_strength, COUNT(*) as count
                FROM accumulation_signals
                WHERE signal_date = ?
                GROUP BY signal_strength
                ORDER BY count DESC
            """, [scan_date]).fetchall()
        elif symbol:
            strength_distribution = conn.execute("""
                SELECT signal_strength, COUNT(*) as count
                FROM accumulation_signals
                WHERE underlying_symbol = ?
                GROUP BY signal_strength
                ORDER BY count DESC
            """, [symbol.upper()]).fetchall()
        else:
            strength_distribution = []
        
        # Get top symbols by premium spent
        if scan_date:
            top_symbols = conn.execute("""
                SELECT underlying_symbol, 
                       COUNT(*) as signal_count,
                       SUM(premium_spent) as total_premium,
                       MAX(confidence_score) as max_confidence
                FROM accumulation_signals
                WHERE signal_date = ?
                GROUP BY underlying_symbol
                ORDER BY total_premium DESC
                LIMIT 10
            """, [scan_date]).fetchall()
        elif symbol:
            # For single symbol view, show breakdown by date instead
            top_symbols = conn.execute("""
                SELECT signal_date, 
                       COUNT(*) as signal_count,
                       SUM(premium_spent) as total_premium,
                       MAX(confidence_score) as max_confidence
                FROM accumulation_signals
                WHERE underlying_symbol = ?
                GROUP BY signal_date
                ORDER BY signal_date DESC
                LIMIT 10
            """, [symbol.upper()]).fetchall()
        else:
            top_symbols = []
        
        conn.close()
        
        return templates.TemplateResponse('options_signals.html', {
            'request': request,
            'signals': signals,
            'signal_types': signal_types,
            'signal_strengths': signal_strengths,
            'sectors': sectors,
            'available_dates': available_dates,
            'selected_signal_type': signal_type,
            'selected_signal_strength': signal_strength,
            'selected_sector': sector,
            'selected_symbol': symbol,
            'selected_min_confidence': min_confidence_val,
            'selected_min_premium': min_premium_val,
            'selected_asset_type': asset_type,
            'selected_scan_date': scan_date,
            'stats': stats,
            'type_distribution': [{'name': r[0], 'count': r[1]} for r in type_distribution],
            'strength_distribution': [{'name': r[0], 'count': r[1]} for r in strength_distribution],
            'top_symbols': [{'symbol': r[0], 'count': r[1], 'premium': r[2], 'confidence': r[3]} for r in top_symbols],
            'error': None
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse('options_signals.html', {
            'request': request,
            'error': f'Error loading options signals: {str(e)}',
            'signals': [],
            'signal_types': [],
            'signal_strengths': [],
            'sectors': [],
            'available_dates': [],
            'stats': {}
        })


@app.get("/ranked", response_class=HTMLResponse)
async def ranked_results(request: Request, date: Optional[str] = Query(None)):
    """Display AI-ranked stock analysis results."""
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    
    # Get today's date or use provided date
    today = datetime.now().strftime('%Y-%m-%d')
    current_date = date if date else today
    is_today = current_date == today
    
    # Parse current date
    date_obj = datetime.strptime(current_date, '%Y-%m-%d')
    
    # Calculate prev/next dates
    prev_date_obj = date_obj - timedelta(days=1)
    next_date_obj = date_obj + timedelta(days=1)
    
    prev_date = prev_date_obj.strftime('%Y-%m-%d')
    next_date = next_date_obj.strftime('%Y-%m-%d') if not is_today else None
    
    # Get ranked results for the date
    try:
        results_query = """
            SELECT 
                r.analysis_date,
                r.rank,
                r.symbol,
                r.scanner_name,
                r.composite_score,
                r.signal_strength,
                r.price,
                r.sector,
                COALESCE(ai.analysis_text, r.reasoning) as analysis_text,
                r.news_headline,
                r.news_sentiment_label,
                COALESCE(f.company_name, r.symbol) as company_name,
                f.market_cap,
                f.industry
            FROM scanner_data.ranked_analysis r
            LEFT JOIN scanner_data.fundamental_cache f ON r.symbol = f.symbol
            LEFT JOIN scanner_data.ai_analysis_individual ai ON r.symbol = ai.symbol AND r.analysis_date = ai.analysis_date
            WHERE r.analysis_date = ?
            ORDER BY r.rank ASC
        """
        results = conn.execute(results_query, [current_date]).fetchall()
        
        # Convert to list of dicts
        ranked_results = []
        for row in results:
            ranked_results.append({
                'analysis_date': row[0],
                'rank': row[1],
                'symbol': row[2],
                'scanner_name': row[3],
                'composite_score': round(row[4], 1),
                'signal_strength': round(row[5], 1),
                'price': row[6],
                'sector': row[7],
                'analysis_text': row[8],  # Full AI analysis from ai_analysis_individual table
                'news_headline': row[9],
                'news_sentiment_label': row[10],
                'company_name': row[11],
                'market_cap': format_market_cap(row[12]),
                'industry': row[13]
            })
    except Exception as e:
        print(f"Error loading ranked results: {e}")
        ranked_results = []
    
    conn.close()
    
    return templates.TemplateResponse('ranked.html', {
        'request': request,
        'results': ranked_results,
        'current_date': current_date,
        'prev_date': prev_date,
        'next_date': next_date,
        'is_today': is_today
    })


@app.get("/stats", response_class=HTMLResponse)
async def stats(request: Request):  # email: str = Depends(require_login)  # TODO: Re-enable after OAuth setup
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
            SELECT CAST(scan_date AS DATE) as date, COUNT(*) as count
            FROM scanner_data.scanner_results
            WHERE scan_date IS NOT NULL
            GROUP BY CAST(scan_date AS DATE)
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
    
    return templates.TemplateResponse('stats.html', {
        'request': request,
        'stats': stats_data
    })


@app.get("/scanner-performance", response_class=HTMLResponse)
async def scanner_performance(request: Request):
    """Display performance analytics for each scanner."""
    
    try:
        # Use a fresh connection for this request
        conn = duckdb.connect(DUCKDB_PATH, read_only=True)
        
        # Get all scanners
        scanners_query = """
            SELECT DISTINCT scanner_name
            FROM scanner_data.scanner_results
            ORDER BY scanner_name
        """
        scanners = [row[0] for row in conn.execute(scanners_query).fetchall()]
        conn.close()
        
        performance_data = []
        
        for scanner in scanners:
            # Reconnect for each scanner to avoid timeout
            conn = duckdb.connect(DUCKDB_PATH, read_only=True)
            
            # Get all picks for this scanner
            picks_query = """
                SELECT 
                    sr.symbol,
                    sr.scan_date,
                    sr.signal_strength
                FROM scanner_data.scanner_results sr
                WHERE sr.scanner_name = ?
                ORDER BY sr.scan_date, sr.symbol
            """
            picks = conn.execute(picks_query, [scanner]).fetchall()
            
            if not picks:
                conn.close()
                continue
            
            # Calculate performance for each pick
            pick_performances = []
            
            for symbol, scan_date, strength in picks:
                # Get entry price (closing price on pick date)
                entry_query = """
                    SELECT close
                    FROM scanner_data.daily_cache
                    WHERE symbol = ?
                    AND date = ?
                """
                entry_result = conn.execute(entry_query, [symbol, scan_date]).fetchone()
                
                if not entry_result:
                    continue
                
                entry_price = entry_result[0]
                
                # Get price history after pick date
                price_query = """
                    SELECT date, close, high, low
                    FROM scanner_data.daily_cache
                    WHERE symbol = ?
                    AND date > ?
                    ORDER BY date
                    LIMIT 60
                """
                prices = conn.execute(price_query, [symbol, scan_date]).fetchall()
                
                if not prices:
                    continue
                
                # Calculate metrics
                max_gain = 0
                max_drawdown = 0
                current_price = prices[-1][1]  # Last close
                
                peak_price = entry_price
                for date, close, high, low in prices:
                    # Max gain (from entry to peak)
                    gain_pct = ((high - entry_price) / entry_price) * 100
                    max_gain = max(max_gain, gain_pct)
                    
                    # Track peak for drawdown calculation
                    peak_price = max(peak_price, high)
                    
                    # Max drawdown (from peak to low)
                    if peak_price > entry_price:
                        drawdown_pct = ((low - peak_price) / peak_price) * 100
                        max_drawdown = min(max_drawdown, drawdown_pct)
                
                # Current P&L
                current_pnl = ((current_price - entry_price) / entry_price) * 100
                
                # Days held
                days_held = len(prices)
                
                pick_performances.append({
                    'symbol': symbol,
                    'scan_date': str(scan_date)[:10],
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'max_gain': max_gain,
                    'max_drawdown': max_drawdown,
                    'current_pnl': current_pnl,
                    'days_held': days_held,
                    'strength': strength
                })
            
            if not pick_performances:
                conn.close()
                continue
            
            # Calculate scanner summary stats
            total_picks = len(pick_performances)
            avg_max_gain = sum(p['max_gain'] for p in pick_performances) / total_picks
            avg_drawdown = sum(p['max_drawdown'] for p in pick_performances) / total_picks
            avg_current_pnl = sum(p['current_pnl'] for p in pick_performances) / total_picks
            
            # Best and worst picks
            best_pick = max(pick_performances, key=lambda x: x['max_gain'])
            worst_pick = min(pick_performances, key=lambda x: x['max_drawdown'])
            
            # Win rate (picks with positive current P&L)
            winners = [p for p in pick_performances if p['current_pnl'] > 0]
            win_rate = (len(winners) / total_picks) * 100 if total_picks > 0 else 0
            
            performance_data.append({
                'scanner_name': scanner,
                'total_picks': total_picks,
                'avg_max_gain': round(avg_max_gain, 2),
                'avg_drawdown': round(avg_drawdown, 2),
                'avg_current_pnl': round(avg_current_pnl, 2),
                'win_rate': round(win_rate, 1),
                'best_pick': best_pick,
                'worst_pick': worst_pick,
                'all_picks': sorted(pick_performances, key=lambda x: x['max_gain'], reverse=True)
            })
            
            # Close connection after processing each scanner
            conn.close()
        
        # Sort by avg max gain
        performance_data.sort(key=lambda x: x['avg_max_gain'], reverse=True)
        
    except Exception as e:
        print(f"Error calculating scanner performance: {e}")
        import traceback
        traceback.print_exc()
        performance_data = []
    
    return templates.TemplateResponse('scanner_performance.html', {
        'request': request,
        'performance_data': performance_data
    })


@app.get("/scanner-docs", response_class=HTMLResponse)
async def scanner_docs(request: Request):  # email: str = Depends(require_login)  # TODO: Re-enable after OAuth setup
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
        'candlestick_bullish': 'TA-Lib bullish reversal patterns - finds bottoms and new uptrends (20 patterns)',
        'candlestick_continuation': 'TA-Lib continuation patterns - pullbacks in existing uptrends (13 patterns)',
        'momentum_burst': 'Spots explosive momentum moves with high volume',
        'tight_consolidation': 'Detects tight consolidation patterns before potential breakouts'
    }
    
    # Custom display names for specific scanners
    custom_display_names = {
        'candlestick_bullish': 'Candlestick Bullish (Reversal)',
        'candlestick_continuation': 'Candlestick Bullish (Continuation)'
    }
    
    scanners = []
    for name, count in scanner_data:
        display_name = custom_display_names.get(name, name.replace('_', ' ').title())
        scanners.append({
            'name': name,
            'display_name': display_name,
            'short_desc': scanner_descriptions.get(name, 'Technical pattern scanner'),
            'count': count
        })
    
    return templates.TemplateResponse('scanner_docs.html', {
        'request': request,
        'scanners': scanners
    })


@app.get("/scanner-docs/{scanner_name}", response_class=HTMLResponse)
async def scanner_detail(request: Request, scanner_name: str):  # email: str = Depends(require_login)  # TODO: Re-enable after OAuth setup
    """Display detailed documentation for a specific scanner."""
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    
    # Get overall scanner stats
    stats = conn.execute("""
        SELECT 
            COUNT(*) as total,
            AVG(signal_strength) as avg_strength,
            COUNT(DISTINCT symbol) as unique_symbols,
            SUM(CASE WHEN signal_strength >= 90 THEN 1 ELSE 0 END) as excellent,
            SUM(CASE WHEN signal_strength >= 80 AND signal_strength < 90 THEN 1 ELSE 0 END) as very_good,
            SUM(CASE WHEN signal_strength >= 70 AND signal_strength < 80 THEN 1 ELSE 0 END) as good,
            SUM(CASE WHEN signal_strength >= 60 AND signal_strength < 70 THEN 1 ELSE 0 END) as fair,
            SUM(CASE WHEN signal_strength < 60 THEN 1 ELSE 0 END) as weak
        FROM scanner_data.scanner_results
        WHERE scanner_name = ?
    """, [scanner_name]).fetchone()
    
    # Get latest scan date
    latest_date = conn.execute("""
        SELECT CAST(MAX(scan_date) AS DATE)
        FROM scanner_data.scanner_results
        WHERE scanner_name = ?
    """, [scanner_name]).fetchone()
    
    # Get current performance (from latest scan date)
    current_perf = None
    if latest_date and latest_date[0]:
        current_perf = conn.execute("""
            SELECT 
                COUNT(*) as current_total,
                AVG(signal_strength) as current_avg_strength
            FROM scanner_data.scanner_results
            WHERE scanner_name = ? 
            AND CAST(scan_date AS DATE) = ?
        """, [scanner_name, str(latest_date[0])]).fetchone()
    
    # Get recent 30-day performance
    recent_perf = conn.execute("""
        SELECT 
            COUNT(*) as recent_total,
            AVG(signal_strength) as recent_avg_strength
        FROM scanner_data.scanner_results
        WHERE scanner_name = ? 
        AND CAST(scan_date AS DATE) >= CURRENT_DATE - INTERVAL '30 days'
    """, [scanner_name]).fetchone()
    
    conn.close()
    
    scanner_info = {
        'name': scanner_name,
        'display_name': scanner_name.replace('_', ' ').title(),
        'total_setups': stats[0] if stats else 0,
        'avg_strength': f"{stats[1]:.1f}" if stats and stats[1] else "N/A",
        'unique_symbols': stats[2] if stats else 0,
        'excellent': stats[3] if stats else 0,
        'very_good': stats[4] if stats else 0,
        'good': stats[5] if stats else 0,
        'fair': stats[6] if stats else 0,
        'weak': stats[7] if stats else 0,
        'current_total': current_perf[0] if current_perf else 0,
        'current_avg_strength': f"{current_perf[1]:.1f}" if current_perf and current_perf[1] else "N/A",
        'recent_total': recent_perf[0] if recent_perf else 0,
        'recent_avg_strength': f"{recent_perf[1]:.1f}" if recent_perf and recent_perf[1] else "N/A"
    }
    
    # Load scanner-specific content
    content = get_scanner_documentation(scanner_name)
    
    return templates.TemplateResponse('scanner_detail.html', {
        'request': request,
        'scanner_info': scanner_info,
        'content': content
    })


@app.get("/ticker-search", response_class=HTMLResponse)
async def ticker_search(request: Request, ticker: Optional[str] = Query(None)):
    """Search for all scanner results for a specific ticker."""
    if not ticker:
        ticker = ''
    ticker = ticker.strip().upper()
    
    if not ticker:
        return templates.TemplateResponse('ticker_search.html', {
            'request': request,
            'ticker': None,
            'results': None
        })
    
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
                # 'entry_price': f"${row[3]:.2f}" if row[3] else "N/A",
                'current_price': f"${current_price:.2f}" if current_price else "N/A",
                'gain_pct': f"{gain_pct:+.1f}%" if gain_pct is not None else "N/A",
                'signal_strength': f"{row[4]:.1f}" if row[4] else "N/A",
                'notes': row[5] if row[5] else ""
            })
        
        return templates.TemplateResponse('ticker_search.html', {
            'request': request,
            'ticker': ticker,
            'results': formatted_results,
            'result_count': len(formatted_results)
        })
    
    except Exception as e:
        print(f"Error searching ticker: {e}")
        return templates.TemplateResponse('ticker_search.html', {
            'request': request,
            'ticker': ticker,
            'results': None,
            'error': str(e)
        })
    finally:
        conn.close()


def get_scanner_documentation(scanner_name):
    """Return HTML documentation for specific scanner."""
    
    docs = {
        'accumulation_distribution': '''
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
''',
        'candlestick_bullish': '''
<h2>🎯 Overview</h2>
<p>Detects <strong>trend reversals from bearish to bullish</strong> using TA-Lib's proven candlestick pattern recognition algorithms. Each pattern is weighted based on historical reliability and combined with volume, trend, and technical context for comprehensive signal strength scoring (0-100).</p>

<div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 20px; margin: 20px 0; border-radius: 6px;">
    <h3 style="margin-top: 0;">💡 Key Insight</h3>
    <p>This scanner finds <strong>bottoms and catches new uptrends starting</strong>. No uptrend required - it identifies reversal signals after downtrends or in oversold conditions. Best for finding early entries before major moves.</p>
</div>

<h2>🕯️ Top Reversal Patterns</h2>

<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; margin: 20px 0;">
    <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px;">
        <span style="float: right; color: #667eea; font-size: 1.2em; font-weight: bold;">10.0</span>
        <div style="font-weight: bold; color: #333; margin-bottom: 5px;">Three White Soldiers</div>
        <div style="font-size: 0.9em; color: #666; margin-top: 8px;">Three consecutive bullish candles with higher highs and closes. Very strong reversal signal.</div>
    </div>
    
    <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px;">
        <span style="float: right; color: #667eea; font-size: 1.2em; font-weight: bold;">9.5</span>
        <div style="font-weight: bold; color: #333; margin-bottom: 5px;">Morning Star</div>
        <div style="font-size: 0.9em; color: #666; margin-top: 8px;">Three-candle pattern: bearish, small-bodied, then bullish. Classic bottom reversal.</div>
    </div>
    
    <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px;">
        <span style="float: right; color: #667eea; font-size: 1.2em; font-weight: bold;">9.0</span>
        <div style="font-weight: bold; color: #333; margin-bottom: 5px;">Bullish Engulfing</div>
        <div style="font-size: 0.9em; color: #666; margin-top: 8px;">Bullish candle completely engulfs prior bearish candle. Strong reversal confirmation.</div>
    </div>
    
    <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px;">
        <span style="float: right; color: #667eea; font-size: 1.2em; font-weight: bold;">8.7</span>
        <div style="font-weight: bold; color: #333; margin-bottom: 5px;">Piercing Pattern</div>
        <div style="font-size: 0.9em; color: #666; margin-top: 8px;">Bullish candle closes above midpoint of prior bearish candle.</div>
    </div>
    
    <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px;">
        <span style="float: right; color: #667eea; font-size: 1.2em; font-weight: bold;">8.5</span>
        <div style="font-weight: bold; color: #333; margin-bottom: 5px;">Hammer</div>
        <div style="font-size: 0.9em; color: #666; margin-top: 8px;">Long lower shadow, small body at top. Shows rejection of lower prices.</div>
    </div>
    
    <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px;">
        <span style="float: right; color: #667eea; font-size: 1.2em; font-weight: bold;">8.2</span>
        <div style="font-weight: bold; color: #333; margin-bottom: 5px;">Inverted Hammer</div>
        <div style="font-size: 0.9em; color: #666; margin-top: 8px;">Long upper shadow, small body at bottom. Tests resistance before reversal.</div>
    </div>
</div>

<p style="color: #666; font-style: italic;">...and 14 more patterns (Doji variations, Three Inside Up, Morning Doji Star, Abandoned Baby, etc.)</p>

<h2>📊 Signal Strength Calculation (0-100)</h2>

<table style="width: 100%; border-collapse: collapse; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <thead>
        <tr style="background: #667eea; color: white;">
            <th style="padding: 12px; text-align: left;">Component</th>
            <th style="padding: 12px; text-align: left;">Points</th>
            <th style="padding: 12px; text-align: left;">Calculation</th>
        </tr>
    </thead>
    <tbody>
        <tr style="border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 12px;"><strong>Base Pattern Weight</strong></td>
            <td style="padding: 12px;">0-70</td>
            <td style="padding: 12px;">Pattern Weight × 7 (5.0-10.0 → 35-70 pts)</td>
        </tr>
        <tr style="border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 12px;"><strong>Multiple Patterns</strong></td>
            <td style="padding: 12px;">0-15</td>
            <td style="padding: 12px;">4+ = 15, 3 = 10, 2 = 5 pts</td>
        </tr>
        <tr style="border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 12px;"><strong>Volume Confirmation</strong></td>
            <td style="padding: 12px;">0-10</td>
            <td style="padding: 12px;">RVOL: 2.0+ = 10, 1.5+ = 7, 1.2+ = 4 pts</td>
        </tr>
        <tr style="border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 12px;"><strong>Total Score</strong></td>
            <td style="padding: 12px;"><strong>0-100</strong></td>
            <td style="padding: 12px;">Sum of all components</td>
        </tr>
    </tbody>
</table>

<h2>✅ Entry Criteria</h2>
<ul>
    <li><strong>Pattern Detection:</strong> 1+ TA-Lib bullish reversal patterns confirmed</li>
    <li><strong>No Trend Required:</strong> Works in downtrends, sideways, or oversold conditions</li>
    <li><strong>Volume Preference:</strong> Higher RVOL (1.2x+) increases score</li>
    <li><strong>Multiple Patterns:</strong> Bonus when 2+ patterns confirm same signal</li>
</ul>

<h2>🎯 Usage Guidelines</h2>

<div style="background: #fff7ed; border-left: 4px solid #f59e0b; padding: 20px; margin: 20px 0; border-radius: 4px;">
    <h4 style="color: #f59e0b; margin-top: 0;">Best For:</h4>
    <ul>
        <li>Finding bottoms in oversold stocks</li>
        <li>Catching new uptrends starting</li>
        <li>Counter-trend trades (higher risk)</li>
        <li>Stocks breaking downtrends</li>
    </ul>
    <p style="margin-top: 15px;"><strong>Quality Threshold:</strong> 70+ recommended (pattern weight 8.5+)</p>
    <p><strong>Hold Period:</strong> 1-4 weeks typical</p>
    <p><strong>Risk Level:</strong> Medium-High (reversals can fail)</p>
</div>

<h2>⚠️ Risk Management</h2>
<ul>
    <li><strong>Stop Loss:</strong> Below pattern low (typically 3-5% below entry)</li>
    <li><strong>Position Size:</strong> 90+ = 3%, 80-89 = 2%, 70-79 = 1%</li>
    <li><strong>Time Stop:</strong> Exit if pattern doesn't work within 5-10 days</li>
</ul>

<h2>📝 Summary</h2>
<p><strong>20 TA-Lib reversal patterns</strong> detect trend changes from bearish to bullish. Best for finding early entries at bottoms before major moves. Higher risk than continuation patterns but offers better risk/reward at reversals.</p>
''',
        'candlestick_continuation': '''
<h2>🎯 Overview</h2>
<p>Detects <strong>trend continuation in existing uptrends</strong> using TA-Lib's proven candlestick pattern recognition. Identifies pullbacks and consolidations within strong trends for optimal re-entry points. Requires price above SMA20/50 for trend confirmation.</p>

<div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 20px; margin: 20px 0; border-radius: 6px;">
    <h3 style="margin-top: 0; color: #10b981;">💡 Key Insight</h3>
    <p>This scanner finds <strong>pullbacks in trending stocks</strong> for lower-risk entries. Requires existing uptrend confirmation, making it safer than reversal trading. Best for entering during temporary weakness in strong stocks.</p>
</div>

<h2>⬆️ Top Continuation Patterns</h2>

<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; margin: 20px 0;">
    <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px;">
        <span style="float: right; color: #10b981; font-size: 1.2em; font-weight: bold;">9.2</span>
        <div style="font-weight: bold; color: #333; margin-bottom: 5px;">Mat Hold</div>
        <div style="font-size: 0.9em; color: #666; margin-top: 8px;">Bullish candle, three small pullback candles, then bullish continuation. Classic continuation.</div>
    </div>
    
    <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px;">
        <span style="float: right; color: #10b981; font-size: 1.2em; font-weight: bold;">8.8</span>
        <div style="font-weight: bold; color: #333; margin-bottom: 5px;">Rising Three Methods</div>
        <div style="font-size: 0.9em; color: #666; margin-top: 8px;">Bullish candle, three small consolidation candles, bullish continuation. Very reliable.</div>
    </div>
    
    <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px;">
        <span style="float: right; color: #10b981; font-size: 1.2em; font-weight: bold;">8.0</span>
        <div style="font-weight: bold; color: #333; margin-bottom: 5px;">Separating Lines</div>
        <div style="font-size: 0.9em; color: #666; margin-top: 8px;">Bullish candle opens at same level as prior bearish candle. Trend resumes.</div>
    </div>
    
    <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px;">
        <span style="float: right; color: #10b981; font-size: 1.2em; font-weight: bold;">7.8</span>
        <div style="font-weight: bold; color: #333; margin-bottom: 5px;">Tasuki Gap</div>
        <div style="font-size: 0.9em; color: #666; margin-top: 8px;">Gap up, pullback doesn't fill gap, continuation. Gap acts as support.</div>
    </div>
    
    <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px;">
        <span style="float: right; color: #10b981; font-size: 1.2em; font-weight: bold;">7.4</span>
        <div style="font-weight: bold; color: #333; margin-bottom: 5px;">Stick Sandwich</div>
        <div style="font-size: 0.9em; color: #666; margin-top: 8px;">Two bullish candles sandwich a bearish candle. Support confirmed.</div>
    </div>
    
    <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px;">
        <span style="float: right; color: #10b981; font-size: 1.2em; font-weight: bold;">7.0</span>
        <div style="font-weight: bold; color: #333; margin-bottom: 5px;">Marubozu</div>
        <div style="font-size: 0.9em; color: #666; margin-top: 8px;">No shadows, all body. Shows strong conviction in direction.</div>
    </div>
</div>

<p style="color: #666; font-style: italic;">...and 7 more patterns (Belt Hold, Three Outside Up, Kicking, Gap Side-by-Side, etc.)</p>

<h2>📊 Signal Strength Calculation (0-100)</h2>

<table style="width: 100%; border-collapse: collapse; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <thead>
        <tr style="background: #10b981; color: white;">
            <th style="padding: 12px; text-align: left;">Component</th>
            <th style="padding: 12px; text-align: left;">Points</th>
            <th style="padding: 12px; text-align: left;">Calculation</th>
        </tr>
    </thead>
    <tbody>
        <tr style="border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 12px;"><strong>Base Pattern Weight</strong></td>
            <td style="padding: 12px;">0-70</td>
            <td style="padding: 12px;">Pattern Weight × 7 (5.0-9.2 → 35-64 pts)</td>
        </tr>
        <tr style="border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 12px;"><strong>Multiple Patterns</strong></td>
            <td style="padding: 12px;">0-15</td>
            <td style="padding: 12px;">4+ = 15, 3 = 10, 2 = 5 pts</td>
        </tr>
        <tr style="border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 12px;"><strong>Volume Confirmation</strong></td>
            <td style="padding: 12px;">0-10</td>
            <td style="padding: 12px;">RVOL: 2.0+ = 10, 1.5+ = 7, 1.2+ = 4 pts</td>
        </tr>
        <tr style="border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 12px;"><strong>Trend Strength Bonus</strong></td>
            <td style="padding: 12px;">0-5</td>
            <td style="padding: 12px;">Strong uptrend = 5, Moderate = 3, Weak = 1</td>
        </tr>
        <tr style="border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 12px;"><strong>Total Score</strong></td>
            <td style="padding: 12px;"><strong>0-100</strong></td>
            <td style="padding: 12px;">Sum of all components</td>
        </tr>
    </tbody>
</table>

<div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 20px 0; border-radius: 4px;">
    <strong>Key Difference:</strong> Continuation scanner requires existing uptrend (price above SMA20/50) and gets 0-5 point bonus for trend strength. Reversal scanner has no trend requirement.
</div>

<h2>✅ Entry Criteria</h2>
<ul>
    <li><strong>Pattern Detection:</strong> 1+ TA-Lib bullish continuation patterns confirmed</li>
    <li><strong>Trend Required:</strong> Price must be above SMA20 and/or SMA50</li>
    <li><strong>Trend Strength:</strong> Bonus points for strong uptrends (both SMAs rising)</li>
    <li><strong>Volume Preference:</strong> Higher RVOL (1.2x+) increases score</li>
</ul>

<h2>🎯 Usage Guidelines</h2>

<div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 20px; margin: 20px 0; border-radius: 4px;">
    <h4 style="color: #10b981; margin-top: 0;">Best For:</h4>
    <ul>
        <li>Entering pullbacks in uptrends</li>
        <li>Adding to winners on dips</li>
        <li>Trend-following trades (lower risk)</li>
        <li>Stocks with momentum</li>
    </ul>
    <p style="margin-top: 15px;"><strong>Quality Threshold:</strong> 65+ recommended (pattern weight 7.5+)</p>
    <p><strong>Hold Period:</strong> 1-3 weeks typical</p>
    <p><strong>Risk Level:</strong> Low-Medium (trend already confirmed)</p>
</div>

<h2>⚠️ Risk Management</h2>
<ul>
    <li><strong>Stop Loss:</strong> Below recent swing low (typically 2-4% below entry)</li>
    <li><strong>Position Size:</strong> 90+ = 3%, 80-89 = 2%, 65-79 = 1%</li>
    <li><strong>Volume Stop:</strong> Exit if volume dries up significantly</li>
</ul>

<h2>📝 Summary</h2>
<p><strong>13 TA-Lib continuation patterns</strong> confirm existing uptrends will continue. Lower risk than reversal trading since trend is already established. Best for entering pullbacks in strong stocks with momentum.</p>
'''
    }
    
    return docs.get(scanner_name, '<p>Documentation coming soon...</p>')


@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    # email: str = Depends(require_login),  # TODO: Re-enable after OAuth setup
    min_market_cap: Optional[str] = Query(None),
    sector: Optional[str] = Query(None),
    min_strength: Optional[str] = Query(None),
    scan_date: Optional[str] = Query(None),
    confirmed_only: Optional[str] = Query(None),
    ticker: Optional[str] = Query(None),
    pattern: Optional[str] = Query(None)
):
    sector_filter = sector or ''
    selected_ticker = ticker.strip().upper() if ticker else ''
    stocks = {}

    # Connect to DuckDB and get list of symbols
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    
    # For ticker-only searches, ignore date filter to show all historical results
    if selected_ticker and not pattern:
        selected_scan_date = ''
        print(f"INFO: Ticker-only search - clearing date filter to show all results")
    else:
        # Get latest scan date if none provided
        selected_scan_date = scan_date
        if not selected_scan_date:
            try:
                latest_date = conn.execute("""
                    SELECT CAST(MAX(scan_date) AS DATE)
                    FROM scanner_data.scanner_results
                    WHERE scan_date IS NOT NULL
                """).fetchone()
                if latest_date and latest_date[0]:
                    selected_scan_date = str(latest_date[0])
                    print(f"INFO: Auto-selected latest scan date: {selected_scan_date}")
            except Exception as e:
                print(f"Could not get latest scan date: {e}")
                selected_scan_date = ''
    
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
    
    # Don't auto-select a scanner - show all by default
    pattern = pattern if pattern and pattern != '' else False
    
    # Build query with filters
    symbols_query = '''
        SELECT DISTINCT d.symbol, 
               COALESCE(f.company_name, d.symbol) as company,
               f.market_cap,
               f.sector,
               f.industry
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
    
    # Don't populate stocks yet - wait until we know which symbols have scanner results
    # This prevents showing extra rows in the table
    all_symbols_query_result = conn.execute(symbols_query, params).fetchall()
    
    # Create a lookup dict for symbol metadata (don't add to stocks yet)
    symbol_metadata = {}
    if min_market_cap:
        for row in all_symbols_query_result:
            symbol, company, market_cap, sector, earnings_date, industry = row
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
                        symbol_metadata[symbol] = {
                            'company': company,
                            'market_cap': format_market_cap(market_cap),
                            'sector': sector,
                            'industry': industry
                        }
                except Exception:
                    pass
    else:
        for row in all_symbols_query_result:
            symbol, company, market_cap, sector, industry = row[:5]
            symbol_metadata[symbol] = {
                'company': company,
                'market_cap': format_market_cap(market_cap),
                'sector': sector,
                'industry': industry
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
                   news_url,
                   metadata
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
            # Use date range with CAST to ensure type compatibility
            date_obj = datetime.strptime(selected_scan_date, '%Y-%m-%d')
            next_day = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
            scanner_query += (
                ' AND scan_date >= CAST(? AS TIMESTAMP) '
                'AND scan_date < CAST(? AS TIMESTAMP)'
            )
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
                    'news_url': row[13] if len(row) > 13 else None,
                    'metadata': row[14] if len(row) > 14 else None
                } for row in scanner_results
            }
            print(f'Found {len(scanner_dict)} results for {pattern}')
        except Exception as e:
            print(f'Scanner query failed: {e}')
            scanner_dict = {}
        
        # Optimization: Fetch ALL volume data in one query instead of N queries
        # Use scanner_dict keys since those are the symbols with actual results
        symbols_list = list(scanner_dict.keys())
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
        
        # Fetch ALL scanners that identified each symbol (for the same date as current scanner)
        all_scanners_dict = {}
        if symbols_list and selected_scan_date:
            try:
                placeholders = ','.join(['?' for _ in symbols_list])
                all_scanners_query = f'''
                    SELECT symbol, scanner_name
                    FROM scanner_data.scanner_results
                    WHERE symbol IN ({placeholders})
                    AND scan_date >= CAST(? AS TIMESTAMP)
                    AND scan_date < CAST((CAST(? AS TIMESTAMP) + INTERVAL 1 DAY) AS TIMESTAMP)
                    ORDER BY symbol, scanner_name
                '''
                date_obj = datetime.strptime(selected_scan_date, '%Y-%m-%d')
                next_day = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
                all_scanner_results = conn.execute(all_scanners_query, symbols_list + [selected_scan_date, selected_scan_date]).fetchall()
                
                for row in all_scanner_results:
                    sym = row[0]
                    scanner = row[1]
                    if sym not in all_scanners_dict:
                        all_scanners_dict[sym] = []
                    all_scanners_dict[sym].append(scanner)
                
                print(f'Loaded all scanners for {len(all_scanners_dict)} symbols on {selected_scan_date}')
            except Exception as e:
                print(f'All scanners query failed: {e}')
                all_scanners_dict = {}
        
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
        
        # Fetch options signals for each symbol (from options_data database)
        options_signals_dict = {}
        if symbols_list and OPTIONS_DUCKDB_PATH:
            try:
                options_conn = get_options_db_connection()
                if options_conn:
                    placeholders = ','.join(['?' for _ in symbols_list])
                    options_query = f'''
                        SELECT underlying_symbol, signal_date, signal_type, 
                               signal_strength, confidence_score, strike, dte,
                               premium_spent, notes
                        FROM accumulation_signals
                        WHERE underlying_symbol IN ({placeholders})
                        ORDER BY underlying_symbol, signal_date DESC
                    '''
                    options_results = options_conn.execute(
                        options_query, symbols_list
                    ).fetchall()
                    
                    for row in options_results:
                        sym = row[0]
                        sig_date = str(row[1]) if row[1] else ''
                        sig_type = row[2]
                        sig_strength = row[3]
                        confidence = row[4]
                        strike = row[5]
                        dte = row[6]
                        premium = row[7]
                        notes = row[8]
                        
                        if sym not in options_signals_dict:
                            options_signals_dict[sym] = []
                        options_signals_dict[sym].append({
                            'date': sig_date,
                            'signal_type': sig_type,
                            'strength': sig_strength,
                            'confidence': confidence,
                            'strike': strike,
                            'dte': dte,
                            'premium': premium,
                            'notes': notes
                        })
                    
                    options_conn.close()
                    print(f'Loaded options signals for {len(options_signals_dict)} symbols')
            except Exception as e:
                print(f'Options signals query failed: {e}')
                options_signals_dict = {}
        
        # Fetch options walls data for each symbol (from options_data database)
        # Filter by selected scan date if one is chosen, otherwise get latest
        # Note: Scanner runs after market close, so walls should be from previous trading day
        options_walls_dict = {}
        if symbols_list and OPTIONS_DUCKDB_PATH:
            try:
                options_conn = get_options_db_connection()
                if options_conn:
                    placeholders = ','.join(['?' for _ in symbols_list])
                    
                    # If a specific scan date is selected, get walls from the previous trading day
                    # (scanner analyzes data from previous day's close)
                    if selected_scan_date:
                        walls_query = f'''
                            SELECT underlying_symbol, scan_date, stock_price,
                                   call_wall_strike, call_wall_oi,
                                   call_wall_2_strike, call_wall_2_oi,
                                   call_wall_3_strike, call_wall_3_oi,
                                   put_wall_strike, put_wall_oi,
                                   put_wall_2_strike, put_wall_2_oi,
                                   put_wall_3_strike, put_wall_3_oi,
                                   total_call_oi, total_put_oi, put_call_ratio
                            FROM options_walls
                            WHERE underlying_symbol IN ({placeholders})
                            AND CAST(scan_date AS DATE) < CAST(? AS DATE)
                            ORDER BY underlying_symbol, scan_date DESC
                        '''
                        walls_results = options_conn.execute(
                            walls_query, symbols_list + [selected_scan_date]
                        ).fetchall()
                    else:
                        # No date selected - get latest walls per symbol
                        walls_query = f'''
                            SELECT underlying_symbol, scan_date, stock_price,
                                   call_wall_strike, call_wall_oi,
                                   call_wall_2_strike, call_wall_2_oi,
                                   call_wall_3_strike, call_wall_3_oi,
                                   put_wall_strike, put_wall_oi,
                                   put_wall_2_strike, put_wall_2_oi,
                                   put_wall_3_strike, put_wall_3_oi,
                                   total_call_oi, total_put_oi, put_call_ratio
                            FROM options_walls
                            WHERE underlying_symbol IN ({placeholders})
                            ORDER BY underlying_symbol, scan_date DESC
                        '''
                        walls_results = options_conn.execute(
                            walls_query, symbols_list
                        ).fetchall()
                    
                    for row in walls_results:
                        sym = row[0]
                        if sym not in options_walls_dict:
                            stock_price = row[2] or 0
                            call_wall_1_strike = row[3] or 0
                            put_wall_1_strike = row[9] or 0
                            
                            # Calculate gamma flip price (midpoint between highest call and put walls)
                            if call_wall_1_strike and put_wall_1_strike:
                                gamma_flip = (call_wall_1_strike + put_wall_1_strike) / 2
                            else:
                                gamma_flip = None
                            
                            # Calculate distance to gamma flip
                            if gamma_flip and stock_price:
                                gamma_flip_distance = ((gamma_flip - stock_price) / stock_price) * 100
                            else:
                                gamma_flip_distance = None
                            
                            # Only keep the latest wall data per symbol
                            options_walls_dict[sym] = {
                                'scan_date': str(row[1]) if row[1] else '',
                                'stock_price': stock_price,
                                'call_wall_1': {'strike': row[3], 'oi': row[4]},
                                'call_wall_2': {'strike': row[5], 'oi': row[6]},
                                'call_wall_3': {'strike': row[7], 'oi': row[8]},
                                'put_wall_1': {'strike': row[9], 'oi': row[10]},
                                'put_wall_2': {'strike': row[11], 'oi': row[12]},
                                'put_wall_3': {'strike': row[13], 'oi': row[14]},
                                'total_call_oi': row[15],
                                'total_put_oi': row[16],
                                'put_call_ratio': row[17],
                                'gamma_flip': gamma_flip,
                                'gamma_flip_distance': gamma_flip_distance
                            }
                    
                    options_conn.close()
                    date_info = f" for date {selected_scan_date}" if selected_scan_date else " (latest)"
                    print(f'Loaded options walls for {len(options_walls_dict)} symbols{date_info}')
            except Exception as e:
                print(f'Options walls query failed: {e}')
                options_walls_dict = {}
        
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
                metadata = scanner_result.get('metadata')
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
                        # Initialize stock entry if not exists
                        if symbol not in stocks:
                            if symbol in symbol_metadata:
                                stocks[symbol] = symbol_metadata[symbol].copy()
                            else:
                                stocks[symbol] = {
                                    'company': symbol,
                                    'market_cap': None,
                                    'sector': None,
                                    'industry': None
                                }
                        
                        # Get pre-fetched volume data if available
                        if symbol in volume_data_dict:
                            vol_data = volume_data_dict[symbol]
                            latest_volume = vol_data['volume']
                            avg_volume_20 = vol_data['avg_volume_20']
                            volume_ratio = latest_volume / avg_volume_20 if avg_volume_20 > 0 else 0
                            
                            stocks[symbol][f'{pattern}_volume'] = latest_volume
                            stocks[symbol][f'{pattern}_avg_volume'] = avg_volume_20
                            stocks[symbol][f'{pattern}_volume_ratio'] = round(volume_ratio, 2)
                        else:
                            # No volume data available - use placeholders
                            stocks[symbol][f'{pattern}_volume'] = 0
                            stocks[symbol][f'{pattern}_avg_volume'] = 0
                            stocks[symbol][f'{pattern}_volume_ratio'] = 0
                        
                        stocks[symbol][pattern] = result
                        stocks[symbol][f'{pattern}_strength'] = strength
                        stocks[symbol][f'{pattern}_quality'] = quality
                        stocks[symbol][f'{pattern}_scan_date'] = scan_date
                        if entry_price is not None:
                            stocks[symbol][f'{pattern}_entry_price'] = entry_price
                        else:
                            stocks[symbol][f'{pattern}_entry_price'] = None
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
                        
                        # Add candlestick patterns metadata (parse JSON if string)
                        if metadata:
                            import json
                            import ast
                            try:
                                # Try to parse as JSON first
                                if isinstance(metadata, str):
                                    try:
                                        metadata_dict = json.loads(metadata)
                                    except json.JSONDecodeError:
                                        # If JSON fails, try Python literal eval
                                        metadata_dict = ast.literal_eval(metadata)
                                else:
                                    metadata_dict = metadata
                                
                                # Extract pattern names
                                all_patterns = (metadata_dict.get('all_patterns') or 
                                               metadata_dict.get('pattern_name') or
                                               metadata_dict.get('pattern') or '')
                                
                                # Get pattern weighting info
                                pattern_weight = metadata_dict.get('pattern_weight', 0)
                                
                                # Try different field names for the date
                                pattern_date = (metadata_dict.get('pattern_date') or 
                                              metadata_dict.get('date') or 
                                              metadata_dict.get('signal_date') or
                                              scan_date)
                                
                                # Format all patterns nicely with weights
                                if all_patterns:
                                    # Split by comma and format each pattern
                                    if ',' in all_patterns:
                                        pattern_list = [p.strip() for p in all_patterns.split(',')]
                                        readable_patterns = []
                                        for p in pattern_list:
                                            readable = p.replace('CDL', '').replace('_', ' ').title()
                                            # Add weight from pattern weights dictionary
                                            weight = PATTERN_WEIGHTS.get(p, 5.0)
                                            readable += f" ({weight})"
                                            readable_patterns.append(readable)
                                        pattern_display = ', '.join(readable_patterns)
                                    else:
                                        # Single pattern
                                        pattern_display = all_patterns.replace('CDL', '').replace('_', ' ').title()
                                        weight = PATTERN_WEIGHTS.get(all_patterns, 5.0)
                                        pattern_display += f" ({weight})"
                                    
                                    # Calculate days ago from pattern_date
                                    if pattern_date:
                                        pattern_dt = datetime.strptime(str(pattern_date), '%Y-%m-%d')
                                        today = datetime.now()
                                        days_ago = (today - pattern_dt).days
                                        
                                        if days_ago == 0:
                                            pattern_display += " - today"
                                        elif days_ago == 1:
                                            pattern_display += " - yesterday"
                                        else:
                                            pattern_display += f" - {days_ago}d ago"
                                    
                                    stocks[symbol][f'{pattern}_patterns'] = pattern_display
                            except Exception as e:
                                # If parsing fails, skip pattern display
                                pass
                        
                        # Add all scanners that identified this symbol on this date
                        if symbol in all_scanners_dict:
                            stocks[symbol]['picked_by_scanners'] = ','.join(all_scanners_dict[symbol])
                        else:
                            stocks[symbol]['picked_by_scanners'] = ''
                        
                        # Add scanner confirmations
                        if symbol in confirmations_dict:
                            stocks[symbol][f'{pattern}_confirmations'] = confirmations_dict[symbol]
                        else:
                            stocks[symbol][f'{pattern}_confirmations'] = []
                        
                        # Add options signals for this symbol
                        if symbol in options_signals_dict:
                            stocks[symbol][f'{pattern}_options_signals'] = options_signals_dict[symbol]
                        else:
                            stocks[symbol][f'{pattern}_options_signals'] = []
                        
                        # Add options walls for this symbol
                        if symbol in options_walls_dict:
                            stocks[symbol][f'{pattern}_options_walls'] = options_walls_dict[symbol]
                        else:
                            stocks[symbol][f'{pattern}_options_walls'] = None
                        
                        # Skip external API calls - too slow for Render
                        stocks[symbol][f'{pattern}_earnings_date'] = None
                        stocks[symbol][f'{pattern}_earnings_days'] = None
                        
                        # Skip sentiment API calls - too slow
                        stocks[symbol][f'{pattern}_sentiment_score'] = None
                        stocks[symbol][f'{pattern}_sentiment_label'] = None
                        stocks[symbol][f'{pattern}_sentiment_articles'] = None
                        # Don't add symbol if no volume data
                    except Exception as e:
                        print(f'failed on {symbol}: {e}')
                        # Don't add symbol to stocks if there was an error
                # Symbol doesn't meet minimum strength - skip it
            # Symbol not in scanner_dict - skip it
    
    print(f'INFO: After processing, stocks dict has {len(stocks)} symbols with pattern data')
    
    # Handle ticker-only search (no scanner selected)
    if selected_ticker and not pattern:
        print(f"DEBUG: Ticker search for {selected_ticker} without scanner selection")
        print(f"DEBUG: selected_ticker={selected_ticker}, pattern={pattern}")
        try:
            # Query all scanners that have this ticker
            ticker_query = '''
                SELECT scanner_name,
                       signal_type,
                       COALESCE(signal_strength, 75) as signal_strength,
                       COALESCE(setup_stage, 'N/A') as quality_placeholder,
                       scan_date,
                       metadata
                FROM scanner_data.scanner_results
                WHERE symbol = ?
            '''
            query_params = [selected_ticker]
            print(f"DEBUG: Running ticker query with params: {query_params}")
            
            # Ticker-only searches should show ALL historical results
            # Don't filter by date at all
            print(f"DEBUG: Showing all historical results (no date filter)")
            
            ticker_query += ' ORDER BY scan_date DESC, scanner_name'
            
            print(f"DEBUG: Executing query: {ticker_query}")
            ticker_results = conn.execute(ticker_query, query_params).fetchall()
            print(f"DEBUG: Query returned {len(ticker_results)} results")
            
            if ticker_results:
                # Initialize stock entry
                if selected_ticker in symbol_metadata:
                    stocks[selected_ticker] = symbol_metadata[selected_ticker].copy()
                else:
                    stocks[selected_ticker] = {
                        'company': selected_ticker,
                        'market_cap': None,
                        'sector': None,
                        'industry': None
                    }
                
                # Add each scanner as a separate "pattern"
                for row in ticker_results:
                    scanner_name = row[0]
                    signal_type = row[1]
                    strength = row[2]
                    quality = row[3]
                    scan_date = str(row[4])[:10] if row[4] else ''
                    metadata = row[5]
                    
                    # Parse metadata JSON
                    if metadata:
                        try:
                            import json
                            meta = json.loads(metadata) if isinstance(metadata, str) else metadata
                            picked_by = meta.get('picked_by_scanners')
                            setup_stage = meta.get('setup_stage')
                        except:
                            picked_by = None
                            setup_stage = quality
                    else:
                        picked_by = None
                        setup_stage = quality
                    
                    stocks[selected_ticker][scanner_name] = signal_type
                    stocks[selected_ticker][f'{scanner_name}_strength'] = strength
                    stocks[selected_ticker][f'{scanner_name}_quality'] = quality
                    stocks[selected_ticker][f'{scanner_name}_scan_date'] = scan_date
                    if picked_by:
                        stocks[selected_ticker][f'{scanner_name}_picked_count'] = picked_by
                    if setup_stage:
                        stocks[selected_ticker][f'{scanner_name}_setup_stage'] = setup_stage
                    print(f"DEBUG: Added {scanner_name}: {signal_type}, str={strength}")
                
                print(f"DEBUG: Stocks keys: {list(stocks.keys())}")
                print(f"DEBUG: {selected_ticker} keys: {list(stocks[selected_ticker].keys())}")
                print(f"INFO: Found {len(ticker_results)} scanner results for {selected_ticker}")
            else:
                print(f"INFO: No scanner results found for {selected_ticker}")
        except Exception as e:
            print(f"ERROR: Ticker search failed: {e}")
    
    # Apply confirmed_only filter
    if confirmed_only == 'yes' and pattern:
        filtered_stocks = {}
        for symbol, data in stocks.items():
            # Check if symbol was detected by other scanners on the SAME date
            # Use the 'picked_by_scanners' field which contains all scanners
            picked_by = data.get('picked_by_scanners', '')
            if picked_by:
                scanners_list = [s.strip() for s in picked_by.split(',')]
                # Filter to show only symbols picked by 2+ scanners
                if len(scanners_list) > 1:
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
                SELECT MAX(CAST(scan_date AS DATE))
                FROM scanner_data.scanner_results
            """).fetchone()
            date_to_use = str(latest_date_result[0]) if latest_date_result and latest_date_result[0] else None
            print(f"INFO: Using latest scan date: {date_to_use}")
        
        if date_to_use:
            scanner_counts_query = """
                SELECT scanner_name, COUNT(*) as count
                FROM scanner_data.scanner_results
                WHERE CAST(scan_date AS DATE) = ?
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
        # Filter out deprecated/invalid scanner names
        deprecated_scanners = ['bullish', 'Candlestick Bullish', 'Fundamental Swing']
        all_patterns = {}
        scanner_distribution = []  # For pie chart
        for row in scanner_counts:
            scanner_name = row[0]
            count = row[1]
            # Skip deprecated scanner names
            if scanner_name in deprecated_scanners:
                continue
            display_name = f"{scanner_name.replace('_', ' ').title()} ({count})"
            all_patterns[scanner_name] = display_name
            scanner_distribution.append({
                'name': scanner_name.replace('_', ' ').title(),
                'count': count
            })
        
        available_scanners = list(all_patterns.keys())
        print(f"INFO: Loaded {len(all_patterns)} scanner patterns")
    except Exception as e:
        print(f"ERROR: Could not load scanners from DB: {e}")
        import traceback
        traceback.print_exc()
        all_patterns = {}
        available_scanners = []
        scanner_distribution = []
    
    # Get available scan dates with setup counts
    available_scan_dates = []
    try:
        dates = conn.execute("""
            SELECT CAST(scan_date AS DATE) as date, COUNT(*) as count
            FROM scanner_data.scanner_results
            WHERE scan_date IS NOT NULL
            GROUP BY CAST(scan_date AS DATE)
            ORDER BY date DESC
        """).fetchall()
        available_scan_dates = [(str(row[0]), row[1]) for row in dates]
    except Exception as e:
        print(f"Could not load scan dates: {e}")
    
    # Get most recent scan timestamp for display
    last_updated = None
    try:
        latest_timestamp = conn.execute("""
            SELECT MAX(scan_date)
            FROM scanner_data.scanner_results
        """).fetchone()
        if latest_timestamp and latest_timestamp[0]:
            # Format as datetime string
            last_updated = str(latest_timestamp[0])
    except Exception as e:
        print(f"Could not get last update time: {e}")

    # Get historical scanner data for line chart (last 30 days)
    scanner_history = []
    try:
        history_query = """
            SELECT 
                CAST(scan_date AS DATE) as date,
                scanner_name,
                COUNT(*) as count
            FROM scanner_data.scanner_results
            WHERE scan_date >= CURRENT_DATE - INTERVAL '30 days'
                AND scanner_name NOT IN ('bullish', 'Candlestick Bullish', 'Fundamental Swing')
            GROUP BY CAST(scan_date AS DATE), scanner_name
            ORDER BY date ASC, scanner_name
        """
        history_results = conn.execute(history_query).fetchall()
        
        # Organize data by date
        from collections import defaultdict
        dates_dict = defaultdict(dict)
        all_scanner_names = set()
        
        for row in history_results:
            date_str = str(row[0])
            scanner_name = row[1]
            count = row[2]
            dates_dict[date_str][scanner_name] = count
            all_scanner_names.add(scanner_name)
        
        # Convert to format suitable for Chart.js
        dates = sorted(dates_dict.keys())
        scanner_history = {
            'dates': dates,
            'scanners': {}
        }
        
        for scanner_name in sorted(all_scanner_names):
            scanner_history['scanners'][scanner_name] = {
                'name': scanner_name.replace('_', ' ').title(),
                'data': [dates_dict[date].get(scanner_name, 0) for date in dates]
            }
        
        print(f"INFO: Loaded historical data for {len(dates)} dates and {len(all_scanner_names)} scanners")
    except Exception as e:
        print(f"ERROR: Could not load scanner history: {e}")
        import traceback
        traceback.print_exc()
        scanner_history = {'dates': [], 'scanners': {}}

    conn.close()

    return templates.TemplateResponse('index.html', {
        'request': request,
        'candlestick_patterns': all_patterns,
        'stocks': stocks,
        'pattern': pattern,
        'available_sectors': available_sectors,
        'available_scanners': available_scanners,
        'available_scan_dates': available_scan_dates,
        'available_tickers': available_tickers,
        'selected_scan_date': selected_scan_date,
        'selected_sector': sector_filter,
        'selected_market_cap': min_market_cap,
        'selected_min_strength': min_strength,
        'selected_ticker': selected_ticker,
        'confirmed_only': confirmed_only,
        'last_updated': last_updated,
        'scanner_distribution': scanner_distribution,
        'scanner_history': scanner_history,
        'scanner_colors': SCANNER_COLORS  # Pass hardcoded color mapping to template
    })


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
