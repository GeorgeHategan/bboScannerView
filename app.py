# BBO Scanner View - Updated Dec 13, 2025
import os
import sys
import json
import sqlite3
import asyncio
import logging
import threading
from functools import wraps
from pathlib import Path
from dotenv import load_dotenv
import duckdb

# Base directory for resolving relative paths (works on Render and locally)
BASE_DIR = Path(__file__).resolve().parent
from fastapi import FastAPI, Request, Query, Form, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from authlib.integrations.starlette_client import OAuth, OAuthError
from contextlib import asynccontextmanager

# ============================================================
# CONNECTION POOL & CACHING SYSTEM
# ============================================================

class ConnectionPool:
    """Thread-safe DuckDB/MotherDuck connection pool with health checking."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._connections = {}
                    cls._instance._conn_lock = threading.Lock()
        return cls._instance
    
    def _is_connection_healthy(self, conn) -> bool:
        """Check if a connection is still valid."""
        try:
            conn.execute("SELECT 1").fetchone()
            return True
        except:
            return False
    
    def _create_connection(self, db_path: str):
        """Create a new connection."""
        conn = duckdb.connect(db_path)
        print(f"[ConnectionPool] Created new connection for: {db_path[:50]}...")
        return conn
    
    def get_connection(self, db_path: str):
        """Get or create a healthy connection for the given database path."""
        with self._conn_lock:
            # Check if we have an existing connection
            if db_path in self._connections:
                conn = self._connections[db_path]
                # Verify it's still healthy
                if self._is_connection_healthy(conn):
                    return conn
                else:
                    # Connection is stale, close and remove it
                    print(f"[ConnectionPool] Stale connection detected, reconnecting: {db_path[:50]}...")
                    try:
                        conn.close()
                    except:
                        pass
                    del self._connections[db_path]
            
            # Create new connection
            try:
                self._connections[db_path] = self._create_connection(db_path)
            except Exception as e:
                print(f"[ConnectionPool] Error creating connection: {e}")
                raise
            return self._connections[db_path]
    
    def invalidate_connection(self, db_path: str):
        """Invalidate a specific connection (force reconnect on next use)."""
        with self._conn_lock:
            if db_path in self._connections:
                try:
                    self._connections[db_path].close()
                except:
                    pass
                del self._connections[db_path]
                print(f"[ConnectionPool] Invalidated connection: {db_path[:50]}...")
    
    def close_all(self):
        """Close all connections."""
        with self._conn_lock:
            for path, conn in self._connections.items():
                try:
                    conn.close()
                    print(f"[ConnectionPool] Closed connection: {path[:50]}...")
                except:
                    pass
            self._connections.clear()

# Global connection pool instance
_connection_pool = ConnectionPool()

def get_db_connection(db_path: str = None):
    """Get a database connection.
    
    For MotherDuck connections (which can be unstable on free tier),
    create a fresh connection each time to avoid 'connection closed' errors.
    For local DuckDB files, use the connection pool.
    """
    if db_path is None:
        db_path = DUCKDB_PATH
    
    # MotherDuck connections should NOT be pooled - they get throttled/closed
    if 'motherduck_token' in db_path or db_path.startswith('md:'):
        try:
            return duckdb.connect(db_path)
        except Exception as e:
            print(f"[DB] Error connecting to MotherDuck: {e}")
            raise
    
    # Local DuckDB files can be pooled
    return _connection_pool.get_connection(db_path)


class TTLCache:
    """Simple thread-safe TTL cache for query results with memory limits."""
    # Reduced maxsize for 512MB Render limit - each cached query can be large
    def __init__(self, maxsize: int = 30, max_memory_mb: int = 50):
        self._cache = {}
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        self._estimated_memory = 0
    
    def _estimate_size(self, value) -> int:
        """Rough estimate of object memory size."""
        import sys
        try:
            if isinstance(value, (list, tuple)):
                return sys.getsizeof(value) + sum(self._estimate_size(item) for item in value[:100])
            elif isinstance(value, dict):
                return sys.getsizeof(value) + sum(
                    self._estimate_size(k) + self._estimate_size(v) 
                    for k, v in list(value.items())[:100]
                )
            else:
                return sys.getsizeof(value)
        except:
            return 1000  # Default estimate
    
    def get(self, key: str):
        """Get value from cache if not expired."""
        with self._lock:
            if key in self._cache:
                value, expiry, size = self._cache[key]
                if datetime.now() < expiry:
                    return value
                else:
                    self._estimated_memory -= size
                    del self._cache[key]
        return None
    
    def _cleanup_expired(self):
        """Remove all expired entries."""
        now = datetime.now()
        expired_keys = [k for k, (_, exp, _) in self._cache.items() if now >= exp]
        for k in expired_keys:
            _, _, size = self._cache[k]
            self._estimated_memory -= size
            del self._cache[k]
    
    def set(self, key: str, value, ttl_seconds: int = 60):
        """Set value in cache with TTL (default 1 minute for memory conservation)."""
        with self._lock:
            # Estimate size of new value
            value_size = self._estimate_size(value)
            
            # Don't cache very large values (>10MB)
            if value_size > 10 * 1024 * 1024:
                print(f"[Cache] Skipping large value ({value_size/1024/1024:.1f}MB) for key: {key[:50]}")
                return
            
            # Cleanup expired entries first
            self._cleanup_expired()
            
            # If still over memory limit, remove oldest entries
            while (self._estimated_memory + value_size > self._max_memory_bytes or 
                   len(self._cache) >= self._maxsize) and self._cache:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                _, _, old_size = self._cache[oldest_key]
                self._estimated_memory -= old_size
                del self._cache[oldest_key]
            
            expiry = datetime.now() + timedelta(seconds=ttl_seconds)
            self._cache[key] = (value, expiry, value_size)
            self._estimated_memory += value_size
    
    def invalidate(self, key: str = None):
        """Invalidate a specific key or all keys."""
        with self._lock:
            if key:
                if key in self._cache:
                    _, _, size = self._cache[key]
                    self._estimated_memory -= size
                self._cache.pop(key, None)
            else:
                self._cache.clear()
                self._estimated_memory = 0
    
    def stats(self):
        """Get cache stats including memory usage."""
        with self._lock:
            self._cleanup_expired()
            return {
                "total": len(self._cache), 
                "valid": len(self._cache),
                "memory_mb": round(self._estimated_memory / 1024 / 1024, 2),
                "max_memory_mb": round(self._max_memory_bytes / 1024 / 1024, 2)
            }

# Global cache instance
_query_cache = TTLCache()

def cached(ttl_seconds: int = 120, key_prefix: str = ""):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{key_prefix}{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            # Check cache first
            cached_result = _query_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            _query_cache.set(cache_key, result, ttl_seconds)
            return result
        return wrapper
    return decorator

# ============================================================
# END CONNECTION POOL & CACHING SYSTEM
# ============================================================

# OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("WARNING: openai package not installed - AI analysis disabled")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("=" * 50)
logger.info("APPLICATION STARTING")
logger.info("=" * 50)

# Google Sheets integration
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
    logger.info("gspread imported successfully")
except ImportError as e:
    GSPREAD_AVAILABLE = False
    logger.warning(f"gspread not installed: {e}")

# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables loaded")

# Log environment check
logger.info(f"GOOGLE_CLIENT_ID set: {bool(os.environ.get('GOOGLE_CLIENT_ID'))}")
logger.info(f"GOOGLE_CLIENT_SECRET set: {bool(os.environ.get('GOOGLE_CLIENT_SECRET'))}")
logger.info(f"SECRET_KEY set: {bool(os.environ.get('SECRET_KEY'))}")
logger.info(f"motherduck_token set: {bool(os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN'))}")
logger.info(f"options_motherduck_token set: {bool(os.environ.get('options_motherduck_token') or os.environ.get('OPTIONS_MOTHERDUCK_TOKEN'))}")

# Initialize OAuth
oauth = OAuth()
logger.info("OAuth initialized")

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

# Background sync task flag
_sync_task = None

async def background_vix_sync():
    """Background task to sync VIX data to MotherDuck every hour."""
    while True:
        try:
            await asyncio.sleep(3600)  # Wait 1 hour
            # Import here to avoid circular reference
            from app import sync_sqlite_to_motherduck, get_vix_sqlite_count
            count = get_vix_sqlite_count()
            if count > 0:
                print(f"[Background Sync] Syncing {count} VIX records to MotherDuck...")
                result = sync_sqlite_to_motherduck()
                print(f"[Background Sync] Result: {result}")
            else:
                print("[Background Sync] No pending records to sync")
        except asyncio.CancelledError:
            print("[Background Sync] Task cancelled")
            break
        except Exception as e:
            print(f"[Background Sync] Error: {e}")

@asynccontextmanager
async def lifespan(app):
    """Handle app startup and shutdown."""
    global _sync_task
    logger.info("=== APPLICATION STARTING ===")
    print("=== APPLICATION STARTING ===")
    
    # Log all environment variables status
    logger.info(f"GOOGLE_CLIENT_ID configured: {bool(os.environ.get('GOOGLE_CLIENT_ID'))}")
    logger.info(f"SECRET_KEY configured: {bool(os.environ.get('SECRET_KEY'))}")
    logger.info(f"motherduck_token configured: {bool(os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN'))}")
    logger.info(f"options_motherduck_token configured: {bool(os.environ.get('options_motherduck_token') or os.environ.get('OPTIONS_MOTHERDUCK_TOKEN'))}")
    
    # Startup: start background sync task
    _sync_task = asyncio.create_task(background_vix_sync())
    logger.info("[Startup] Background VIX sync task started (every 1 hour)")
    print("[Startup] Background VIX sync task started (every 1 hour)")
    
    logger.info("=== APPLICATION READY ===")
    print("=== APPLICATION READY ===")
    yield
    # Shutdown: cancel background task and do final sync
    logger.info("=== APPLICATION SHUTTING DOWN ===")
    if _sync_task:
        _sync_task.cancel()
        try:
            await _sync_task
        except asyncio.CancelledError:
            pass
    # Final sync before shutdown
    logger.info("[Shutdown] Performing final VIX data sync...")
    print("[Shutdown] Performing final VIX data sync...")
    from app import sync_sqlite_to_motherduck
    result = sync_sqlite_to_motherduck()
    logger.info(f"[Shutdown] Final sync result: {result}")
    print(f"[Shutdown] Final sync result: {result}")
    
    # Close all database connections
    logger.info("[Shutdown] Closing database connections...")
    _connection_pool.close_all()
    logger.info("[Shutdown] All connections closed")

app = FastAPI(title="BBO Scanner View", description="Stock Scanner Dashboard", lifespan=lifespan)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        public_paths = ['/login', '/auth/google', '/auth/google/callback',
                        '/favicon.ico', '/static', '/webhook', '/api/health', '/api/memory']
        path = request.url.path

        # DISABLED: Visitor logging was creating a new thread per request
        # causing memory leaks on 512MB Render tier
        # To re-enable, uncomment and use a background task queue instead
        # 
        # # Log all visitors to non-static paths
        # if not any(path.startswith(p) for p in ['/favicon.ico', '/static']):
        #     session_data = request.scope.get('session')
        #     user_email = None
        #     if session_data:
        #         user = session_data.get('user')
        #         if user:
        #             user_email = user.get('email')
        #     # Use asyncio task instead of thread to avoid memory leak
        #     asyncio.create_task(async_log_visitor(request, path, user_email))

        # Continue with authentication check
        if not any(path.startswith(p) for p in public_paths):
            session_data = request.scope.get('session')
            if not session_data:
                return RedirectResponse('/login', status_code=302)

            user = session_data.get('user')
            if not user:
                return RedirectResponse('/login', status_code=302)

            email = user.get('email')
            allowed = get_allowed_emails()
            if email not in allowed:
                session_data.clear()
                return RedirectResponse('/login', status_code=302)

        return await call_next(request)


app.add_middleware(AuthMiddleware)
app.add_middleware(SessionMiddleware, secret_key=os.environ.get('SECRET_KEY', 'supersecret'))

# Mount static files directory
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Favicon route
@app.get("/favicon.ico")
async def favicon():
    return FileResponse(str(BASE_DIR / "static" / "favicon.png"), media_type="image/png")

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

# Admin email - only this user can access admin panel
ADMIN_EMAIL = 'hategan@gmail.com'

# User management database - Now using MotherDuck (scanner_data) instead of PostgreSQL
# This saves the cost of a separate Render PostgreSQL database

def get_user_db_connection():
    """Get connection to user management database (MotherDuck scanner_data)."""
    try:
        # Use the same MotherDuck connection as scanner data
        return get_db_connection(SCANNER_DATA_PATH)
    except Exception as e:
        logger.error(f"Error connecting to MotherDuck for user DB: {e}")
        return None


def init_user_db():
    """Initialize user management database (MotherDuck scanner_data)."""
    conn = get_user_db_connection()
    if not conn:
        logger.error("Could not initialize user database")
        return

    try:
        # DuckDB/MotherDuck syntax - tables should already exist from migration
        # Just ensure admin is in allowed users
        conn.execute("""
            CREATE TABLE IF NOT EXISTS main.allowed_users (
                id INTEGER PRIMARY KEY,
                email VARCHAR UNIQUE NOT NULL,
                added_by VARCHAR NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT true
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS main.login_logs (
                id INTEGER PRIMARY KEY,
                email VARCHAR NOT NULL,
                login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address VARCHAR,
                user_agent VARCHAR,
                success BOOLEAN DEFAULT true,
                failure_reason VARCHAR,
                country VARCHAR,
                country_code VARCHAR,
                city VARCHAR
            )
        """)

        # Ensure admin is always in allowed users
        existing = conn.execute(
            "SELECT email FROM main.allowed_users WHERE email = ?", [ADMIN_EMAIL]
        ).fetchone()
        if not existing:
            # Get next ID
            max_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM main.allowed_users").fetchone()[0]
            conn.execute(
                "INSERT INTO main.allowed_users (id, email, added_by, is_active) VALUES (?, ?, 'system', true)",
                [max_id + 1, ADMIN_EMAIL]
            )

        logger.info("User database initialized successfully (MotherDuck)")
    except Exception as e:
        logger.error(f"Error initializing user database: {e}")


# Initialize user database
init_user_db()


# Helper: get allowed emails from database
def get_allowed_emails():
    """Get list of allowed emails from MotherDuck database."""
    try:
        conn = get_user_db_connection()
        if not conn:
            return [ADMIN_EMAIL]

        # DuckDB syntax
        result = conn.execute(
            'SELECT email FROM main.allowed_users WHERE is_active = true'
        ).fetchall()
        
        emails = [row[0] for row in result]
        return emails if emails else [ADMIN_EMAIL]
    except Exception as e:
        logger.error(f"Error fetching allowed emails: {e}")
        return [ADMIN_EMAIL]

def get_ip_location(ip: str) -> dict:
    """Get country and city from IP address using ip-api.com."""
    try:
        # Skip localhost/private IPs
        if ip in ['127.0.0.1', 'localhost', 'unknown'] or ip.startswith('192.168.') or ip.startswith('10.'):
            return {'country': 'Local', 'countryCode': 'LO', 'city': 'localhost'}
        
        import urllib.request
        import json
        url = f"http://ip-api.com/json/{ip}?fields=status,country,countryCode,city"
        with urllib.request.urlopen(url, timeout=3) as response:
            data = json.loads(response.read().decode())
            if data.get('status') == 'success':
                return data
    except Exception as e:
        logger.warning(f"IP geolocation failed for {ip}: {e}")
    return {'country': 'Unknown', 'countryCode': '??', 'city': 'Unknown'}


def log_visitor(request: Request, page_path: str = None,
                email: str = None):
    """Log any visitor to the site, even if not logged in."""
    try:
        conn = get_user_db_connection()
        if not conn:
            return

        ip = request.client.host if request.client else 'unknown'
        user_agent = request.headers.get('user-agent', 'unknown')[:500]
        page = page_path or request.url.path

        # Get IP location for all visitors
        location = get_ip_location(ip)
        country = location.get('country')
        country_code = location.get('countryCode')
        city = location.get('city')

        # DuckDB/MotherDuck syntax - get next ID
        max_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM main.login_logs").fetchone()[0]
        conn.execute('''
            INSERT INTO main.login_logs (id, email, ip_address, user_agent,
                                   success, failure_reason, country,
                                   country_code, city)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [max_id + 1, email or 'visitor', ip, user_agent, False,
              f'page_visit:{page}', country, country_code, city])
    except Exception as e:
        logger.error(f"Error logging visitor: {e}")


def log_login_attempt(email: str, request: Request, success: bool,
                      failure_reason: str = None):
    """Log a login attempt with failure reason and IP geolocation."""
    try:
        conn = get_user_db_connection()
        if not conn:
            logger.warning("Could not log login - no database connection")
            return

        ip = request.client.host if request.client else 'unknown'
        user_agent = request.headers.get('user-agent', 'unknown')[:500]

        # Get IP location for failed attempts (to track intruders)
        country, country_code, city = None, None, None
        if not success:
            location = get_ip_location(ip)
            country = location.get('country')
            country_code = location.get('countryCode')
            city = location.get('city')

        # DuckDB/MotherDuck syntax - get next ID
        max_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM main.login_logs").fetchone()[0]
        conn.execute('''
            INSERT INTO main.login_logs (id, email, ip_address, user_agent,
                                   success, failure_reason, country,
                                   country_code, city)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [max_id + 1, email, ip, user_agent, success, failure_reason,
              country, country_code, city])
    except Exception as e:
        logger.error(f"Error logging login attempt: {e}")

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

# Cached helper function for login page chart data
def get_login_page_charts():
    """Fetch login page chart data with 5-minute caching."""
    cache_key = "login_page_charts"
    cached = _query_cache.get(cache_key)
    if cached is not None:
        return cached
    
    scanner_distribution = []
    scanner_history = {"dates": [], "scanners": {}, "unique_symbols": []}
    today_total = 0
    active_scanners = 0
    month_avg = 0
    
    try:
        conn = get_db_connection(DUCKDB_PATH)
        
        # Today's distribution - use scanner_results table with scan_date column
        today = datetime.now().strftime('%Y-%m-%d')
        dist_query = f"""
            SELECT scanner_name, COUNT(*) as count
            FROM scanner_results
            WHERE CAST(scan_date AS DATE) = '{today}'
            GROUP BY scanner_name
            ORDER BY count DESC
        """
        dist_result = conn.execute(dist_query).fetchall()
        scanner_distribution = [{"name": row[0], "count": row[1]} for row in dist_result]
        today_total = sum(r["count"] for r in scanner_distribution)
        active_scanners = len(scanner_distribution)
        
        # 30-day history
        history_query = """
            SELECT 
                CAST(scan_date AS DATE) as date,
                scanner_name,
                COUNT(*) as count
            FROM scanner_results
            WHERE CAST(scan_date AS DATE) >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY CAST(scan_date AS DATE), scanner_name
            ORDER BY date
        """
        history_result = conn.execute(history_query).fetchall()
        
        # Get unique symbols per day
        unique_symbols_query = """
            SELECT 
                CAST(scan_date AS DATE) as date,
                COUNT(DISTINCT symbol) as unique_symbols
            FROM scanner_results
            WHERE CAST(scan_date AS DATE) >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY CAST(scan_date AS DATE)
            ORDER BY date
        """
        unique_symbols_result = conn.execute(unique_symbols_query).fetchall()
        unique_symbols_by_date = {row[0].strftime('%m/%d'): row[1] for row in unique_symbols_result}
        
        # Process history data
        dates_set = set()
        scanner_data = {}
        for row in history_result:
            date_str = row[0].strftime('%m/%d')
            dates_set.add(date_str)
            scanner = row[1]
            if scanner not in scanner_data:
                scanner_data[scanner] = {}
            scanner_data[scanner][date_str] = row[2]
        
        dates = sorted(list(dates_set))
        scanner_history["dates"] = dates
        scanner_history["unique_symbols"] = [unique_symbols_by_date.get(d, 0) for d in dates]
        
        for scanner, data in scanner_data.items():
            scanner_key = scanner.lower().replace(' ', '_')
            scanner_history["scanners"][scanner_key] = {
                "name": scanner,
                "data": [data.get(d, 0) for d in dates]
            }
        
        # 30-day average
        if len(dates) > 0:
            total_30_days = sum(sum(s["data"]) for s in scanner_history["scanners"].values())
            month_avg = round(total_30_days / len(dates))
        
    except Exception as e:
        logger.error(f"Error fetching login page data: {e}")
    
    result = {
        "scanner_distribution": scanner_distribution,
        "scanner_history": scanner_history,
        "today_total": today_total,
        "active_scanners": active_scanners,
        "month_avg": month_avg
    }
    
    # Cache for 5 minutes
    _query_cache.set(cache_key, result, 300)
    return result


# ============================================================
# CACHED DATA HELPERS FOR INDEX PAGE
# ============================================================

def get_cached_latest_scan_date():
    """Get latest scan date with 1-minute cache."""
    cache_key = "latest_scan_date"
    cached = _query_cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        conn = get_db_connection(DUCKDB_PATH)
        latest_date = conn.execute("""
            SELECT CAST(MAX(scan_date) AS DATE)
            FROM scanner_results
            WHERE scan_date IS NOT NULL
        """).fetchone()
        result = str(latest_date[0]) if latest_date and latest_date[0] else ''
        _query_cache.set(cache_key, result, 60)  # 1 minute cache
        return result
    except Exception as e:
        print(f"Could not get latest scan date: {e}")
        return ''


def get_cached_ticker_list():
    """Get list of all tickers with 5-minute cache (reduced for memory)."""
    cache_key = "ticker_list"
    cached = _query_cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        conn = get_db_connection(DUCKDB_PATH)
        # Reduced from 10000 to 2000 to save memory on 512MB Render tier
        ticker_list = conn.execute("""
            SELECT DISTINCT symbol 
            FROM scanner_results
            ORDER BY symbol
            LIMIT 2000
        """).fetchall()
        result = [row[0] for row in ticker_list]
        _query_cache.set(cache_key, result, 300)  # 5 minute cache (reduced from 10)
        return result
    except Exception as e:
        print(f"Could not fetch ticker list: {e}")
        return []


def get_cached_symbol_metadata():
    """Get symbol metadata (company, market_cap, sector) with 10-minute cache (reduced for memory)."""
    cache_key = "symbol_metadata"
    cached = _query_cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        conn = get_db_connection(SCANNER_DATA_PATH)
        # Reduced from 10000 to 2000 to save memory on 512MB Render tier
        result = conn.execute('''
            SELECT DISTINCT d.symbol, 
                   COALESCE(f.company_name, d.symbol) as company,
                   f.market_cap,
                   f.sector,
                   f.industry
            FROM main.daily_cache d
            LEFT JOIN main.fundamental_cache f ON d.symbol = f.symbol
            ORDER BY d.symbol
            LIMIT 2000
        ''').fetchall()
        
        metadata = {}
        for row in result:
            symbol, company, market_cap, sector, industry = row[:5]
            metadata[symbol] = {
                'company': company,
                'market_cap': market_cap,
                'sector': sector,
                'industry': industry
            }
        
        _query_cache.set(cache_key, metadata, 600)  # 10 minute cache (reduced from 30)
        return metadata
    except Exception as e:
        print(f"Could not fetch symbol metadata: {e}")
        return {}


def get_alpha_vantage_asset_type(symbol: str):
    """Get asset type (ETF/Stock) from Alpha Vantage API with caching."""
    cache_key = f"av_asset_type:{symbol}"
    cached = _query_cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        import httpx
        api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            print("ALPHA_VANTAGE_API_KEY not configured")
            return None
        
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}"
        
        response = httpx.get(url, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            asset_type = data.get('AssetType', '')
            sector = data.get('Sector', '')
            industry = data.get('Industry', '')
            
            result = {
                'asset_type': asset_type,
                'sector': sector,
                'industry': industry,
                'is_etf': asset_type == 'ETF'
            }
            
            # Cache for 7 days (asset type doesn't change)
            _query_cache.set(cache_key, result, 604800)
            return result
        else:
            print(f"Alpha Vantage API error for {symbol}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching Alpha Vantage data for {symbol}: {e}")
        return None


def get_cached_scanner_results(pattern: str, scan_date: str = None, ticker: str = None):
    """Get scanner results with 2-minute cache."""
    cache_key = f"scanner_results:{pattern}:{scan_date}:{ticker}"
    cached = _query_cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        conn = get_db_connection(DUCKDB_PATH)
        
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
            FROM scanner_results
            WHERE scanner_name = ?
        '''
        query_params = [pattern]
        
        if ticker:
            scanner_query += ' AND symbol = ?'
            query_params.append(ticker)
        
        if scan_date:
            date_obj = datetime.strptime(scan_date, '%Y-%m-%d')
            next_day = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
            scanner_query += (
                ' AND scan_date >= CAST(? AS TIMESTAMP) '
                'AND scan_date < CAST(? AS TIMESTAMP)'
            )
            query_params.extend([scan_date, next_day])

        results = conn.execute(scanner_query, query_params).fetchall()
        _query_cache.set(cache_key, results, 120)  # 2 minute cache
        return results
    except Exception as e:
        print(f"Scanner query failed: {e}")
        return []


def get_cached_available_scanners(scan_date: str = None):
    """Get list of available scanners with 5-minute cache."""
    cache_key = f"available_scanners:{scan_date}"
    cached = _query_cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        conn = get_db_connection(DUCKDB_PATH)
        
        if scan_date:
            date_obj = datetime.strptime(scan_date, '%Y-%m-%d')
            next_day = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
            query = """
                SELECT DISTINCT scanner_name 
                FROM scanner_results
                WHERE scan_date >= CAST(? AS TIMESTAMP) 
                  AND scan_date < CAST(? AS TIMESTAMP)
                ORDER BY scanner_name
            """
            results = conn.execute(query, [scan_date, next_day]).fetchall()
        else:
            query = """
                SELECT DISTINCT scanner_name 
                FROM scanner_results
                ORDER BY scanner_name
            """
            results = conn.execute(query).fetchall()
        
        scanners = [row[0] for row in results]
        _query_cache.set(cache_key, scanners, 300)  # 5 minute cache
        return scanners
    except Exception as e:
        print(f"Could not fetch scanners: {e}")
        return []


def get_cached_historical_chart_data(num_days: int = 5):
    """Get historical chart data with 10-minute cache."""
    cache_key = f"historical_chart:{num_days}"
    cached = _query_cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        conn = get_db_connection(DUCKDB_PATH)
        
        history_query = f"""
            SELECT 
                CAST(scan_date AS DATE) as date,
                scanner_name,
                COUNT(*) as count
            FROM scanner_results
            WHERE CAST(scan_date AS DATE) >= CURRENT_DATE - INTERVAL '{num_days} days'
            GROUP BY CAST(scan_date AS DATE), scanner_name
            ORDER BY date
        """
        results = conn.execute(history_query).fetchall()
        
        # Process into structured data
        dates_set = set()
        scanner_data = {}
        for row in results:
            date_str = str(row[0])
            dates_set.add(date_str)
            scanner = row[1]
            if scanner not in scanner_data:
                scanner_data[scanner] = {}
            scanner_data[scanner][date_str] = row[2]
        
        dates = sorted(list(dates_set))
        result = {
            "dates": dates,
            "scanner_data": scanner_data,
            "scanners": list(scanner_data.keys())
        }
        
        _query_cache.set(cache_key, result, 600)  # 10 minute cache
        return result
    except Exception as e:
        print(f"Could not fetch historical data: {e}")
        return {"dates": [], "scanner_data": {}, "scanners": []}


# ============================================================
# END CACHED DATA HELPERS
# ============================================================

# Login page with charts
@app.get('/login', response_class=HTMLResponse)
async def login_form(request: Request):
    # Check if user is already logged in
    user = request.session.get('user')
    if user:
        email = user.get('email')
        if email in get_allowed_emails():
            return RedirectResponse('/', status_code=302)
    
    # Fetch cached chart data
    chart_data = get_login_page_charts()
    
    return templates.TemplateResponse('login.html', {
        'request': request, 
        'error': None,
        'scanner_distribution': chart_data['scanner_distribution'],
        'scanner_history': chart_data['scanner_history'],
        'today_total': chart_data['today_total'],
        'active_scanners': chart_data['active_scanners'],
        'month_avg': chart_data['month_avg']
    })

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
                'error': 'Failed to get user information from Google',
                'scanner_distribution': [],
                'scanner_history': {"dates": [], "scanners": {}, "unique_symbols": []},
                'today_total': 0,
                'active_scanners': 0,
                'month_avg': 0
            })
        
        email = user_info.get('email')
        allowed = get_allowed_emails()
        
        logger.info(f"Google OAuth callback - Email: {email}")
        logger.info(f"Allowed emails: {allowed}")
        
        if email not in allowed:
            log_login_attempt(email, request, success=False, failure_reason="Email not in allowed list")
            return templates.TemplateResponse('login.html', {
                'request': request,
                'error': f'Access denied. Email {email} is not authorized to access this application.',
                'scanner_distribution': [],
                'scanner_history': {"dates": [], "scanners": {}, "unique_symbols": []},
                'today_total': 0,
                'active_scanners': 0,
                'month_avg': 0
            })
        
        # Log successful login
        log_login_attempt(email, request, success=True)
        
        # Store user info in session
        request.session['user'] = {
            'email': email,
            'name': user_info.get('name'),
            'picture': user_info.get('picture')
        }
        
        logger.info(f"Login successful for {email}")
        return RedirectResponse('/', status_code=302)
        
    except OAuthError as e:
        logger.error(f"OAuth error: {e}")
        # Log the OAuth error as a failed attempt
        log_login_attempt("OAuth Error", request, success=False, failure_reason=str(e)[:200])
        return templates.TemplateResponse('login.html', {
            'request': request,
            'error': f'Authentication failed: {str(e)}',
            'scanner_distribution': [],
            'scanner_history': {"dates": [], "scanners": {}, "unique_symbols": []},
            'today_total': 0,
            'active_scanners': 0,
            'month_avg': 0
        })

# Logout
@app.get('/logout')
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse('/login', status_code=302)


# Admin check helper
def is_admin(request: Request) -> bool:
    """Check if the current user is an admin."""
    user = request.session.get('user')
    if not user:
        return False
    return user.get('email') == ADMIN_EMAIL


# Admin page - User management
@app.get('/admin', response_class=HTMLResponse)
async def admin_page(request: Request):
    """Admin page for managing users and viewing login logs."""
    user = request.session.get('user')
    if not user or user.get('email') != ADMIN_EMAIL:
        return RedirectResponse('/login', status_code=302)
    
    # Get all users (DuckDB/MotherDuck)
    conn = get_user_db_connection()
    if not conn:
        return HTMLResponse("Database connection error", status_code=500)

    users_result = conn.execute('''
        SELECT id, email, added_by, added_at, is_active
        FROM main.allowed_users
        ORDER BY added_at DESC
    ''').fetchall()
    users = [
        {
            'id': row[0],
            'email': row[1],
            'added_by': row[2],
            'added_at': row[3],
            'is_active': row[4]
        }
        for row in users_result
    ]

    # Get recent login logs
    logs_result = conn.execute('''
        SELECT email, login_time, ip_address, success, failure_reason,
               country, country_code, city
        FROM main.login_logs
        ORDER BY login_time DESC
        LIMIT 100
    ''').fetchall()
    logs = [
        {
            'email': row[0],
            'login_time': row[1],
            'ip_address': row[2],
            'success': row[3],
            'failure_reason': row[4],
            'country': row[5],
            'country_code': row[6],
            'city': row[7]
        }
        for row in logs_result
    ]

    # Get intruders (failed attempts grouped by IP) - DuckDB syntax
    intruders_result = conn.execute('''
        SELECT ip_address, country, country_code, city,
               COUNT(*) as attempts,
               STRING_AGG(DISTINCT email, ', ') as emails,
               MAX(login_time) as last_attempt
        FROM main.login_logs
        WHERE success = false
        GROUP BY ip_address, country, country_code, city
        ORDER BY attempts DESC, last_attempt DESC
        LIMIT 50
    ''').fetchall()

    intruders = [
        {
            'ip_address': row[0],
            'country': row[1] or 'Unknown',
            'country_code': row[2] or '??',
            'city': row[3] or 'Unknown',
            'attempts': row[4],
            'emails': row[5],
            'last_attempt': row[6]
        }
        for row in intruders_result
    ]

    return templates.TemplateResponse('admin.html', {
        'request': request,
        'user': user,
        'users': users,
        'logs': logs,
        'intruders': intruders,
        'admin_email': ADMIN_EMAIL
    })


# Admin API - Add user
@app.post('/admin/add-user')
async def admin_add_user(request: Request, email: str = Form(...)):
    """Add a new allowed user."""
    user = request.session.get('user')
    if not user or user.get('email') != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail='Admin access required')
    
    email = email.strip().lower()
    if not email or '@' not in email:
        raise HTTPException(status_code=400, detail='Invalid email address')

    try:
        conn = get_user_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail='Database error')

        # DuckDB/MotherDuck syntax - get next ID
        max_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM main.allowed_users").fetchone()[0]
        conn.execute('''
            INSERT INTO main.allowed_users (id, email, added_by, is_active)
            VALUES (?, ?, ?, true)
        ''', [max_id + 1, email, user.get('email')])
        
        logger.info(f"Admin added user: {email}")
    except Exception as e:
        error_str = str(e)
        if 'duplicate' in error_str.lower() or 'unique' in error_str.lower():
            raise HTTPException(status_code=400, detail='User already exists')
        raise HTTPException(status_code=500, detail=str(e))

    return RedirectResponse('/admin', status_code=302)


# Admin API - Remove user
@app.post('/admin/remove-user/{user_id}')
async def admin_remove_user(request: Request, user_id: int):
    """Remove/deactivate a user."""
    user = request.session.get('user')
    if not user or user.get('email') != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail='Admin access required')

    conn = get_user_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail='Database error')

    # Get the user email first (DuckDB syntax)
    result = conn.execute(
        'SELECT email FROM main.allowed_users WHERE id = ?', [user_id]
    ).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail='User not found')

    target_email = result[0]

    # Prevent removing admin
    if target_email == ADMIN_EMAIL:
        raise HTTPException(
            status_code=400, detail='Cannot remove admin user'
        )

    # Delete the user
    conn.execute('DELETE FROM main.allowed_users WHERE id = ?', [user_id])

    logger.info(f"Admin removed user: {target_email}")
    return RedirectResponse('/admin', status_code=302)


# Admin API - Toggle user active status
@app.post('/admin/toggle-user/{user_id}')
async def admin_toggle_user(request: Request, user_id: int):
    """Toggle a user's active status."""
    user = request.session.get('user')
    if not user or user.get('email') != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail='Admin access required')

    conn = get_user_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail='Database error')

    # Get the user email first (DuckDB syntax)
    result = conn.execute(
        'SELECT email, is_active FROM main.allowed_users WHERE id = ?',
        [user_id]
    ).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail='User not found')

    target_email, is_active = result

    # Prevent deactivating admin
    if target_email == ADMIN_EMAIL:
        raise HTTPException(
            status_code=400,
            detail='Cannot deactivate admin user'
        )

    # Toggle status
    new_status = not is_active
    conn.execute(
        'UPDATE main.allowed_users SET is_active = ? WHERE id = ?',
        [new_status, user_id]
    )

    logger.info(
        f"Admin toggled user {target_email} status to {new_status}"
    )
    return RedirectResponse('/admin', status_code=302)


# Admin API - Cache stats
@app.get('/admin/cache-stats')
async def admin_cache_stats(request: Request):
    """Get cache statistics (admin only)."""
    user = request.session.get('user')
    if not user or user.get('email') != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail='Admin access required')
    
    stats = _query_cache.stats()
    pool_connections = len(_connection_pool._connections) if hasattr(_connection_pool, '_connections') else 0
    
    return JSONResponse({
        'cache': stats,
        'connection_pool': {
            'active_connections': pool_connections
        }
    })


# Admin API - Clear cache
@app.post('/admin/clear-cache')
async def admin_clear_cache(request: Request):
    """Clear the query cache (admin only)."""
    user = request.session.get('user')
    if not user or user.get('email') != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail='Admin access required')
    
    _query_cache.invalidate()
    logger.info("Admin cleared query cache")
    
    return JSONResponse({'status': 'ok', 'message': 'Cache cleared'})


# Setup templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Database configuration
# For local development, use MotherDuck to access production data
# For production (Render), use environment variable

# Scanner results database (primary - for scanner picks)
motherduck_token = os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN', '')
print(f"DEBUG: motherduck_token found: {bool(motherduck_token)}")
if motherduck_token:
    # Use scanner_results database for scanner picks
    DUCKDB_PATH = f'md:scanner_results?motherduck_token={motherduck_token}'
    # Also connect to scanner_data for fundamental/cache data
    SCANNER_DATA_PATH = f'md:scanner_data?motherduck_token={motherduck_token}'
    print("INFO: Connecting to MotherDuck production database")
    print(f"INFO: Scanner results DB: md:scanner_results?motherduck_token=***")
    print(f"INFO: Scanner data DB: md:scanner_data?motherduck_token=***")
else:
    # Fallback to local DB if no MotherDuck token - this will fail on Render
    DUCKDB_PATH = '/Users/george/scannerPOC/breakoutScannersPOCs/scanner_data.main.duckdb'
    SCANNER_DATA_PATH = DUCKDB_PATH
    print("WARNING: No motherduck_token found, using local database")
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
    """Get a connection to the options signals database.
    Uses the connection pool to avoid configuration conflicts.
    """
    if OPTIONS_DUCKDB_PATH:
        return get_db_connection(OPTIONS_DUCKDB_PATH)
    return None


def get_options_db_connection_write():
    """Get a write-enabled connection to the options database.
    Alias for get_options_db_connection() - all connections support writes.
    """
    return get_options_db_connection()


# ============================================
# SQLite VIX Database (for webhook data)
# ============================================
VIX_SQLITE_PATH = os.path.join(os.path.dirname(__file__), 'vix_webhook.db')

def init_vix_sqlite():
    """Initialize the SQLite database for VIX webhook data."""
    conn = sqlite3.connect(VIX_SQLITE_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS vix_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            vix REAL,
            vx30 REAL,
            source TEXT DEFAULT 'webhook',
            notes TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print(f"VIX SQLite database initialized at {VIX_SQLITE_PATH}")

# Initialize on startup
init_vix_sqlite()

def insert_vix_sqlite(vix: float = None, vx30: float = None, source: str = "tradingview", notes: str = None):
    """Insert VIX data into local SQLite database."""
    try:
        conn = sqlite3.connect(VIX_SQLITE_PATH)
        cursor = conn.execute(
            "INSERT INTO vix_data (vix, vx30, source, notes) VALUES (?, ?, ?, ?)",
            (vix, vx30, source, notes)
        )
        inserted_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return {"status": "ok", "id": inserted_id, "vix": vix, "vx30": vx30}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_vix_sqlite_history(limit: int = 500):
    """Get VIX history from SQLite database."""
    try:
        conn = sqlite3.connect(VIX_SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM vix_data ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"Error reading VIX SQLite: {e}")
        return []

def get_vix_sqlite_count():
    """Get total count of VIX records in SQLite."""
    try:
        conn = sqlite3.connect(VIX_SQLITE_PATH)
        count = conn.execute("SELECT COUNT(*) FROM vix_data").fetchone()[0]
        conn.close()
        return count
    except:
        return 0


def sync_sqlite_to_motherduck():
    """Sync VIX data from local SQLite to MotherDuck, avoiding duplicates, then clear SQLite."""
    # Reduced from 10000 to 500 to prevent memory spikes during sync
    sqlite_data = get_vix_sqlite_history(limit=500)
    if not sqlite_data:
        return {"status": "ok", "message": "No data to sync", "synced": 0}
    
    conn = get_options_db_connection_write()
    if not conn:
        return {"status": "error", "message": "MotherDuck not configured"}
    
    synced = 0
    skipped = 0
    errors = 0
    
    try:
        for row in sqlite_data:
            try:
                # Check for duplicate by timestamp, vix, and vx30 values
                existing = conn.execute("""
                    SELECT COUNT(*) FROM vix_data 
                    WHERE timestamp = ? AND vix = ? AND vx30 = ?
                """, [row['timestamp'], row['vix'], row['vx30']]).fetchone()[0]
                
                if existing > 0:
                    skipped += 1
                    continue
                
                # Get next ID
                result = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM vix_data").fetchone()
                next_id = result[0]
                
                # Insert into MotherDuck
                conn.execute("""
                    INSERT INTO vix_data (id, timestamp, vix, vx30, source, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, [next_id, row['timestamp'], row['vix'], row['vx30'], row['source'], row['notes']])
                synced += 1
            except Exception as e:
                print(f"Error syncing row: {e}")
                errors += 1
        
        conn.close()
        
        # Clear SQLite after sync (even if some were skipped)
        if (synced > 0 or skipped > 0) and errors == 0:
            sqlite_conn = sqlite3.connect(VIX_SQLITE_PATH)
            sqlite_conn.execute("DELETE FROM vix_data")
            sqlite_conn.commit()
            sqlite_conn.close()
        
        return {"status": "ok", "synced": synced, "skipped": skipped, "errors": errors}
    except Exception as e:
        if conn:
            conn.close()
        return {"status": "error", "message": str(e)}


def insert_vix_data(vix: float = None, vx30: float = None, source: str = "webhook", notes: str = None):
    """
    Insert VIX and VX30 values into the vix_data table.
    
    Args:
        vix: Current VIX value
        vx30: Current VX30 (30-day VIX futures) value
        source: Data source (e.g., 'webhook', 'tradingview', 'manual')
        notes: Optional notes
    
    Returns:
        dict with status and inserted id
    """
    import time
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        conn = None
        try:
            conn = get_options_db_connection_write()
            if not conn:
                return {"status": "error", "message": "Options database not configured"}
            
            # Get next ID
            result = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM vix_data").fetchone()
            next_id = result[0]
            
            # Insert the data
            conn.execute("""
                INSERT INTO vix_data (id, timestamp, vix, vx30, source, notes)
                VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
            """, [next_id, vix, vx30, source, notes])
            
            conn.close()
            return {"status": "ok", "id": next_id, "vix": vix, "vx30": vx30}
        except Exception as e:
            if conn:
                try:
                    conn.close()
                except:
                    pass
            
            error_msg = str(e)
            # Retry on connection conflicts
            if "connection" in error_msg.lower() or "database" in error_msg.lower():
                if attempt < max_retries - 1:
                    print(f"Database connection error (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
            
            return {"status": "error", "message": error_msg}
    
    return {"status": "error", "message": "Max retries exceeded"}


# ============================================
# Focus List (MotherDuck only)
# ============================================

def init_focus_list_table():
    """Initialize the focus_list table in MotherDuck options_data database."""
    conn = get_options_db_connection_write()
    if not conn:
        print("WARNING: Cannot initialize focus_list - MotherDuck not configured")
        return False
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS focus_list (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                scanner_name VARCHAR,
                scan_date DATE,
                entry_price DOUBLE,
                strength INTEGER,
                quality VARCHAR,
                notes VARCHAR,
                ai_analysis VARCHAR,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Add ai_analysis column if it doesn't exist (for existing tables)
        try:
            conn.execute("ALTER TABLE focus_list ADD COLUMN ai_analysis VARCHAR")
        except:
            pass  # Column already exists
        conn.close()
        print("Focus list table initialized in MotherDuck")
        return True
    except Exception as e:
        print(f"Error initializing focus_list table: {e}")
        if conn:
            conn.close()
        return False

# Initialize focus list on module load
init_focus_list_table()


# ============================================
# OpenAI Analysis for Focus List
# ============================================

def get_openai_client():
    """Get OpenAI client if available."""
    if not OPENAI_AVAILABLE:
        return None
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("WARNING: OPENAI_API_KEY not set - AI analysis disabled")
        return None
    return OpenAI(api_key=api_key)


def get_stock_data_for_analysis(symbol: str) -> dict:
    """Fetch dark pool signals, options signals, price history, technicals, and options walls for AI analysis."""
    data = {
        'symbol': symbol, 
        'darkpool_signals': [], 
        'options_signals': [],
        'price_history': [],
        'options_walls': None,
        'technical_indicators': None
    }
    
    # Get options data (dark pool signals, options signals, and walls)
    conn = get_options_db_connection()
    if conn:
        try:
            # Get recent dark pool signals (last 20)
            dp_results = conn.execute("""
                SELECT signal_date, signal_type, direction, signal_strength,
                       dp_premium, block_count, buy_sell_ratio
                FROM darkpool_signals 
                WHERE ticker = ?
                ORDER BY signal_date DESC
                LIMIT 20
            """, [symbol]).fetchall()
            
            for row in dp_results:
                data['darkpool_signals'].append({
                    'date': str(row[0]),
                    'type': row[1],
                    'direction': row[2],
                    'strength': row[3],
                    'premium': row[4],
                    'blocks': row[5],
                    'buy_sell_ratio': row[6]
                })
            
            # Get recent options signals (last 20)
            opt_results = conn.execute("""
                SELECT signal_date, signal_type, direction, signal_strength,
                       strike, dte, premium_spent, confidence_score
                FROM accumulation_signals
                WHERE underlying_symbol = ?
                ORDER BY signal_date DESC
                LIMIT 20
            """, [symbol]).fetchall()
            
            for row in opt_results:
                data['options_signals'].append({
                    'date': str(row[0]),
                    'type': row[1],
                    'direction': row[2],
                    'strength': row[3],
                    'strike': row[4],
                    'dte': row[5],
                    'premium': row[6],
                    'confidence': row[7]
                })
            
            # Get latest options walls (call/put walls)
            walls_result = conn.execute("""
                SELECT scan_date, stock_price,
                       call_wall_strike, call_wall_oi,
                       call_wall_2_strike, call_wall_2_oi,
                       call_wall_3_strike, call_wall_3_oi,
                       put_wall_strike, put_wall_oi,
                       put_wall_2_strike, put_wall_2_oi,
                       put_wall_3_strike, put_wall_3_oi,
                       total_call_oi, total_put_oi, put_call_ratio
                FROM options_walls
                WHERE underlying_symbol = ?
                ORDER BY scan_date DESC
                LIMIT 1
            """, [symbol]).fetchone()
            
            if walls_result:
                stock_price = walls_result[1] or 0
                call_wall_1 = walls_result[2] or 0
                put_wall_1 = walls_result[8] or 0
                gamma_flip = (call_wall_1 + put_wall_1) / 2 if call_wall_1 and put_wall_1 else None
                
                data['options_walls'] = {
                    'scan_date': str(walls_result[0]),
                    'stock_price': stock_price,
                    'call_walls': [
                        {'strike': walls_result[2], 'oi': walls_result[3]},
                        {'strike': walls_result[4], 'oi': walls_result[5]},
                        {'strike': walls_result[6], 'oi': walls_result[7]}
                    ],
                    'put_walls': [
                        {'strike': walls_result[8], 'oi': walls_result[9]},
                        {'strike': walls_result[10], 'oi': walls_result[11]},
                        {'strike': walls_result[12], 'oi': walls_result[13]}
                    ],
                    'total_call_oi': walls_result[14],
                    'total_put_oi': walls_result[15],
                    'put_call_ratio': walls_result[16],
                    'gamma_flip': gamma_flip
                }
            
            conn.close()
        except Exception as e:
            print(f"Error fetching options data for AI analysis: {e}")
            if conn:
                conn.close()
    
    # Get price history and technical indicators from scanner_data database
    try:
        scanner_conn = get_db_connection(DUCKDB_PATH)
        price_results = scanner_conn.execute("""
            SELECT date, open, high, low, close, volume,
                   sma_20, sma_50, sma_200,
                   ema_9, ema_21,
                   atr_10, atr_14,
                   rsi_14
            FROM scanner_data.main.daily_cache
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT 20
        """, [symbol]).fetchall()
        
        for row in price_results:
            data['price_history'].append({
                'date': str(row[0]),
                'open': round(row[1], 2) if row[1] else None,
                'high': round(row[2], 2) if row[2] else None,
                'low': round(row[3], 2) if row[3] else None,
                'close': round(row[4], 2) if row[4] else None,
                'volume': row[5]
            })
        
        # Get the latest technical indicators (first row since ordered DESC)
        if price_results:
            latest = price_results[0]
            data['technical_indicators'] = {
                'sma_20': round(latest[6], 2) if latest[6] else None,
                'sma_50': round(latest[7], 2) if latest[7] else None,
                'sma_200': round(latest[8], 2) if latest[8] else None,
                'ema_9': round(latest[9], 2) if latest[9] else None,
                'ema_21': round(latest[10], 2) if latest[10] else None,
                'atr_10': round(latest[11], 2) if latest[11] else None,
                'atr_14': round(latest[12], 2) if latest[12] else None,
                'rsi_14': round(latest[13], 2) if latest[13] else None
            }
        
        # Reverse to get chronological order (oldest first)
        data['price_history'] = list(reversed(data['price_history']))
        
        scanner_conn.close()
    except Exception as e:
        print(f"Error fetching price history for AI analysis: {e}")
    
    return data


def analyze_stock_with_openai(symbol: str, scanner_name: str = None) -> str:
    """Call OpenAI to analyze stock based on price action, flow data, and options walls."""
    client = get_openai_client()
    if not client:
        return None
    
    # Get stock data
    data = get_stock_data_for_analysis(symbol)
    
    # Build price history summary
    price_summary = ""
    if data['price_history']:
        prices = data['price_history']
        latest = prices[-1] if prices else None
        oldest = prices[0] if prices else None
        
        if latest and oldest:
            price_change = ((latest['close'] - oldest['close']) / oldest['close'] * 100) if oldest['close'] else 0
            high_20d = max(p['high'] for p in prices if p['high'])
            low_20d = min(p['low'] for p in prices if p['low'])
            
            price_summary = f"""
Price Action (Last {len(prices)} days):
- Current Price: ${latest['close']}
- 20-Day Range: ${low_20d} - ${high_20d}
- 20-Day Change: {price_change:+.1f}%
- Recent OHLC: {json.dumps(prices[-5:], indent=2)}
"""
    
    # Build options walls summary
    walls_summary = ""
    if data['options_walls']:
        walls = data['options_walls']
        call_walls = [w for w in walls['call_walls'] if w['strike']]
        put_walls = [w for w in walls['put_walls'] if w['strike']]
        
        gamma_flip_str = f"${walls['gamma_flip']:.2f}" if walls['gamma_flip'] else 'N/A'
        pcr_str = f"{walls['put_call_ratio']:.2f}" if walls['put_call_ratio'] else 'N/A'
        
        walls_summary = f"""
Options Walls (Key Levels):
- Stock Price: ${walls['stock_price']:.2f}
- Gamma Flip (resistance/support pivot): {gamma_flip_str}
- Call Walls (resistance): {', '.join([f"${w['strike']:.0f} ({w['oi']:,} OI)" for w in call_walls[:3]])}
- Put Walls (support): {', '.join([f"${w['strike']:.0f} ({w['oi']:,} OI)" for w in put_walls[:3]])}
- Put/Call Ratio: {pcr_str}
"""
    
    # Build dark pool summary
    dp_summary = ""
    if data['darkpool_signals']:
        bullish = sum(1 for s in data['darkpool_signals'] 
                     if s['direction'] in ('BUY', 'BULLISH'))
        bearish = sum(1 for s in data['darkpool_signals'] 
                     if s['direction'] in ('SELL', 'BEARISH'))
        dp_summary = f"""
Dark Pool Signals ({len(data['darkpool_signals'])} recent):
- Bullish signals: {bullish}
- Bearish signals: {bearish}
- Recent activity: {json.dumps(data['darkpool_signals'][:5], indent=2)}
"""
    
    # Build options flow summary
    opt_summary = ""
    if data['options_signals']:
        bullish = sum(1 for s in data['options_signals'] 
                     if s['direction'] in ('BULLISH', 'BUY'))
        bearish = sum(1 for s in data['options_signals'] 
                     if s['direction'] in ('BEARISH', 'SELL'))
        opt_summary = f"""
Options Flow Signals ({len(data['options_signals'])} recent):
- Bullish signals: {bullish}
- Bearish signals: {bearish}
- Recent activity: {json.dumps(data['options_signals'][:5], indent=2)}
"""
    
    # Build technical indicators summary
    tech_summary = ""
    if data['technical_indicators']:
        ti = data['technical_indicators']
        current_price = data['price_history'][-1]['close'] if data['price_history'] else 0
        
        # Determine trend based on SMAs
        trend_signals = []
        if ti['sma_20'] and current_price:
            trend_signals.append(f"{'Above' if current_price > ti['sma_20'] else 'Below'} SMA20 (${ti['sma_20']})")
        if ti['sma_50'] and current_price:
            trend_signals.append(f"{'Above' if current_price > ti['sma_50'] else 'Below'} SMA50 (${ti['sma_50']})")
        if ti['sma_200'] and current_price:
            trend_signals.append(f"{'Above' if current_price > ti['sma_200'] else 'Below'} SMA200 (${ti['sma_200']})")
        
        rsi_status = "Overbought" if ti['rsi_14'] and ti['rsi_14'] > 70 else "Oversold" if ti['rsi_14'] and ti['rsi_14'] < 30 else "Neutral"
        
        tech_summary = f"""
Technical Indicators:
- RSI(14): {ti['rsi_14']} ({rsi_status})
- SMA Trend: {', '.join(trend_signals)}
- EMA(9): ${ti['ema_9']}, EMA(21): ${ti['ema_21']}
- ATR(14): ${ti['atr_14']} (volatility measure)
"""
    
    if not dp_summary and not opt_summary and not price_summary:
        return "No data available for analysis."
    
    prompt = f"""Analyze {symbol} for a swing trader based on the following data:

Scanner: {scanner_name or 'N/A'}
{price_summary}
{tech_summary}
{walls_summary}
{dp_summary}
{opt_summary}

Provide analysis in exactly this format with 4 numbered points:

1. **Institutional Sentiment**: [BULLISH/BEARISH/NEUTRAL] - Brief explanation of dark pool and options flow signals.

2. **Technical Setup**: Current trend status, RSI reading, and position relative to key moving averages.

3. **Key Levels**: Important support levels (from put walls) and resistance levels (from call walls).

4. **Trade Plan**: Entry zone: $X-$Y | Stop Loss: $Z (below support) | Take Profit: $W (at resistance) | Risk/Reward ratio.

Be specific with price levels. Use the ATR for stop loss sizing."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional swing trader and stock analyst. Always provide specific price levels for entries, stops, and targets. Be concise but actionable."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return f"AI analysis failed: {str(e)}"


def add_to_focus_list(symbol: str, scanner_name: str = None, scan_date: str = None, 
                      entry_price: float = None, strength: int = None, quality: str = None, 
                      notes: str = None):
    """Add a setup to the focus list with AI analysis."""
    conn = get_options_db_connection_write()
    if not conn:
        return {"status": "error", "message": "Database not available"}
    
    try:
        # Get AI analysis for the stock
        ai_analysis = analyze_stock_with_openai(symbol, scanner_name)
        
        # Get next ID
        result = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM focus_list").fetchone()
        next_id = result[0]
        
        conn.execute("""
            INSERT INTO focus_list 
            (id, symbol, scanner_name, scan_date, entry_price, strength, quality, notes, ai_analysis)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [next_id, symbol, scanner_name, scan_date, entry_price, strength, quality, notes, ai_analysis])
        conn.close()
        return {"status": "ok", "id": next_id, "symbol": symbol, "ai_analysis": ai_analysis}
    except Exception as e:
        if conn:
            conn.close()
        return {"status": "error", "message": str(e)}


def remove_from_focus_list(item_id: int):
    """Remove an item from the focus list by ID."""
    conn = get_options_db_connection_write()
    if not conn:
        return {"status": "error", "message": "Database not available"}
    
    try:
        conn.execute("DELETE FROM focus_list WHERE id = ?", [item_id])
        conn.close()
        return {"status": "ok", "deleted_id": item_id}
    except Exception as e:
        if conn:
            conn.close()
        return {"status": "error", "message": str(e)}


def get_focus_list():
    """Get all items from focus list, sorted by added_date DESC (newest first)."""
    conn = get_options_db_connection()
    if not conn:
        return []
    
    try:
        result = conn.execute("""
            SELECT id, symbol, scanner_name, scan_date, entry_price, 
                   strength, quality, notes, ai_analysis, added_date
            FROM focus_list 
            ORDER BY added_date DESC
        """).fetchall()
        conn.close()
        
        items = []
        for row in result:
            items.append({
                'id': row[0],
                'symbol': row[1],
                'scanner_name': row[2],
                'scan_date': str(row[3]) if row[3] else None,
                'entry_price': row[4],
                'strength': row[5],
                'quality': row[6],
                'notes': row[7],
                'ai_analysis': row[8],
                'added_date': str(row[9]) if row[9] else None
            })
        return items
    except Exception as e:
        print(f"Error getting focus list: {e}")
        if conn:
            conn.close()
        return []


def get_latest_vix_data():
    """Get the most recent VIX and VX30 values."""
    conn = get_options_db_connection()
    if not conn:
        return None
    
    try:
        result = conn.execute("""
            SELECT timestamp, vix, vx30, source 
            FROM vix_data 
            ORDER BY timestamp DESC 
            LIMIT 1
        """).fetchone()
        conn.close()
        
        if result:
            return {
                "timestamp": result[0],
                "vix": result[1],
                "vx30": result[2],
                "source": result[3]
            }
        return None
    except Exception as e:
        conn.close()
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



@app.get("/health", include_in_schema=False)
async def health_check():
    """Simple health check endpoint - no DB, no auth."""
    return JSONResponse({"status": "ok"})


@app.get("/debug", include_in_schema=False)
async def debug_info():
    """Debug endpoint to check app configuration."""
    debug_data = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "GOOGLE_CLIENT_ID": bool(os.environ.get('GOOGLE_CLIENT_ID')),
            "GOOGLE_CLIENT_SECRET": bool(os.environ.get('GOOGLE_CLIENT_SECRET')),
            "SECRET_KEY": bool(os.environ.get('SECRET_KEY')),
            "motherduck_token": bool(os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN')),
            "options_motherduck_token": bool(os.environ.get('options_motherduck_token') or os.environ.get('OPTIONS_MOTHERDUCK_TOKEN')),
            "ALLOWED_EMAILS": bool(os.environ.get('ALLOWED_EMAILS')),
        },
        "database": {
            "DUCKDB_PATH_set": bool(DUCKDB_PATH),
            "OPTIONS_DUCKDB_PATH_set": bool(OPTIONS_DUCKDB_PATH),
        },
        "sqlite": {
            "vix_db_path": VIX_SQLITE_PATH,
            "vix_db_exists": os.path.exists(VIX_SQLITE_PATH) if VIX_SQLITE_PATH else False,
        }
    }
    
    # Test MotherDuck connection (scanner_data)
    try:
        if DUCKDB_PATH:
            conn = get_db_connection(DUCKDB_PATH)
            if conn:
                debug_data["motherduck_scanner"] = "connected"
                conn.close()
            else:
                debug_data["motherduck_scanner"] = "no connection"
        else:
            debug_data["motherduck_scanner"] = "DUCKDB_PATH not configured"
    except Exception as e:
        debug_data["motherduck_scanner"] = f"error: {str(e)}"
    
    # Test Options DB connection
    try:
        conn = get_options_db_connection()
        if conn:
            debug_data["motherduck_options"] = "connected"
            conn.close()
        else:
            debug_data["motherduck_options"] = "no connection"
    except Exception as e:
        debug_data["motherduck_options"] = f"error: {str(e)}"
    
    return JSONResponse(debug_data)


# ============================================================================
# TradingView Webhook to Google Sheets Integration
# ============================================================================

# TODO: Enable Google Sheets integration when ready
# def get_google_sheets_client():
#     """
#     Get authenticated Google Sheets client using service account credentials.
#     
#     Requires environment variable GOOGLE_SHEETS_CREDENTIALS containing the
#     service account JSON credentials (as a JSON string).
#     
#     Also requires GOOGLE_SHEET_ID - the ID of the spreadsheet to write to.
#     """
#     if not GSPREAD_AVAILABLE:
#         return None
#     
#     creds_json = os.environ.get('GOOGLE_SHEETS_CREDENTIALS')
#     if not creds_json:
#         print("WARNING: GOOGLE_SHEETS_CREDENTIALS not set")
#         return None
#     
#     try:
#         import json
#         creds_dict = json.loads(creds_json)
#         
#         scopes = [
#             'https://www.googleapis.com/auth/spreadsheets',
#             'https://www.googleapis.com/auth/drive'
#         ]
#         
#         credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
#         client = gspread.authorize(credentials)
#         return client
#     except Exception as e:
#         print(f"ERROR: Failed to authenticate with Google Sheets: {e}")
#         return None


@app.post("/webhook/tradingview")
async def tradingview_webhook(request: Request):
    """
    Receive webhooks from TradingView and insert data into Google Sheets or database.
    
    TradingView Alert Message Format (JSON):
    {
        "symbol": "{{ticker}}",
        "price": "{{close}}",
        "time": "{{time}}",
        "indicator_name": "YOUR_INDICATOR_NAME",
        "indicator_value": "{{plot_0}}",
        "signal": "BUY/SELL/NEUTRAL",
        "notes": "Optional notes"
    }
    
    For VIX data specifically, use:
    {
        "type": "vix",
        "vix": "{{close}}",       // or {{plot_0}} for indicator value
        "vx30": "{{plot_1}}",     // optional VX30 value
        "notes": "TradingView alert"
    }
    
    You can customize the JSON message in TradingView alert settings.
    Use {{ticker}}, {{close}}, {{time}}, {{plot_0}}, etc. placeholders.
    """
    try:
        # Get the webhook payload - log everything for debugging
        content_type = request.headers.get('content-type', '')
        body = await request.body()
        body_str = body.decode('utf-8').strip()
        
        print(f"TradingView webhook - Content-Type: {content_type}")
        print(f"TradingView webhook - Raw body: {body_str}")
        
        # Try to parse as JSON
        import json
        data = None
        
        try:
            data = json.loads(body_str)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            # Try to fix common issues - TradingView sometimes sends with extra quotes
            try:
                # Remove outer quotes if present
                if body_str.startswith('"') and body_str.endswith('"'):
                    body_str = body_str[1:-1].replace('\\"', '"')
                    data = json.loads(body_str)
            except:
                pass
        
        if data is None:
            # If still not JSON, create a simple dict with the raw message
            data = {"raw_message": body_str}
        
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        
        # Log the received data
        print(f"TradingView webhook parsed: {data}")
        
        # Check if this is a VIX data webhook
        if data.get('type') == 'vix' or data.get('indicator_name', '').lower() in ['vix', 'vx30', 'volatility']:
            # Extract VIX values - handle both numeric and string values
            vix_val = data.get('vix') or data.get('indicator_value')
            vx30_val = data.get('vx30')
            
            # Convert to float if strings
            try:
                vix = float(vix_val) if vix_val else None
            except (ValueError, TypeError):
                vix = None
                
            try:
                vx30 = float(vx30_val) if vx30_val else None
            except (ValueError, TypeError):
                vx30 = None
            
            # Insert into LOCAL SQLite database (avoids MotherDuck connection issues)
            result = insert_vix_sqlite(
                vix=vix,
                vx30=vx30,
                source="webhook_update",
                notes=data.get('notes', data.get('signal', None))
            )
            
            print(f"VIX data inserted to SQLite: {result}")
            
            # Auto-sync to MotherDuck every 10 records
            sqlite_count = get_vix_sqlite_count()
            sync_result = None
            if sqlite_count >= 10:
                print(f"Auto-syncing {sqlite_count} records to MotherDuck...")
                sync_result = sync_sqlite_to_motherduck()
                print(f"Sync result: {sync_result}")
            
            return {
                "status": result.get("status", "error"),
                "message": f"VIX data recorded: VIX={vix}, VX30={vx30}",
                "data": result,
                "pending_sync": sqlite_count if sqlite_count < 10 else 0,
                "sync_result": sync_result
            }
        
        # TODO: Enable Google Sheets integration when ready
        # For now, just return success with the received data
        
        return {
            "status": "ok",
            "message": f"Webhook received for {data.get('symbol', data.get('ticker', 'unknown'))}",
            "data": data
        }
    
    except Exception as e:
        print(f"ERROR: Webhook processing failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/webhook/test")
async def test_webhook():
    """Test endpoint to verify webhook is working."""
    return {
        "status": "ok",
        "message": "Webhook endpoint is active",
        "webhook_url": "/webhook/tradingview",
        "example_tradingview_message": {
            "symbol": "{{ticker}}",
            "price": "{{close}}",
            "time": "{{time}}",
            "indicator_name": "My Indicator",
            "indicator_value": "{{plot_0}}",
            "signal": "BUY",
            "notes": "Custom alert message"
        },
        "example_vix_message": {
            "type": "vix",
            "vix": "{{close}}",
            "vx30": "{{plot_0}}",
            "notes": "VIX alert from TradingView"
        }
    }


# ============================================================
# MEMORY MONITORING ENDPOINTS (for debugging 512MB Render limit)
# ============================================================

@app.get("/api/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/memory")
async def memory_stats():
    """Get current memory usage stats for debugging.
    
    Useful for monitoring memory on Render's 512MB limit.
    Access: /api/memory
    """
    import gc
    
    # Force garbage collection first
    gc.collect()
    
    try:
        import resource
        # Get memory usage on Unix systems (macOS/Linux)
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        memory_mb = rusage.ru_maxrss / 1024 / 1024  # macOS returns bytes
        # On Linux it's in KB, so adjust if needed
        import platform
        if platform.system() == 'Linux':
            memory_mb = rusage.ru_maxrss / 1024
    except ImportError:
        memory_mb = None
    
    # Alternative: use psutil if available
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        psutil_memory_mb = memory_info.rss / 1024 / 1024
    except ImportError:
        psutil_memory_mb = None
    
    # Get cache stats
    cache_stats = _query_cache.stats()
    
    # Get connection pool info
    pool_connections = len(_connection_pool._connections)
    
    # Get garbage collector stats
    gc_stats = {
        "collections": gc.get_count(),
        "objects": len(gc.get_objects()),
        "garbage": len(gc.garbage)
    }
    
    return {
        "status": "ok",
        "memory": {
            "resource_maxrss_mb": round(memory_mb, 2) if memory_mb else None,
            "psutil_rss_mb": round(psutil_memory_mb, 2) if psutil_memory_mb else None,
            "render_limit_mb": 512,
            "warning_threshold_mb": 450
        },
        "cache": cache_stats,
        "connection_pool": {
            "active_connections": pool_connections
        },
        "garbage_collector": gc_stats,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear all cached data to free memory.
    
    Use this if memory usage is too high.
    """
    import gc
    
    # Clear the query cache
    _query_cache.invalidate()
    
    # Force garbage collection
    gc.collect()
    
    return {
        "status": "ok",
        "message": "Cache cleared and garbage collected",
        "cache_stats": _query_cache.stats()
    }


@app.post("/api/vix/sync")
async def sync_vix_to_motherduck():
    """Sync VIX data from local SQLite to MotherDuck.
    Call this periodically or before app restarts to preserve data.
    """
    result = sync_sqlite_to_motherduck()
    return result


@app.get("/api/vix/sqlite-count")
async def get_sqlite_count():
    """Get count of records in local SQLite (pending sync)."""
    count = get_vix_sqlite_count()
    return {"status": "ok", "pending_records": count}


@app.get("/api/vix")
async def get_vix():
    """Get the latest VIX and VX30 values from the database."""
    latest = get_latest_vix_data()
    if latest:
        return {
            "status": "ok",
            "data": latest
        }
    return {
        "status": "ok",
        "data": None,
        "message": "No VIX data available"
    }


@app.get("/api/vix/history")
async def get_vix_history(limit: int = 100):
    """Get historical VIX and VX30 values."""
    conn = get_options_db_connection()
    if not conn:
        return {"status": "error", "message": "Options database not configured"}
    
    try:
        result = conn.execute(f"""
            SELECT timestamp, vix, vx30, source, notes
            FROM vix_data 
            ORDER BY timestamp DESC 
            LIMIT {min(limit, 1000)}
        """).fetchall()
        conn.close()
        
        history = []
        for row in result:
            history.append({
                "timestamp": row[0].isoformat() if hasattr(row[0], 'isoformat') else str(row[0]),
                "vix": row[1],
                "vx30": row[2],
                "source": row[3],
                "notes": row[4]
            })
        
        return {
            "status": "ok",
            "count": len(history),
            "data": history
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/vix-chart", response_class=HTMLResponse)
async def vix_chart(request: Request):
    """Display VIX and VX30 history with Plotly chart.
    Combines data from SQLite (webhook) and MotherDuck (historical).
    """
    history = []
    timestamps = []
    vix_values = []
    vx30_values = []
    
    # First, get webhook data from SQLite - reduced from 10000 to 500 for memory
    sqlite_data = get_vix_sqlite_history(limit=500)
    sqlite_count = get_vix_sqlite_count()
    
    for row in sqlite_data:
        ts = row['timestamp']
        history.append({
            "timestamp": ts,
            "vix": row['vix'],
            "vx30": row['vx30'],
            "source": row['source'],
            "notes": row['notes']
        })
        timestamps.append(ts)
        vix_values.append(row['vix'])
        vx30_values.append(row['vx30'])
    
    # Then get historical data from MotherDuck - limit to 1000 for memory
    motherduck_count = 0
    conn = get_options_db_connection()
    if conn:
        try:
            motherduck_count = conn.execute("SELECT COUNT(*) FROM vix_data").fetchone()[0]
            
            # Added LIMIT 1000 to prevent memory issues
            result = conn.execute("""
                SELECT timestamp, vix, vx30, source, notes
                FROM vix_data 
                ORDER BY timestamp DESC
                LIMIT 1000
            """).fetchall()
            conn.close()
            
            for row in result:
                ts = row[0].isoformat() if hasattr(row[0], 'isoformat') else str(row[0])
                history.append({
                    "timestamp": ts,
                    "vix": row[1],
                    "vx30": row[2],
                    "source": row[3],
                    "notes": row[4]
                })
                timestamps.append(ts)
                vix_values.append(row[1])
                vx30_values.append(row[2])
        except Exception as e:
            print(f"Error reading MotherDuck VIX data: {e}")
            if conn:
                conn.close()
    
    total_count = sqlite_count + motherduck_count
    
    # Sort by timestamp descending (newest first for table)
    # Parse timestamps to datetime for proper sorting (handles mixed formats)
    from datetime import datetime as dt

    def parse_timestamp(ts_str):
        """Parse timestamp string to datetime for proper sorting."""
        try:
            # Handle ISO format with T separator
            if 'T' in ts_str:
                return dt.fromisoformat(ts_str.replace('Z', '+00:00'))
            # Handle space-separated format
            return dt.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
        except Exception:
            return dt.min  # Fallback for unparseable timestamps
    
    combined = list(zip(timestamps, vix_values, vx30_values, history))
    combined.sort(key=lambda x: parse_timestamp(x[0]), reverse=True)
    
    if combined:
        timestamps, vix_values, vx30_values, history = zip(*combined)
        timestamps = list(timestamps)
        vix_values = list(vix_values)
        vx30_values = list(vx30_values)
        history = list(history)
    
    # Reverse for chronological order in chart
    chart_timestamps = timestamps.copy()
    chart_vix = vix_values.copy()
    chart_vx30 = vx30_values.copy()
    chart_timestamps.reverse()
    chart_vix.reverse()
    chart_vx30.reverse()
    
    # Get latest
    latest = history[0] if history else None
    
    return templates.TemplateResponse("vix_chart.html", {
        "request": request,
        "latest": latest,
        "history": history,
        "timestamps": chart_timestamps,
        "vix_values": chart_vix,
        "vx30_values": chart_vx30,
        "total_count": total_count
    })


@app.get("/vix-chart/download")
async def download_vix_csv():
    """Download complete VIX/VX30 history as CSV."""
    from fastapi.responses import StreamingResponse
    import io
    
    conn = get_options_db_connection()
    if not conn:
        return {"status": "error", "message": "Database not available"}
    
    try:
        result = conn.execute("""
            SELECT timestamp, vix, vx30, source, notes
            FROM vix_data 
            ORDER BY timestamp ASC
        """).fetchall()
        conn.close()
        
        # Create CSV in memory
        output = io.StringIO()
        output.write("timestamp,vix,vx30,spread_pct,source,notes\n")
        
        for row in result:
            ts = row[0].isoformat() if hasattr(row[0], 'isoformat') else str(row[0])
            vix = row[1] if row[1] else ''
            vx30 = row[2] if row[2] else ''
            source = row[3] if row[3] else ''
            notes = row[4] if row[4] else ''
            
            # Calculate spread
            if vix and vx30:
                spread = ((vx30 - vix) / vix * 100)
                spread_str = f"{spread:.2f}"
            else:
                spread_str = ''
            
            # Escape notes for CSV
            notes = str(notes).replace('"', '""')
            
            output.write(f'{ts},{vix},{vx30},{spread_str},{source},"{notes}"\n')
        
        output.seek(0)
        
        # Return as downloadable CSV
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=vix_history_{datetime.now().strftime('%Y%m%d')}.csv"}
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================
# Focus List Endpoints
# ============================================

@app.post("/api/focus-list/add")
async def api_add_to_focus_list(
    symbol: str = Form(...),
    scanner_name: str = Form(None),
    scan_date: str = Form(None),
    entry_price: float = Form(None),
    strength: float = Form(None),
    quality: str = Form(None),
    notes: str = Form(None)
):
    """Add a setup to the focus list."""
    result = add_to_focus_list(
        symbol=symbol.upper(),
        scanner_name=scanner_name,
        scan_date=scan_date,
        entry_price=entry_price,
        strength=int(strength) if strength is not None else None,
        quality=quality,
        notes=notes
    )
    return JSONResponse(result)


@app.post("/api/focus-list/remove")
async def api_remove_from_focus_list(item_id: int = Form(...)):
    """Remove an item from the focus list."""
    result = remove_from_focus_list(item_id)
    return JSONResponse(result)


@app.get("/api/focus-list")
async def api_get_focus_list():
    """Get all focus list items."""
    items = get_focus_list()
    return JSONResponse({"status": "ok", "items": items, "count": len(items)})


@app.post("/api/trigger-workflow")
async def api_trigger_workflow(request: Request):
    """Trigger GitHub Actions workflow with selected symbols."""
    try:
        import httpx
        body = await request.json()
        data = body.get('data', '')
        
        if not data:
            return JSONResponse({"error": "No data provided"}, status_code=400)
        
        # Get GitHub token from environment
        github_token = os.environ.get('GITHUB_TOKEN')
        if not github_token:
            return JSONResponse({"error": "GitHub token not configured"}, status_code=500)
        
        # GitHub API endpoint for workflow dispatch
        owner = os.environ.get('GITHUB_REPO_OWNER', 'GeorgeHategan')
        repo = os.environ.get('GITHUB_REPO_NAME', 'scanner_data')
        workflow_id = os.environ.get('GITHUB_WORKFLOW_ID', '217568428')
        
        url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches"
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {github_token}",
        }
        
        payload = {
            "ref": "main",
            "inputs": {
                "symbols": data
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=10.0)
        
        if response.status_code == 204:
            return JSONResponse({"status": "ok", "message": "Workflow triggered successfully"})
        else:
            return JSONResponse({
                "error": f"GitHub API error: {response.status_code}",
                "details": response.text
            }, status_code=response.status_code)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/chart-data/{symbol}")
async def api_get_chart_data(symbol: str, days: int = 30):
    """Get OHLCV data and options walls for charting."""
    try:
        # Get price data
        conn = get_db_connection(DUCKDB_PATH)
        results = conn.execute("""
            SELECT date, open, high, low, close, volume
            FROM scanner_data.main.daily_cache
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT ?
        """, [symbol.upper(), days]).fetchall()
        conn.close()
        
        if not results:
            return JSONResponse({
                "status": "error",
                "message": "No data found"
            })
        
        # Reverse to get chronological order
        results.reverse()
        
        chart_data = {
            "symbol": symbol.upper(),
            "dates": [str(row[0]) for row in results],
            "open": [float(row[1]) if row[1] else None for row in results],
            "high": [float(row[2]) if row[2] else None for row in results],
            "low": [float(row[3]) if row[3] else None for row in results],
            "close": [
                float(row[4]) if row[4] else None for row in results
            ],
            "volume": [int(row[5]) if row[5] else 0 for row in results]
        }
        
        # Get options walls if available
        walls_data = None
        try:
            opt_conn = get_options_db_connection()
            walls_result = opt_conn.execute("""
                SELECT stock_price,
                       call_wall_strike, call_wall_oi,
                       call_wall_2_strike, call_wall_2_oi,
                       call_wall_3_strike, call_wall_3_oi,
                       put_wall_strike, put_wall_oi,
                       put_wall_2_strike, put_wall_2_oi,
                       put_wall_3_strike, put_wall_3_oi
                FROM options_walls
                WHERE underlying_symbol = ?
                ORDER BY scan_date DESC
                LIMIT 1
            """, [symbol.upper()]).fetchone()
            opt_conn.close()
            
            if walls_result:
                walls_data = {
                    'stock_price': (
                        float(walls_result[0]) if walls_result[0] else None
                    ),
                    'call_walls': [
                        {
                            'strike': float(walls_result[1]),
                            'oi': int(walls_result[2])
                        } if walls_result[1] else None,
                        {
                            'strike': float(walls_result[3]),
                            'oi': int(walls_result[4])
                        } if walls_result[3] else None,
                        {
                            'strike': float(walls_result[5]),
                            'oi': int(walls_result[6])
                        } if walls_result[5] else None
                    ],
                    'put_walls': [
                        {
                            'strike': float(walls_result[7]),
                            'oi': int(walls_result[8])
                        } if walls_result[7] else None,
                        {
                            'strike': float(walls_result[9]),
                            'oi': int(walls_result[10])
                        } if walls_result[9] else None,
                        {
                            'strike': float(walls_result[11]),
                            'oi': int(walls_result[12])
                        } if walls_result[11] else None
                    ]
                }
                # Filter out None values
                walls_data['call_walls'] = [
                    w for w in walls_data['call_walls'] if w
                ]
                walls_data['put_walls'] = [
                    w for w in walls_data['put_walls'] if w
                ]
        except Exception as e:
            print(f"Options walls not available for {symbol}: {e}")
        
        chart_data['walls'] = walls_data
        
        return JSONResponse({"status": "ok", "data": chart_data})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


@app.get("/focus-list", response_class=HTMLResponse)
async def focus_list_page(request: Request):
    """Display the focus list page with day separators, charts and full stock info."""
    items = get_focus_list()
    
    # Enrich items with stock metadata from the database
    if items:
        symbols = list(set(item['symbol'] for item in items))
        
        # Fetch metadata for all symbols
        try:
            conn = get_db_connection(DUCKDB_PATH)
            
            # Get fundamental cache data (earnings_date not in this table)
            placeholders = ','.join(['?' for _ in symbols])
            metadata_query = f'''
                SELECT symbol, name, market_cap, sector, industry
                FROM scanner_data.main.fundamental_cache
                WHERE symbol IN ({placeholders})
            '''
            metadata_results = conn.execute(metadata_query, symbols).fetchall()
            metadata_dict = {
                row[0]: {
                    'company': row[1] or '',
                    'market_cap': format_market_cap(row[2]) or '',
                    'sector': row[3] or '',
                    'industry': row[4] or ''
                } for row in metadata_results
            }
            
            # Get volume data
            vol_query = f'''
                SELECT dc.symbol, dc.volume, dc.avg_volume_20
                FROM scanner_data.main.daily_cache dc
                INNER JOIN (
                    SELECT symbol, MAX(date) as max_date
                    FROM scanner_data.main.daily_cache
                    WHERE symbol IN ({placeholders})
                    GROUP BY symbol
                ) latest ON dc.symbol = latest.symbol AND dc.date = latest.max_date
            '''
            vol_results = conn.execute(vol_query, symbols).fetchall()
            volume_dict = {
                row[0]: {
                    'volume': int(row[1]) if row[1] else 0,
                    'avg_volume': int(row[2]) if row[2] else 0
                } for row in vol_results
            }
            
            # Get confirmations (other scanners that picked each symbol)
            conf_query = f'''
                SELECT symbol, scanner_name, scan_date, signal_strength
                FROM scanner_results
                WHERE symbol IN ({placeholders})
                ORDER BY symbol, scan_date DESC, scanner_name
            '''
            conf_results = conn.execute(conf_query, symbols).fetchall()
            confirmations_dict = {}
            for row in conf_results:
                sym = row[0]
                if sym not in confirmations_dict:
                    confirmations_dict[sym] = []
                confirmations_dict[sym].append({
                    'scanner': row[1],
                    'date': str(row[2])[:10] if row[2] else '',
                    'strength': row[3]
                })
            
            conn.close()
            
            # Fetch options signals for all symbols
            options_signals_dict = {}
            darkpool_signals_dict = {}
            options_walls_dict = {}
            fund_quality_dict = {}
            
            if OPTIONS_DUCKDB_PATH:
                try:
                    options_conn = get_options_db_connection()
                    if options_conn:
                        placeholders = ','.join(['?' for _ in symbols])
                        
                        # Get options signals
                        options_query = f'''
                            SELECT underlying_symbol, signal_date, signal_type, 
                                   signal_strength, confidence_score, strike, dte,
                                   premium_spent, notes, direction
                            FROM accumulation_signals
                            WHERE underlying_symbol IN ({placeholders})
                            ORDER BY underlying_symbol, signal_date DESC
                        '''
                        options_results = options_conn.execute(options_query, symbols).fetchall()
                        
                        for row in options_results:
                            sym = row[0]
                            if sym not in options_signals_dict:
                                options_signals_dict[sym] = []
                            options_signals_dict[sym].append({
                                'date': str(row[1]) if row[1] else '',
                                'signal_type': row[2],
                                'strength': row[3],
                                'confidence': row[4],
                                'strike': row[5],
                                'dte': row[6],
                                'premium': row[7],
                                'notes': row[8],
                                'direction': row[9] if len(row) > 9 else None
                            })
                        
                        # Get darkpool signals
                        dp_query = f'''
                            SELECT ticker, signal_date, signal_type, 
                                   signal_strength, confidence_score, direction,
                                   dp_volume, dp_premium, avg_price,
                                   sell_volume, buy_volume, buy_sell_ratio,
                                   block_count, avg_block_size, consecutive_days, notes
                            FROM darkpool_signals
                            WHERE ticker IN ({placeholders})
                            ORDER BY ticker, signal_date DESC
                        '''
                        dp_results = options_conn.execute(dp_query, symbols).fetchall()
                        
                        for row in dp_results:
                            sym = row[0]
                            if sym not in darkpool_signals_dict:
                                darkpool_signals_dict[sym] = []
                            darkpool_signals_dict[sym].append({
                                'date': str(row[1]) if row[1] else '',
                                'signal_type': row[2],
                                'strength': row[3],
                                'confidence': row[4],
                                'direction': row[5],
                                'dp_volume': row[6],
                                'dp_premium': row[7],
                                'avg_price': row[8],
                                'buy_volume': row[10],
                                'sell_volume': row[9],
                                'buy_sell_ratio': row[11],
                                'block_count': row[12],
                                'avg_block_size': row[13],
                                'consecutive_days': row[14],
                                'notes': row[15]
                            })
                        
                        # Get options walls (latest per symbol)
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
                            QUALIFY ROW_NUMBER() OVER (PARTITION BY underlying_symbol ORDER BY scan_date DESC) = 1
                        '''
                        walls_results = options_conn.execute(walls_query, symbols).fetchall()
                        
                        for row in walls_results:
                            sym = row[0]
                            stock_price = row[2] or 0
                            call_wall_1_strike = row[3] or 0
                            put_wall_1_strike = row[9] or 0
                            
                            if call_wall_1_strike and put_wall_1_strike:
                                gamma_flip = (call_wall_1_strike + put_wall_1_strike) / 2
                            else:
                                gamma_flip = None
                            
                            if gamma_flip and stock_price:
                                gamma_flip_distance = ((gamma_flip - stock_price) / stock_price) * 100
                            else:
                                gamma_flip_distance = None
                            
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
                except Exception as e:
                    print(f"Error fetching options/darkpool data for focus list: {e}")
            
            # Fetch fundamental quality scores
            try:
                fund_conn = get_db_connection(DUCKDB_PATH)
                if fund_conn:
                    placeholders = ','.join(['?' for _ in symbols])
                    fund_query = f'''
                        SELECT symbol, fund_score, bar_blocks, bar_bucket, dot_state,
                               score_components, computed_at
                        FROM scanner_data.main.fundamental_quality_scores
                        WHERE symbol IN ({placeholders})
                    '''
                    fund_results = fund_conn.execute(fund_query, symbols).fetchall()
                    
                    for row in fund_results:
                        sym = row[0]
                        raw_inputs = {}
                        if row[5]:
                            import json
                            try:
                                components = json.loads(row[5]) if isinstance(row[5], str) else row[5]
                                raw_inputs = components.get('raw_inputs', {})
                            except:
                                pass
                        
                        fund_quality_dict[sym] = {
                            'fund_score': row[1],
                            'bar_blocks': row[2],
                            'bar_bucket': row[3],
                            'dot_state': row[4],
                            'computed_at': str(row[6]) if row[6] else None,
                            'operating_margin': raw_inputs.get('operating_margin'),
                            'return_on_equity': raw_inputs.get('return_on_equity'),
                            'profit_margin': raw_inputs.get('profit_margin'),
                            'quarterly_earnings_growth': raw_inputs.get('quarterly_earnings_growth'),
                            'pe_ratio': raw_inputs.get('pe_ratio')
                        }
                    
                    fund_conn.close()
            except Exception as e:
                print(f"Error fetching fundamental quality for focus list: {e}")
            
            # Enrich each item
            for item in items:
                sym = item['symbol']
                if sym in metadata_dict:
                    item.update(metadata_dict[sym])
                else:
                    item['company'] = ''
                    item['market_cap'] = ''
                    item['sector'] = ''
                    item['industry'] = ''
                
                if sym in volume_dict:
                    item['volume'] = volume_dict[sym]['volume']
                    item['avg_volume'] = volume_dict[sym]['avg_volume']
                    if item['avg_volume'] > 0:
                        item['volume_ratio'] = round(
                            item['volume'] / item['avg_volume'], 2)
                    else:
                        item['volume_ratio'] = 0
                else:
                    item['volume'] = 0
                    item['avg_volume'] = 0
                    item['volume_ratio'] = 0
                
                # Get confirmations (exclude the scanner this was saved from)
                if sym in confirmations_dict:
                    item['confirmations'] = [
                        c for c in confirmations_dict[sym]
                        if c['scanner'] != item.get('scanner_name')
                    ][:10]  # Limit to 10 most recent
                else:
                    item['confirmations'] = []
                
                # Add options signals
                item['options_signals'] = options_signals_dict.get(sym, [])
                
                # Add darkpool signals
                item['darkpool_signals'] = darkpool_signals_dict.get(sym, [])
                
                # Add options walls
                item['options_walls'] = options_walls_dict.get(sym)
                
                # Calculate OMS
                if item['options_walls'] and item['volume'] and item['volume'] > 0:
                    walls = item['options_walls']
                    oi_values = [
                        walls.get('call_wall_1', {}).get('oi') or 0,
                        walls.get('call_wall_2', {}).get('oi') or 0,
                        walls.get('call_wall_3', {}).get('oi') or 0,
                        walls.get('put_wall_1', {}).get('oi') or 0,
                        walls.get('put_wall_2', {}).get('oi') or 0,
                        walls.get('put_wall_3', {}).get('oi') or 0,
                    ]
                    max_oi = max(oi_values)
                    if max_oi > 0:
                        item['oms'] = round((max_oi * 100) / item['volume'], 2)
                    else:
                        item['oms'] = None
                else:
                    item['oms'] = None
                
                # Add fundamental quality
                item['fund_quality'] = fund_quality_dict.get(sym)
                    
        except Exception as e:
            print(f"Error enriching focus list items: {e}")
            # Set defaults if enrichment fails
            for item in items:
                item.setdefault('company', '')
                item.setdefault('market_cap', '')
                item.setdefault('sector', '')
                item.setdefault('industry', '')
                item.setdefault('volume', 0)
                item.setdefault('avg_volume', 0)
                item.setdefault('volume_ratio', 0)
                item.setdefault('confirmations', [])
                item.setdefault('options_signals', [])
                item.setdefault('darkpool_signals', [])
                item.setdefault('options_walls', None)
                item.setdefault('oms', None)
                item.setdefault('fund_quality', None)
    
    # Group items by added date (day only)
    from collections import OrderedDict
    grouped = OrderedDict()
    for item in items:
        # Extract just the date part
        added_ts = item['added_date']
        if 'T' in added_ts:
            day = added_ts.split('T')[0]
        else:
            day = added_ts.split(' ')[0]
        
        if day not in grouped:
            grouped[day] = []
        grouped[day].append(item)
    
    return templates.TemplateResponse('focus_list.html', {
        'request': request,
        'grouped_items': grouped,
        'total_count': len(items)
    })


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
        
        # Get connection to scanner_data for asset_types
        scanner_conn = get_db_connection(SCANNER_DATA_PATH)
        
        # Load asset types into memory for quick lookup
        asset_type_map = {}
        try:
            asset_types_result = scanner_conn.execute("""
                SELECT symbol, asset_type
                FROM main.asset_types
            """).fetchall()
            for row in asset_types_result:
                symbol_key = row[0].upper() if row[0] else ''
                if symbol_key:
                    asset_type_map[symbol_key] = row[1]
        except Exception as e:
            print(f"Warning: Could not load asset types: {e}")
        
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
            "SELECT DISTINCT scan_date FROM accumulation_signals ORDER BY scan_date DESC LIMIT 30"
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
                notes,
                direction
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
            # When searching by symbol, ignore date filter to show full history
            scan_date = None
        
        if min_confidence_val:
            query += " AND confidence_score >= ?"
            params.append(min_confidence_val)
        
        if min_premium_val:
            query += " AND premium_spent >= ?"
            params.append(min_premium_val)
        
        if asset_type:
            query += " AND asset_type = ?"
            params.append(asset_type)
        
        # Only apply date filter if no symbol is being searched
        if scan_date and not symbol:
            query += " AND scan_date = ?"
            params.append(scan_date)
        elif not symbol:
            # Default to latest date only if no specific symbol is requested
            # When viewing a specific symbol, show all dates
            latest_date = conn.execute(
                "SELECT MAX(scan_date) FROM accumulation_signals"
            ).fetchone()[0]
            if latest_date:
                query += " AND scan_date = ?"
                params.append(str(latest_date))
                scan_date = str(latest_date)
        
        # Increase limit when searching by symbol (showing full history)
        if symbol:
            query += " ORDER BY signal_date DESC, confidence_score DESC, premium_spent DESC LIMIT 500"
        else:
            query += " ORDER BY signal_date DESC, confidence_score DESC, premium_spent DESC LIMIT 200"
        
        results = conn.execute(query, params).fetchall()
        
        # Convert to list of dicts
        signals = []
        for row in results:
            underlying = row[3]
            signals.append({
                'signal_id': row[0],
                'signal_date': str(row[1]) if row[1] else '',
                'signal_type': row[2],
                'underlying_symbol': underlying,
                'sector': row[4] or 'ETF',
                'asset_type': asset_type_map.get(underlying.upper(), row[5] or 'Stock') if underlying else 'Stock',
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
                'notes': row[16],
                'direction': row[17] if len(row) > 17 else None
            })
        
        # Filter by asset_type if specified (use asset_types table value)
        if asset_type:
            signals = [s for s in signals if s['asset_type'] == asset_type]
        
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
            stats_query += " WHERE scan_date = ?"
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
                WHERE scan_date = ?
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
                WHERE scan_date = ?
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
                WHERE scan_date = ?
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
                ORDER BY signal_date ASC
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


@app.get("/api/options-chart-data")
async def options_chart_data(symbol: str):
    """API endpoint to get options flow data for a symbol over the last 60 days."""
    
    if not OPTIONS_DUCKDB_PATH:
        return {"error": "Options database not configured"}
    
    try:
        # Get price data for the last 60 days to establish date range
        scanner_conn = get_db_connection(SCANNER_DATA_PATH)
        if not scanner_conn:
            return {"error": "Could not connect to scanner database"}
        
        price_query = """
            SELECT date, close
            FROM main.daily_cache
            WHERE symbol = ?
                AND CAST(date AS DATE) >= CURRENT_DATE - INTERVAL '60 days'
            ORDER BY date
        """
        price_results = scanner_conn.execute(price_query, [symbol.upper()]).fetchall()
        
        # Use only trading days from price data (no weekends/holidays)
        all_dates = []
        price_map = {}
        for row in price_results:
            date_str = str(row[0])
            all_dates.append(date_str)
            price_map[date_str] = float(row[1]) if row[1] else None
        
        # Now get options flow data
        conn = get_options_db_connection()
        if not conn:
            return {"error": "Could not connect to options database"}
        
        query = """
            SELECT 
                signal_date,
                SUM(premium_spent) as total_premium,
                SUM(volume) as total_volume,
                COUNT(*) as signal_count,
                MAX(confidence_score) as max_confidence,
                STRING_AGG(DISTINCT signal_strength, ',') as strengths,
                STRING_AGG(DISTINCT signal_type, ',') as signal_types,
                STRING_AGG(DISTINCT direction, ',') as directions,
                AVG(CASE WHEN oi_change_pct IS NOT NULL THEN oi_change_pct ELSE 0 END) as avg_oi_change_pct,
                SUM(curr_oi) as total_oi,
                COUNT(CASE WHEN direction = 'BULLISH' THEN 1 END) as bullish_count,
                COUNT(CASE WHEN direction = 'BEARISH' THEN 1 END) as bearish_count,
                SUM(CASE WHEN direction = 'BULLISH' THEN premium_spent ELSE 0 END) as bullish_premium,
                SUM(CASE WHEN direction = 'BEARISH' THEN premium_spent ELSE 0 END) as bearish_premium,
                COUNT(CASE WHEN signal_type LIKE '%CALL%' OR option_symbol LIKE '%C%' THEN 1 END) as call_count,
                COUNT(CASE WHEN signal_type LIKE '%PUT%' OR option_symbol LIKE '%P%' THEN 1 END) as put_count
            FROM accumulation_signals
            WHERE underlying_symbol = ?
                AND signal_date >= CURRENT_DATE - INTERVAL '60 days'
            GROUP BY signal_date
            ORDER BY signal_date
        """
        
        results = conn.execute(query, [symbol.upper()]).fetchall()
        conn.close()
        
        # Create options map
        options_map = {}
        for row in results:
            date_str = str(row[0])
            options_map[date_str] = {
                'premium': float(row[1]) if row[1] else 0,
                'volume': int(row[2]) if row[2] else 0,
                'signal_count': int(row[3]) if row[3] else 0,
                'confidence': float(row[4]) if row[4] else 0,
                'strengths': str(row[5]) if row[5] else '',
                'signal_types': str(row[6]) if row[6] else '',
                'directions': str(row[7]) if row[7] else '',
                'avg_oi_change_pct': float(row[8]) if row[8] else 0,
                'total_oi': int(row[9]) if row[9] else 0,
                'bullish_count': int(row[10]) if row[10] else 0,
                'bearish_count': int(row[11]) if row[11] else 0,
                'bullish_premium': float(row[12]) if row[12] else 0,
                'bearish_premium': float(row[13]) if row[13] else 0,
                'call_count': int(row[14]) if row[14] else 0,
                'put_count': int(row[15]) if row[15] else 0
            }
        
        # Align data for all dates
        chart_data = {
            'dates': [],
            'premiums': [],
            'volumes': [],
            'signal_counts': [],
            'confidences': [],
            'strengths': [],
            'signal_types': [],
            'prices': [],
            'directions': [],
            'avg_oi_change_pcts': [],
            'total_ois': [],
            'bullish_counts': [],
            'bearish_counts': [],
            'bullish_premiums': [],
            'bearish_premiums': [],
            'call_counts': [],
            'put_counts': []
        }
        
        for date in all_dates:
            chart_data['dates'].append(date)
            chart_data['prices'].append(price_map.get(date))
            
            # Add options data if exists, otherwise 0/empty
            if date in options_map:
                opt = options_map[date]
                chart_data['premiums'].append(opt['premium'])
                chart_data['volumes'].append(opt['volume'])
                chart_data['signal_counts'].append(opt['signal_count'])
                chart_data['confidences'].append(opt['confidence'])
                chart_data['strengths'].append(opt['strengths'])
                chart_data['signal_types'].append(opt['signal_types'])
                chart_data['directions'].append(opt['directions'])
                chart_data['avg_oi_change_pcts'].append(opt['avg_oi_change_pct'])
                chart_data['total_ois'].append(opt['total_oi'])
                chart_data['bullish_counts'].append(opt['bullish_count'])
                chart_data['bearish_counts'].append(opt['bearish_count'])
                chart_data['bullish_premiums'].append(opt['bullish_premium'])
                chart_data['bearish_premiums'].append(opt['bearish_premium'])
                chart_data['call_counts'].append(opt['call_count'])
                chart_data['put_counts'].append(opt['put_count'])
            else:
                chart_data['premiums'].append(0)
                chart_data['volumes'].append(0)
                chart_data['signal_counts'].append(0)
                chart_data['confidences'].append(0)
                chart_data['strengths'].append('')
                chart_data['signal_types'].append('')
                chart_data['directions'].append('')
                chart_data['avg_oi_change_pcts'].append(0)
                chart_data['total_ois'].append(0)
                chart_data['bullish_counts'].append(0)
                chart_data['bearish_counts'].append(0)
                chart_data['bullish_premiums'].append(0)
                chart_data['bearish_premiums'].append(0)
                chart_data['call_counts'].append(0)
                chart_data['put_counts'].append(0)
        
        return chart_data
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/api/darkpool-chart-data")
async def darkpool_chart_data(symbol: str):
    """API endpoint to get darkpool trade data for a symbol over the last 60 days."""
    
    if not OPTIONS_DUCKDB_PATH:
        return {"error": "Options database not configured"}
    
    try:
        # Get price data for the last 60 days to establish date range
        scanner_conn = get_db_connection(SCANNER_DATA_PATH)
        if not scanner_conn:
            return {"error": "Could not connect to scanner database"}
        
        price_query = """
            SELECT date, close
            FROM main.daily_cache
            WHERE symbol = ?
                AND CAST(date AS DATE) >= CURRENT_DATE - INTERVAL '60 days'
            ORDER BY date
        """
        price_results = scanner_conn.execute(price_query, [symbol.upper()]).fetchall()
        
        # Use only trading days from price data (no weekends/holidays)
        all_dates = []
        price_map = {}
        for row in price_results:
            date_str = str(row[0])
            all_dates.append(date_str)
            price_map[date_str] = float(row[1]) if row[1] else None
        
        # Now get darkpool data
        conn = get_options_db_connection()
        if not conn:
            return {"error": "Could not connect to options database"}
        
        query = """
            SELECT 
                signal_date,
                SUM(dp_volume) as total_volume,
                SUM(dp_premium) as total_premium,
                COUNT(*) as trade_count,
                MAX(confidence_score) as max_confidence,
                STRING_AGG(DISTINCT direction, ',') as directions
            FROM darkpool_signals
            WHERE ticker = ?
                AND signal_date >= CURRENT_DATE - INTERVAL '60 days'
            GROUP BY signal_date
            ORDER BY signal_date
        """
        
        results = conn.execute(query, [symbol.upper()]).fetchall()
        conn.close()
        
        # Create darkpool map
        darkpool_map = {}
        for row in results:
            date_str = str(row[0])
            darkpool_map[date_str] = {
                'volume': int(row[1]) if row[1] else 0,
                'premium': float(row[2]) if row[2] else 0,
                'trade_count': int(row[3]) if row[3] else 0,
                'confidence': float(row[4]) if row[4] else 0,
                'direction': str(row[5]) if row[5] else ''
            }
        
        # Build detailed data for individual chart (with full date strings)
        dates = []
        premiums = []
        directions = []
        volumes = []
        trade_counts = []
        confidences = []
        prices = []
        
        for date in all_dates:
            dates.append(date)  # Keep full YYYY-MM-DD format
            
            # Add darkpool data if exists
            if date in darkpool_map:
                dp = darkpool_map[date]
                premiums.append(dp['premium'])
                directions.append(dp['direction'])
                volumes.append(dp['volume'])
                trade_counts.append(dp['trade_count'])
                confidences.append(dp['confidence'])
            else:
                premiums.append(0)
                directions.append('N/A')
                volumes.append(0)
                trade_counts.append(0)
                confidences.append('N/A')
            
            # Add price if available
            prices.append(price_map.get(date))
        
        return {
            'dates': dates,
            'premiums': premiums,
            'directions': directions,
            'volumes': volumes,
            'trade_counts': trade_counts,
            'confidences': confidences,
            'prices': prices
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/api/darkpool-chart-data-bulk")
async def darkpool_chart_data_bulk(symbols: str):
    """Bulk API endpoint to get darkpool data for multiple symbols at once."""
    
    if not OPTIONS_DUCKDB_PATH:
        return {"error": "Options database not configured"}
    
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        if not symbol_list:
            return {"error": "No symbols provided"}
        
        conn = get_options_db_connection()
        if not conn:
            return {"error": "Could not connect to options database"}
        
        # Find the most recent date with data across all symbols
        placeholders = ','.join(['?' for _ in symbol_list])
        max_date_query = f"""
            SELECT MAX(signal_date) 
            FROM darkpool_signals 
            WHERE ticker IN ({placeholders})
        """
        max_date_result = conn.execute(max_date_query, symbol_list).fetchone()
        
        if max_date_result and max_date_result[0]:
            end_date = datetime.strptime(str(max_date_result[0]), '%Y-%m-%d').date()
        else:
            end_date = datetime.now().date()
        
        # Get trading days from price data (no weekends/holidays)
        scanner_conn = get_db_connection(SCANNER_DATA_PATH)
        if scanner_conn:
            # Get any symbol's price data to determine trading days
            trading_days_query = """
                SELECT DISTINCT date
                FROM main.daily_cache
                WHERE CAST(date AS DATE) >= CURRENT_DATE - INTERVAL '60 days'
                    AND CAST(date AS DATE) <= ?
                ORDER BY date
            """
            trading_days_results = scanner_conn.execute(trading_days_query, [end_date.strftime('%Y-%m-%d')]).fetchall()
            all_dates = [str(row[0]) for row in trading_days_results]
        else:
            # Fallback to calendar days if price data unavailable
            start_date = end_date - timedelta(days=59)
            all_dates = []
            current = start_date
            while current <= end_date:
                all_dates.append(current.strftime('%Y-%m-%d'))
                current += timedelta(days=1)
        
        # Get darkpool data for all symbols in one query
        placeholders = ','.join(['?' for _ in symbol_list])
        query = f"""
            SELECT 
                ticker,
                signal_date,
                SUM(dp_premium) as total_premium,
                STRING_AGG(DISTINCT direction, ',') as directions
            FROM darkpool_signals
            WHERE ticker IN ({placeholders})
                AND signal_date >= CURRENT_DATE - INTERVAL '60 days'
            GROUP BY ticker, signal_date
            ORDER BY ticker, signal_date
        """
        
        results = conn.execute(query, symbol_list).fetchall()
        conn.close()
        
        # Organize raw data by symbol and date
        raw_data = {}
        for row in results:
            ticker = row[0]
            date_str = str(row[1])
            premium = float(row[2]) if row[2] else 0
            direction = str(row[3]) if row[3] else ''
            
            if ticker not in raw_data:
                raw_data[ticker] = {}
            
            raw_data[ticker][date_str] = {
                'premium': premium,
                'direction': direction
            }
        
        # Build complete 60-day data for each symbol
        symbol_data = {}
        for ticker in symbol_list:
            symbol_data[ticker] = []
            
            for date in all_dates:
                # Format date as MM/DD
                date_parts = date.split('-')
                label = f"{date_parts[1]}/{date_parts[2]}" if len(date_parts) == 3 else date
                
                # Get data for this date or use zeros
                if ticker in raw_data and date in raw_data[ticker]:
                    data_point = raw_data[ticker][date]
                    premium = data_point['premium']
                    direction = data_point['direction'].upper()
                    
                    # Determine color
                    if 'BUY' in direction or 'BULLISH' in direction:
                        color = 'rgba(39, 174, 96, 0.7)'
                    elif 'SELL' in direction or 'BEARISH' in direction:
                        color = 'rgba(231, 76, 60, 0.7)'
                    else:
                        color = 'rgba(243, 156, 18, 0.7)'
                else:
                    premium = 0
                    color = 'rgba(189, 195, 199, 0.3)'  # Gray for no data
                
                symbol_data[ticker].append({
                    'label': label,
                    'premium': premium,
                    'color': color
                })
        
        return symbol_data
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/api/options-chart-data-bulk")
async def options_chart_data_bulk(symbols: str):
    """Bulk API endpoint to get options flow data for multiple symbols at once."""
    
    if not OPTIONS_DUCKDB_PATH:
        return {"error": "Options database not configured"}
    
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        if not symbol_list:
            return {"error": "No symbols provided"}
        
        conn = get_options_db_connection()
        if not conn:
            return {"error": "Could not connect to options database"}
        
        # Find the most recent date with data across all symbols
        placeholders = ','.join(['?' for _ in symbol_list])
        max_date_query = f"""
            SELECT MAX(signal_date) 
            FROM accumulation_signals 
            WHERE underlying_symbol IN ({placeholders})
        """
        max_date_result = conn.execute(max_date_query, symbol_list).fetchone()
        
        if max_date_result and max_date_result[0]:
            end_date = datetime.strptime(str(max_date_result[0]), '%Y-%m-%d').date()
        else:
            end_date = datetime.now().date()
        
        # Get trading days from price data (no weekends/holidays)
        scanner_conn = get_db_connection(SCANNER_DATA_PATH)
        if scanner_conn:
            # Get any symbol's price data to determine trading days
            trading_days_query = """
                SELECT DISTINCT date
                FROM main.daily_cache
                WHERE CAST(date AS DATE) >= CURRENT_DATE - INTERVAL '60 days'
                    AND CAST(date AS DATE) <= ?
                ORDER BY date
            """
            trading_days_results = scanner_conn.execute(trading_days_query, [end_date.strftime('%Y-%m-%d')]).fetchall()
            all_dates = [str(row[0]) for row in trading_days_results]
        else:
            # Fallback to calendar days if price data unavailable
            start_date = end_date - timedelta(days=59)
            all_dates = []
            current = start_date
            while current <= end_date:
                all_dates.append(current.strftime('%Y-%m-%d'))
                current += timedelta(days=1)
        
        # Get options flow data for all symbols in one query
        placeholders = ','.join(['?' for _ in symbol_list])
        query = f"""
            SELECT 
                underlying_symbol,
                signal_date,
                SUM(CASE WHEN direction = 'BULLISH' THEN premium_spent ELSE 0 END) as bullish_premium,
                SUM(CASE WHEN direction = 'BEARISH' THEN premium_spent ELSE 0 END) as bearish_premium
            FROM accumulation_signals
            WHERE underlying_symbol IN ({placeholders})
                AND signal_date >= CURRENT_DATE - INTERVAL '60 days'
            GROUP BY underlying_symbol, signal_date
            ORDER BY underlying_symbol, signal_date
        """
        
        results = conn.execute(query, symbol_list).fetchall()
        conn.close()
        
        # Organize raw data by symbol and date
        raw_data = {}
        for row in results:
            ticker = row[0]
            date_str = str(row[1])
            bullish_premium = float(row[2]) if row[2] else 0
            bearish_premium = float(row[3]) if row[3] else 0
            
            if ticker not in raw_data:
                raw_data[ticker] = {}
            
            raw_data[ticker][date_str] = {
                'bullish_premium': bullish_premium,
                'bearish_premium': bearish_premium
            }
        
        # Build complete 60-day data for each symbol
        symbol_data = {}
        for ticker in symbol_list:
            symbol_data[ticker] = []
            
            for date in all_dates:
                # Format date as MM/DD
                date_parts = date.split('-')
                label = f"{date_parts[1]}/{date_parts[2]}" if len(date_parts) == 3 else date
                
                # Get data for this date or use zeros
                if ticker in raw_data and date in raw_data[ticker]:
                    data_point = raw_data[ticker][date]
                    bullish_premium = data_point['bullish_premium']
                    bearish_premium = data_point['bearish_premium']
                else:
                    bullish_premium = 0
                    bearish_premium = 0
                
                symbol_data[ticker].append({
                    'label': label,
                    'bullish_premium': bullish_premium,
                    'bearish_premium': bearish_premium
                })
        
        return symbol_data
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/darkpool-signals", response_class=HTMLResponse)
async def darkpool_signals(
    request: Request,
    signal_type: Optional[str] = Query(None),
    signal_strength: Optional[str] = Query(None),
    direction: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    min_confidence: Optional[str] = Query(None),
    min_premium: Optional[str] = Query(None),
    asset_type: Optional[str] = Query(None),
    scan_date: Optional[str] = Query(None)
):
    """Display dark pool signals from the options_data database."""
    
    min_confidence_val = int(min_confidence) if min_confidence else None
    min_premium_val = int(min_premium) if min_premium else None
    
    if not OPTIONS_DUCKDB_PATH:
        return templates.TemplateResponse('darkpool_signals.html', {
            'request': request,
            'error': 'Options database not configured.',
            'signals': [],
            'signal_types': [],
            'signal_strengths': [],
            'directions': [],
            'stats': {}
        })
    
    try:
        conn = get_options_db_connection()
        if not conn:
            raise Exception("Could not connect to options database")
        
        # Get connection to scanner_data for asset_types
        scanner_conn = get_db_connection(SCANNER_DATA_PATH)
        
        # Load asset types into memory for quick lookup
        asset_type_map = {}
        try:
            asset_types_result = scanner_conn.execute("""
                SELECT symbol, asset_type
                FROM main.asset_types
            """).fetchall()
            for row in asset_types_result:
                symbol_key = row[0].upper() if row[0] else ''
                if symbol_key:
                    asset_type_map[symbol_key] = row[1]
        except Exception as e:
            print(f"Warning: Could not load asset types: {e}")
        
        # Get available filter options
        signal_types = [row[0] for row in conn.execute(
            "SELECT DISTINCT signal_type FROM darkpool_signals ORDER BY signal_type"
        ).fetchall()]
        
        signal_strengths = [row[0] for row in conn.execute(
            "SELECT DISTINCT signal_strength FROM darkpool_signals ORDER BY signal_strength"
        ).fetchall()]
        
        directions = [row[0] for row in conn.execute(
            "SELECT DISTINCT direction FROM darkpool_signals WHERE direction IS NOT NULL ORDER BY direction"
        ).fetchall()]
        
        available_dates = [str(row[0]) for row in conn.execute(
            "SELECT DISTINCT scan_date FROM darkpool_signals ORDER BY scan_date DESC LIMIT 30"
        ).fetchall()]
        
        # Build the query with filters
        query = """
            SELECT 
                signal_id,
                signal_date,
                signal_type,
                ticker,
                direction,
                dp_volume,
                dp_premium,
                avg_price,
                sell_volume,
                buy_volume,
                buy_sell_ratio,
                block_count,
                avg_block_size,
                consecutive_days,
                confidence_score,
                signal_strength,
                notes
            FROM darkpool_signals
            WHERE 1=1
        """
        params = []
        
        if signal_type:
            query += " AND signal_type = ?"
            params.append(signal_type)
        
        if signal_strength:
            query += " AND signal_strength = ?"
            params.append(signal_strength)
        
        if direction:
            query += " AND direction = ?"
            params.append(direction)
        
        if symbol:
            query += " AND ticker = ?"
            params.append(symbol.upper())
            # When searching by symbol, ignore date filter to show full history
            scan_date = None
        
        if min_confidence_val:
            query += " AND confidence_score >= ?"
            params.append(min_confidence_val)
        
        if min_premium_val:
            query += " AND dp_premium >= ?"
            params.append(min_premium_val)
        
        # Only apply date filter if no symbol is being searched
        if scan_date and not symbol:
            query += " AND scan_date = ?"
            params.append(scan_date)
        elif not symbol:
            # Auto-select latest date only when not searching for a symbol
            latest_date = conn.execute(
                "SELECT MAX(scan_date) FROM darkpool_signals"
            ).fetchone()[0]
            if latest_date:
                query += " AND scan_date = ?"
                params.append(str(latest_date))
                scan_date = str(latest_date)
        
        # Increase limit when searching by symbol (showing full history)
        if symbol:
            query += " ORDER BY signal_date DESC, confidence_score DESC, dp_premium DESC LIMIT 500"
        else:
            query += " ORDER BY signal_date DESC, confidence_score DESC, dp_premium DESC LIMIT 200"
        
        results = conn.execute(query, params).fetchall()
        
        signals = []
        for row in results:
            ticker = row[3]
            signals.append({
                'signal_id': row[0],
                'signal_date': str(row[1]) if row[1] else '',
                'signal_type': row[2],
                'ticker': ticker,
                'asset_type': asset_type_map.get(ticker.upper(), 'Stock') if ticker else 'Stock',
                'direction': row[4],
                'dp_volume': row[5],
                'dp_premium': row[6],
                'avg_price': row[7],
                'buy_volume': row[8],
                'sell_volume': row[9],
                'buy_sell_ratio': row[10],
                'block_count': row[11],
                'avg_block_size': row[12],
                'consecutive_days': row[13],
                'confidence_score': row[14],
                'signal_strength': row[15],
                'notes': row[16]
            })
        
        # Filter by asset_type if specified
        if asset_type:
            signals = [s for s in signals if s['asset_type'] == asset_type]
        
        # Get stats
        stats_query = """
            SELECT 
                COUNT(*) as total_signals,
                COUNT(DISTINCT ticker) as unique_symbols,
                SUM(dp_premium) as total_premium,
                AVG(confidence_score) as avg_confidence,
                SUM(CASE WHEN signal_strength = 'EXTREME' THEN 1 ELSE 0 END) as extreme_count,
                SUM(CASE WHEN signal_strength = 'VERY_HIGH' THEN 1 ELSE 0 END) as very_high_count,
                SUM(CASE WHEN signal_strength = 'HIGH' THEN 1 ELSE 0 END) as high_count,
                SUM(CASE WHEN direction = 'BUY' OR direction = 'BULLISH' THEN 1 ELSE 0 END) as buy_count,
                SUM(CASE WHEN direction = 'SELL' OR direction = 'BEARISH' THEN 1 ELSE 0 END) as sell_count
            FROM darkpool_signals
        """
        if scan_date:
            stats_query += " WHERE scan_date = ?"
            stats_result = conn.execute(stats_query, [scan_date]).fetchone()
        elif symbol:
            stats_query += " WHERE ticker = ?"
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
            'high_count': stats_result[6],
            'buy_count': stats_result[7],
            'sell_count': stats_result[8]
        }
        
        # Get top symbols by premium
        if scan_date:
            top_symbols = conn.execute("""
                SELECT ticker, 
                       COUNT(*) as signal_count,
                       SUM(dp_premium) as total_premium,
                       MAX(confidence_score) as max_confidence
                FROM darkpool_signals
                WHERE scan_date = ?
                GROUP BY ticker
                ORDER BY total_premium DESC
                LIMIT 10
            """, [scan_date]).fetchall()
        elif symbol:
            top_symbols = conn.execute("""
                SELECT signal_date, 
                       COUNT(*) as signal_count,
                       SUM(dp_premium) as total_premium,
                       MAX(confidence_score) as max_confidence
                FROM darkpool_signals
                WHERE ticker = ?
                GROUP BY signal_date
                ORDER BY signal_date ASC
                LIMIT 10
            """, [symbol.upper()]).fetchall()
        else:
            top_symbols = []
        
        conn.close()
        
        return templates.TemplateResponse('darkpool_signals.html', {
            'request': request,
            'signals': signals,
            'signal_types': signal_types,
            'signal_strengths': signal_strengths,
            'directions': directions,
            'available_dates': available_dates,
            'selected_signal_type': signal_type,
            'selected_signal_strength': signal_strength,
            'selected_direction': direction,
            'selected_symbol': symbol,
            'selected_min_confidence': min_confidence_val,
            'selected_min_premium': min_premium_val,
            'selected_asset_type': asset_type,
            'selected_scan_date': scan_date,
            'stats': stats,
            'top_symbols': [{'symbol': r[0], 'count': r[1], 'premium': r[2], 'confidence': r[3]} for r in top_symbols],
            'error': None
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse('darkpool_signals.html', {
            'request': request,
            'error': f'Error loading dark pool signals: {str(e)}',
            'signals': [],
            'signal_types': [],
            'signal_strengths': [],
            'directions': [],
            'available_dates': [],
            'stats': {}
        })


@app.get("/discovery", response_class=HTMLResponse)
async def discovery_page(
    request: Request,
    asset_filter: Optional[str] = Query(None),  # 'ETF', 'Stock', or None for all
    min_volume: Optional[str] = Query(None),
    min_signals: Optional[str] = Query(None),
    signal_source: Optional[str] = Query(None),  # 'options', 'darkpool', or None for both
    scan_date: Optional[str] = Query(None)
):
    """Discover symbols with high options/darkpool activity NOT in scanner universe."""
    
    min_volume_val = int(min_volume) if min_volume else None
    min_signals_val = int(min_signals) if min_signals else 1
    
    if not OPTIONS_DUCKDB_PATH:
        return templates.TemplateResponse('discovery.html', {
            'request': request,
            'error': 'Options database not configured.',
            'symbols': [],
            'stats': {},
            'available_dates': []
        })
    
    try:
        options_conn = get_options_db_connection()
        scanner_conn = get_db_connection(SCANNER_DATA_PATH)
        
        if not options_conn:
            raise Exception("Could not connect to options database")
        
        # Get available dates from both tables
        available_dates = []
        try:
            opt_dates = [str(row[0]) for row in options_conn.execute(
                "SELECT DISTINCT scan_date FROM accumulation_signals ORDER BY scan_date DESC LIMIT 30"
            ).fetchall()]
            dp_dates = [str(row[0]) for row in options_conn.execute(
                "SELECT DISTINCT scan_date FROM darkpool_signals ORDER BY scan_date DESC LIMIT 30"
            ).fetchall()]
            available_dates = sorted(set(opt_dates + dp_dates), reverse=True)[:30]
        except:
            pass
        
        # Get latest date if not specified
        if not scan_date and available_dates:
            scan_date = available_dates[0]
        
        # Get ALL symbols from the complete scanner_data cache (the full ticker universe)
        scanner_symbols = set()
        try:
            scanner_result = scanner_conn.execute("""
                SELECT DISTINCT symbol 
                FROM main.daily_cache
            """).fetchall()
            scanner_symbols = {row[0].upper() for row in scanner_result}
        except Exception as e:
            print(f"Error getting scanner symbols from daily_cache: {e}")
        
        # Get asset type classifications from asset_types table
        asset_type_map = {}
        try:
            asset_types_result = scanner_conn.execute("""
                SELECT symbol, asset_type
                FROM main.asset_types
            """).fetchall()
            for row in asset_types_result:
                symbol = row[0].upper() if row[0] else ''
                if symbol:
                    asset_type_map[symbol] = {
                        'asset_type': row[1].upper() if row[1] else 'Stock',
                        'is_etf': row[1].lower() == 'etf' if row[1] else False
                    }
        except Exception as e:
            print(f"Error getting asset types: {e}")
        
        # Hardcoded ETF list for common symbols missing from asset_types
        known_etfs = {
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'BND',
            'IVV', 'EFA', 'VTV', 'VUG', 'IJH', 'IJR', 'VIG', 'VNQ', 'GLD', 'SLV',
            'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE',
            'VGT', 'VYM', 'VB', 'VO', 'VXF', 'MGC', 'MGK', 'MGV', 'IVE', 'IVW',
            'IEMG', 'IEFA', 'ITOT', 'IXUS', 'USMV', 'QUAL', 'MTUM', 'SIZE', 'VLUE', 'ESGU',
            'TLT', 'SHY', 'IEF', 'LQD', 'HYG', 'MUB', 'EMB', 'BNDX', 'VCSH', 'VCIT',
            'ARKK', 'ARKG', 'ARKW', 'ARKQ', 'ARKF', 'TAN', 'ICLN', 'QCLN', 'PBW', 'FAN',
            'TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'TNA', 'TZA', 'SPXL', 'SPXS', 'UDOW', 'SDOW',
            'SOXL', 'SOXS', 'LABU', 'LABD', 'ERX', 'ERY', 'FAS', 'FAZ', 'NUGT', 'DUST',
            'UVXY', 'SVXY', 'VXX', 'VIXY', 'VIX', 'TVIX', 'SVIX',
            'SMH', 'XBI', 'IBB', 'KRE', 'XRT', 'ITB', 'GDX', 'GDXJ', 'USO', 'UNG',
            'EEM', 'EWZ', 'FXI', 'EWJ', 'EWY', 'EWA', 'EWU', 'EWG', 'EWC', 'EWW'
        }
        
        # Build discovery query - get symbols with options/darkpool activity NOT in scanner universe
        discovery_data = {}
        
        # Get options flow signals
        if signal_source in [None, 'options']:
            opt_query = """
                SELECT 
                    underlying_symbol as symbol,
                    sector,
                    asset_type,
                    COUNT(*) as signal_count,
                    SUM(premium_spent) as total_premium,
                    MAX(confidence_score) as max_confidence,
                    MAX(signal_strength) as max_strength,
                    STRING_AGG(DISTINCT signal_type, ', ') as signal_types,
                    MAX(volume) as max_volume
                FROM accumulation_signals
                WHERE scan_date = ?
                GROUP BY underlying_symbol, sector, asset_type
            """
            opt_results = options_conn.execute(opt_query, [scan_date]).fetchall()
            
            for row in opt_results:
                symbol = row[0].upper() if row[0] else ''
                if symbol and symbol not in scanner_symbols:
                    # Get asset type from asset_types table (authoritative source)
                    asset_info = asset_type_map.get(symbol, {})
                    # Use asset_types, then hardcoded ETF list, then database value as last resort
                    if asset_info:
                        asset_type = asset_info.get('asset_type', 'Stock')
                        is_etf = asset_info.get('is_etf', False)
                    elif symbol in known_etfs:
                        asset_type = 'ETF'
                        is_etf = True
                    else:
                        asset_type = (row[2] or 'Stock').upper()
                        is_etf = asset_type == 'ETF'
                    sector = row[1] or 'Unknown'
                    
                    if symbol not in discovery_data:
                        discovery_data[symbol] = {
                            'symbol': symbol,
                            'sector': sector,
                            'asset_type': asset_type,
                            'is_etf': is_etf,
                            'options_signals': 0,
                            'options_premium': 0,
                            'options_confidence': 0,
                            'options_strength': '',
                            'options_types': '',
                            'darkpool_signals': 0,
                            'darkpool_premium': 0,
                            'darkpool_confidence': 0,
                            'darkpool_strength': '',
                            'darkpool_direction': '',
                            'max_volume': 0
                        }
                    discovery_data[symbol]['options_signals'] = row[3] or 0
                    discovery_data[symbol]['options_premium'] = row[4] or 0
                    discovery_data[symbol]['options_confidence'] = row[5] or 0
                    discovery_data[symbol]['options_strength'] = row[6] or ''
                    discovery_data[symbol]['options_types'] = row[7] or ''
                    discovery_data[symbol]['max_volume'] = max(discovery_data[symbol]['max_volume'], row[8] or 0)
        
        # Get dark pool signals
        if signal_source in [None, 'darkpool']:
            dp_query = """
                SELECT 
                    ticker as symbol,
                    COUNT(*) as signal_count,
                    SUM(dp_premium) as total_premium,
                    MAX(confidence_score) as max_confidence,
                    MAX(signal_strength) as max_strength,
                    STRING_AGG(DISTINCT direction, ', ') as directions,
                    MAX(dp_volume) as max_volume
                FROM darkpool_signals
                WHERE scan_date = ?
                GROUP BY ticker
            """
            dp_results = options_conn.execute(dp_query, [scan_date]).fetchall()
            
            for row in dp_results:
                symbol = row[0].upper() if row[0] else ''
                if symbol and symbol not in scanner_symbols:
                    # Get asset type from asset_types table if not already in discovery_data (authoritative source)
                    if symbol not in discovery_data:
                        asset_info = asset_type_map.get(symbol, {})
                        # Use asset_types, then hardcoded ETF list, then default to Stock
                        if asset_info:
                            asset_type = asset_info.get('asset_type', 'Stock')
                            is_etf = asset_info.get('is_etf', False)
                        elif symbol in known_etfs:
                            asset_type = 'ETF'
                            is_etf = True
                        else:
                            asset_type = 'Stock'
                            is_etf = False
                        sector = 'Unknown'
                        
                        discovery_data[symbol] = {
                            'symbol': symbol,
                            'sector': sector,
                            'asset_type': asset_type,
                            'is_etf': is_etf,
                            'options_signals': 0,
                            'options_premium': 0,
                            'options_confidence': 0,
                            'options_strength': '',
                            'options_types': '',
                            'darkpool_signals': 0,
                            'darkpool_premium': 0,
                            'darkpool_confidence': 0,
                            'darkpool_strength': '',
                            'darkpool_direction': '',
                            'max_volume': 0
                        }
                    discovery_data[symbol]['darkpool_signals'] = row[1] or 0
                    discovery_data[symbol]['darkpool_premium'] = row[2] or 0
                    discovery_data[symbol]['darkpool_confidence'] = row[3] or 0
                    discovery_data[symbol]['darkpool_strength'] = row[4] or ''
                    discovery_data[symbol]['darkpool_direction'] = row[5] or ''
                    discovery_data[symbol]['max_volume'] = max(discovery_data[symbol]['max_volume'], row[6] or 0)
        
        # Convert to list and apply filters
        symbols = list(discovery_data.values())
        
        # Filter by asset type using is_etf flag
        if asset_filter == 'ETF':
            symbols = [s for s in symbols if s.get('is_etf', False)]
        elif asset_filter == 'Stock':
            symbols = [s for s in symbols if not s.get('is_etf', False)]
        
        # Filter by minimum volume
        if min_volume_val:
            symbols = [s for s in symbols if s['max_volume'] >= min_volume_val]
        
        # Filter by minimum signals
        if min_signals_val:
            symbols = [s for s in symbols if (s['options_signals'] + s['darkpool_signals']) >= min_signals_val]
        
        # Calculate total signals and score for sorting
        for s in symbols:
            s['total_signals'] = s['options_signals'] + s['darkpool_signals']
            s['total_premium'] = s['options_premium'] + s['darkpool_premium']
            s['max_confidence'] = max(s['options_confidence'], s['darkpool_confidence'])
            # Score based on signals, premium and confidence
            s['score'] = (s['total_signals'] * 10) + (s['total_premium'] / 100000) + s['max_confidence']
        
        # Sort by score descending
        symbols = sorted(symbols, key=lambda x: x['score'], reverse=True)
        
        # Calculate stats
        stats = {
            'total_discovered': len(symbols),
            'scanner_universe_size': len(scanner_symbols),
            'total_options_signals': sum(s['options_signals'] for s in symbols),
            'total_darkpool_signals': sum(s['darkpool_signals'] for s in symbols),
            'total_premium': sum(s['total_premium'] for s in symbols),
            'etf_count': len([s for s in symbols if s.get('is_etf', False)]),
            'stock_count': len([s for s in symbols if not s.get('is_etf', False)])
        }
        
        options_conn.close()
        scanner_conn.close()
        
        return templates.TemplateResponse('discovery.html', {
            'request': request,
            'symbols': symbols[:200],  # Limit to 200 results
            'stats': stats,
            'available_dates': available_dates,
            'selected_scan_date': scan_date,
            'selected_asset_filter': asset_filter,
            'selected_min_volume': min_volume_val,
            'selected_min_signals': min_signals_val,
            'selected_signal_source': signal_source,
            'error': None
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse('discovery.html', {
            'request': request,
            'error': f'Error loading discovery data: {str(e)}',
            'symbols': [],
            'stats': {},
            'available_dates': []
        })


@app.get("/ranked", response_class=HTMLResponse)
async def ranked_results(request: Request, date: Optional[str] = Query(None)):
    """Display AI-ranked stock analysis results."""
    conn = get_db_connection(DUCKDB_PATH)
    
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
                COALESCE(f.name, r.symbol) as company_name,
                f.market_cap,
                f.industry
            FROM scanner_data.main.ranked_analysis r
            LEFT JOIN scanner_data.main.fundamental_cache f ON r.symbol = f.symbol
            LEFT JOIN scanner_data.main.ai_analysis_individual ai ON r.symbol = ai.symbol AND r.analysis_date = ai.analysis_date
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


@app.get("/universe", response_class=HTMLResponse)
async def universe_page(
    request: Request,
    search: Optional[str] = Query(None),
    asset_type: Optional[str] = Query(None),
    sector: Optional[str] = Query(None)
):
    """Display all symbols in scanner_data database with latest pricing and fundamental data."""
    try:
        conn = get_db_connection(SCANNER_DATA_PATH)
        
        # Simple query - just get latest date for each symbol
        query = """
            SELECT DISTINCT
                symbol,
                MAX(close) as price,
                MAX(volume) as volume,
                MAX(date) as last_date
            FROM main.daily_cache
            GROUP BY symbol
            ORDER BY symbol
        """
        
        print(f"DEBUG: Executing simple universe query")
        results = conn.execute(query).fetchall()
        print(f"DEBUG: Got {len(results)} results from daily_cache")
        
        symbols = []
        for row in results:
            symbols.append({
                'symbol': row[0],
                'price': row[1],
                'volume': row[2],
                'last_date': str(row[3]) if row[3] else '-',
                'company_name': row[0],  # Use symbol as company name for now
                'sector': '-',
                'industry': '-',
                'market_cap_formatted': '-',
                'pe_ratio': None,
                'dividend_yield': None,
                'beta': None,
                'fifty_two_week_high': None,
                'fifty_two_week_low': None
            })
        
        print(f"DEBUG: Created {len(symbols)} symbol records")
        
        # Get latest date
        latest_date = conn.execute("SELECT MAX(date) FROM main.daily_cache").fetchone()[0]
        
        # Calculate stats
        stats = {
            'total_symbols': len(symbols),
            'stocks': len(symbols),
            'etfs': 0,
            'sectors': 0
        }
        
        conn.close()
        
        return templates.TemplateResponse('universe.html', {
            'request': request,
            'symbols': symbols,
            'total_symbols': len(symbols),
            'latest_date': str(latest_date) if latest_date else 'Unknown',
            'stats': stats,
            'available_sectors': [],
            'search': search,
            'sector': sector,
            'asset_type': asset_type
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR in universe page: {e}")
        return templates.TemplateResponse('universe.html', {
            'request': request,
            'symbols': [],
            'total_symbols': 0,
            'latest_date': 'Unknown',
            'stats': {'total_symbols': 0, 'stocks': 0, 'etfs': 0, 'sectors': 0},
            'available_sectors': [],
            'search': search,
            'sector': sector,
            'asset_type': asset_type,
            'error': str(e)
        })


@app.get("/stats", response_class=HTMLResponse)
async def stats(request: Request):  # email: str = Depends(require_login)  # TODO: Re-enable after OAuth setup
    """Display database statistics landing page."""
    conn = get_db_connection(DUCKDB_PATH)
    
    stats_data = {}
    
    try:
        # Total number of scanner results
        total_results = conn.execute("""
            SELECT COUNT(*) FROM scanner_results
        """).fetchone()[0]
        stats_data['total_results'] = total_results
        
        # Number of unique assets scanned
        unique_assets = conn.execute("""
            SELECT COUNT(DISTINCT symbol) FROM scanner_results
        """).fetchone()[0]
        stats_data['unique_assets'] = unique_assets
        
        # Number of scanners
        num_scanners = conn.execute("""
            SELECT COUNT(DISTINCT scanner_name) FROM scanner_results
        """).fetchone()[0]
        stats_data['num_scanners'] = num_scanners
        
        # Last updated date
        last_updated = conn.execute("""
            SELECT MAX(scan_date) FROM scanner_results
        """).fetchone()[0]
        stats_data['last_updated'] = str(last_updated)[:10] if last_updated else 'N/A'
        
        # Results per scanner
        scanner_breakdown = conn.execute("""
            SELECT scanner_name, COUNT(*) as count
            FROM scanner_results
            GROUP BY scanner_name
            ORDER BY count DESC
        """).fetchall()
        stats_data['scanner_breakdown'] = [(row[0], row[1]) for row in scanner_breakdown]
        
        # Results per date
        date_breakdown = conn.execute("""
            SELECT CAST(scan_date AS DATE) as date, COUNT(*) as count
            FROM scanner_results
            WHERE scan_date IS NOT NULL
            GROUP BY CAST(scan_date AS DATE)
            ORDER BY date DESC
            LIMIT 10
        """).fetchall()
        stats_data['date_breakdown'] = [(str(row[0]), row[1]) for row in date_breakdown]
        
        # Top picked assets (by multiple scanners)
        top_picks = conn.execute("""
            SELECT symbol, COUNT(DISTINCT scanner_name) as scanner_count
            FROM scanner_results
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
            FROM scanner_results
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
    """Display pre-calculated performance analytics for each scanner."""
    
    try:
        # Read from pre-calculated performance_tracking table in scanner_data
        data_conn = get_db_connection(SCANNER_DATA_PATH)
        
        performance_query = """
            SELECT 
                scanner_name,
                total_picks,
                avg_max_gain,
                avg_drawdown,
                avg_current_pnl,
                win_rate,
                best_symbol,
                best_gain,
                best_pick_date,
                best_entry_price,
                worst_symbol,
                worst_drawdown,
                worst_pick_date,
                worst_entry_price,
                top_10_best,
                top_10_worst,
                calculated_at,
                calculation_date
            FROM main.performance_tracking
            WHERE calculation_date = (SELECT MAX(calculation_date) FROM main.performance_tracking)
            ORDER BY avg_max_gain DESC
        """
        
        results = data_conn.execute(performance_query).fetchall()
        data_conn.close()
        
        # Format data for template
        performance_data = []
        for row in results:
            # Parse JSON columns for top 10 lists
            import json
            top_10_best = json.loads(row[14]) if row[14] else []
            top_10_worst = json.loads(row[15]) if row[15] else []
            
            # Get current P&L from top 10 lists (best and worst picks are the first entries)
            best_current_pnl = top_10_best[0]['current_pnl'] if top_10_best else 0
            worst_current_pnl = top_10_worst[0]['current_pnl'] if top_10_worst else 0
            
            performance_data.append({
                'scanner_name': row[0],
                'total_picks': row[1],
                'avg_max_gain': round(row[2], 2),
                'avg_drawdown': round(row[3], 2),
                'avg_current_pnl': round(row[4], 2),
                'win_rate': round(row[5], 1),
                'best_pick': {
                    'symbol': row[6],
                    'max_gain': round(row[7], 2),
                    'scan_date': str(row[8])[:10] if row[8] else 'N/A',
                    'entry_price': row[9] or 0,
                    'current_pnl': round(best_current_pnl, 1)
                },
                'worst_pick': {
                    'symbol': row[10],
                    'max_drawdown': round(row[11], 2),
                    'scan_date': str(row[12])[:10] if row[12] else 'N/A',
                    'entry_price': row[13] or 0,
                    'current_pnl': round(worst_current_pnl, 1)
                },
                'top_10_best': top_10_best,
                'top_10_worst': top_10_worst,
                'calculated_at': row[16]
            })
        
    except Exception as e:
        print(f"Error loading scanner performance: {e}")
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
    conn = get_db_connection(DUCKDB_PATH)
    
    # Get scanner info
    scanner_data = conn.execute("""
        SELECT scanner_name, COUNT(*) as count
        FROM scanner_results
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
    conn = get_db_connection(DUCKDB_PATH)
    
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
        FROM scanner_results
        WHERE scanner_name = ?
    """, [scanner_name]).fetchone()
    
    # Get latest scan date
    latest_date = conn.execute("""
        SELECT CAST(MAX(scan_date) AS DATE)
        FROM scanner_results
        WHERE scanner_name = ?
    """, [scanner_name]).fetchone()
    
    # Get current performance (from latest scan date)
    current_perf = None
    if latest_date and latest_date[0]:
        current_perf = conn.execute("""
            SELECT 
                COUNT(*) as current_total,
                AVG(signal_strength) as current_avg_strength
            FROM scanner_results
            WHERE scanner_name = ? 
            AND CAST(scan_date AS DATE) = ?
        """, [scanner_name, str(latest_date[0])]).fetchone()
    
    # Get recent 30-day performance
    recent_perf = conn.execute("""
        SELECT 
            COUNT(*) as recent_total,
            AVG(signal_strength) as recent_avg_strength
        FROM scanner_results
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
    
    conn = get_db_connection(DUCKDB_PATH)
    
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
            FROM scanner_results
            WHERE symbol = ?
            ORDER BY scan_date DESC, scanner_name
        """, [ticker]).fetchall()
        
        # Get current price if available
        current_price = None
        try:
            current_data = conn.execute("""
                SELECT close, date
                FROM scanner_data.main.daily_cache
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

    # For ticker-only searches, ignore date filter to show all historical results
    if selected_ticker and not pattern:
        selected_scan_date = ''
        print(f"INFO: Ticker-only search - clearing date filter to show all results")
    else:
        # Get latest scan date if none provided (CACHED)
        selected_scan_date = scan_date
        if not selected_scan_date:
            selected_scan_date = get_cached_latest_scan_date()
            if selected_scan_date:
                print(f"INFO: Auto-selected latest scan date: {selected_scan_date}")
    
    # Get list of all available tickers for autocomplete (CACHED - 10 min)
    available_tickers = get_cached_ticker_list()
    
    # Don't auto-select a scanner - show all by default
    pattern = pattern if pattern and pattern != '' else False
    
    # Get symbol metadata (CACHED - 30 min)
    all_symbol_metadata = get_cached_symbol_metadata()
    
    # Filter metadata based on market cap and sector
    symbol_metadata = {}
    min_cap = 0
    if min_market_cap:
        # Parse market cap values like "1B", "100M", "500M", "5B", "10B"
        cap_value = min_market_cap.upper()
        if 'B' in cap_value:
            min_cap = float(cap_value.replace('B', '')) * 1_000_000_000
        elif 'M' in cap_value:
            min_cap = float(cap_value.replace('M', '')) * 1_000_000
        else:
            min_cap = float(cap_value)
    
    # Apply filters to cached metadata
    cached_metadata = get_cached_symbol_metadata()
    for symbol, meta in cached_metadata.items():
        # Apply sector filter
        if sector_filter and sector_filter != 'All':
            if meta.get('sector') != sector_filter:
                continue
        
        # Apply market cap filter
        if min_market_cap and meta.get('market_cap'):
            try:
                cap_str = str(meta['market_cap']).upper()
                if 'T' in cap_str:
                    cap_num = float(cap_str.replace('T', '')) * 1_000_000_000_000
                elif 'B' in cap_str:
                    cap_num = float(cap_str.replace('B', '')) * 1_000_000_000
                elif 'M' in cap_str:
                    cap_num = float(cap_str.replace('M', '')) * 1_000_000
                else:
                    cap_num = float(cap_str)
                if cap_num < min_cap:
                    continue
            except Exception:
                continue
        elif min_market_cap and not meta.get('market_cap'):
            continue
        
        symbol_metadata[symbol] = {
            'company': meta.get('company', symbol),
            'market_cap': format_market_cap(meta.get('market_cap')),
            'sector': meta.get('sector'),
            'industry': meta.get('industry')
        }

    if pattern:
        # Use pattern name directly as scanner name
        print(f"Loading scanner results for: {pattern}")
        
        # Get a connection for this request (fresh for MotherDuck)
        conn = get_db_connection(DUCKDB_PATH)
        
        # Read pre-calculated scanner results from database
        # Build query with optional date and ticker filters
        # LIMIT to 50 results to prevent timeout on Render (30s limit)
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
            FROM scanner_results
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
        
        # Order by signal strength (strongest first) and limit results to prevent timeout and memory issues
        # Reduced to 25 to stay within 512MB memory limit on Render free tier
        scanner_query += ' ORDER BY signal_strength DESC LIMIT 25'

        scanner_dict = {}
        # Get top 50 results ordered by strength to prevent timeout
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
            print(f'Found {len(scanner_dict)} results for {pattern} (top 25 by strength)')
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
                    FROM scanner_data.main.daily_cache dc
                    INNER JOIN (
                        SELECT symbol, MAX(date) as max_date
                        FROM scanner_data.main.daily_cache
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
                    FROM scanner_results
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
        
        # Fetch recent scanner confirmations for each symbol (limit to last 30 days to save memory)
        confirmations_dict = {}
        if symbols_list:
            try:
                placeholders = ','.join(['?' for _ in symbols_list])
                # MEMORY FIX: Limit to last 30 days and max 5 confirmations per symbol
                confirmations_query = f'''
                    SELECT symbol, scanner_name, scan_date, signal_strength
                    FROM scanner_results
                    WHERE symbol IN ({placeholders})
                    AND scanner_name != ?
                    AND scan_date >= CURRENT_DATE - INTERVAL 30 DAY
                    ORDER BY symbol, scan_date DESC, scanner_name
                    LIMIT 200
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
                    # Get all signals for chart display (60 day history)
                    options_query = f'''
                        SELECT underlying_symbol, signal_date, signal_type, 
                               signal_strength, confidence_score, strike, dte,
                               premium_spent, notes, direction
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
                        direction = row[9] if len(row) > 9 else None
                        
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
                            'notes': notes,
                            'direction': direction
                        })
                    
                    options_conn.close()
                    print(f'Loaded options signals for {len(options_signals_dict)} symbols')
            except Exception as e:
                print(f'Options signals query failed: {e}')
                options_signals_dict = {}
        
        # Fetch dark pool signals for each symbol (from options_data database)
        darkpool_signals_dict = {}
        if symbols_list and OPTIONS_DUCKDB_PATH:
            try:
                dp_conn = get_options_db_connection()
                if dp_conn:
                    placeholders = ','.join(['?' for _ in symbols_list])
                    # Get all signals for chart display (60 day history)
                    dp_query = f'''
                        SELECT ticker, signal_date, signal_type, 
                               signal_strength, confidence_score, direction,
                               dp_volume, dp_premium, avg_price,
                               sell_volume, buy_volume, buy_sell_ratio,
                               block_count, avg_block_size, consecutive_days, notes
                        FROM darkpool_signals
                        WHERE ticker IN ({placeholders})
                        ORDER BY ticker, signal_date DESC
                    '''
                    dp_results = dp_conn.execute(
                        dp_query, symbols_list
                    ).fetchall()
                    
                    for row in dp_results:
                        sym = row[0]
                        sig_date = str(row[1]) if row[1] else ''
                        sig_type = row[2]
                        sig_strength = row[3]
                        confidence = row[4]
                        direction = row[5]
                        dp_volume = row[6]
                        dp_premium = row[7]
                        avg_price = row[8]
                        buy_vol = row[9]
                        sell_vol = row[10]
                        buy_sell_ratio = row[11]
                        block_count = row[12]
                        avg_block_size = row[13]
                        consecutive_days = row[14]
                        notes = row[15]
                        
                        if sym not in darkpool_signals_dict:
                            darkpool_signals_dict[sym] = []
                        darkpool_signals_dict[sym].append({
                            'date': sig_date,
                            'signal_type': sig_type,
                            'strength': sig_strength,
                            'confidence': confidence,
                            'direction': direction,
                            'dp_volume': dp_volume,
                            'dp_premium': dp_premium,
                            'avg_price': avg_price,
                            'buy_volume': buy_vol,
                            'sell_volume': sell_vol,
                            'buy_sell_ratio': buy_sell_ratio,
                            'block_count': block_count,
                            'avg_block_size': avg_block_size,
                            'consecutive_days': consecutive_days,
                            'notes': notes
                        })
                    
                    dp_conn.close()
                    print(f'Loaded dark pool signals for {len(darkpool_signals_dict)} symbols')
                    
            except Exception as e:
                print(f'Dark pool signals query failed: {e}')
                darkpool_signals_dict = {}
        
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
                        # Get exactly one row per symbol (latest before scan date)
                        # Uses QUALIFY to get only latest row per symbol
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
                            QUALIFY ROW_NUMBER() OVER (PARTITION BY underlying_symbol ORDER BY scan_date DESC) = 1
                        '''
                        walls_results = options_conn.execute(
                            walls_query, symbols_list + [selected_scan_date]
                        ).fetchall()
                    else:
                        # No date selected - get exactly one row per symbol (latest)
                        # Uses QUALIFY to get only latest row per symbol
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
                            QUALIFY ROW_NUMBER() OVER (PARTITION BY underlying_symbol ORDER BY scan_date DESC) = 1
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
        
        # Fetch fundamental quality scores for each symbol
        fund_quality_dict = {}
        if symbols_list:
            try:
                fund_conn = get_db_connection(DUCKDB_PATH)
                if fund_conn:
                    placeholders = ','.join(['?' for _ in symbols_list])
                    fund_query = f'''
                        SELECT symbol, fund_score, bar_blocks, bar_bucket, dot_state,
                               score_components, computed_at
                        FROM scanner_data.main.fundamental_quality_scores
                        WHERE symbol IN ({placeholders})
                    '''
                    fund_results = fund_conn.execute(fund_query, symbols_list).fetchall()
                    
                    for row in fund_results:
                        sym = row[0]
                        # Parse score_components JSON for tooltip data
                        raw_inputs = {}
                        if row[5]:
                            import json
                            try:
                                components = json.loads(row[5]) if isinstance(row[5], str) else row[5]
                                raw_inputs = components.get('raw_inputs', {})
                            except:
                                pass
                        
                        fund_quality_dict[sym] = {
                            'fund_score': row[1],
                            'bar_blocks': row[2],
                            'bar_bucket': row[3],
                            'dot_state': row[4],
                            'computed_at': str(row[6]) if row[6] else None,
                            'operating_margin': raw_inputs.get('operating_margin'),
                            'return_on_equity': raw_inputs.get('return_on_equity'),
                            'profit_margin': raw_inputs.get('profit_margin'),
                            'quarterly_earnings_growth': raw_inputs.get('quarterly_earnings_growth'),
                            'pe_ratio': raw_inputs.get('pe_ratio')
                        }
                    
                    fund_conn.close()
                    print(f'Loaded fundamental quality scores for {len(fund_quality_dict)} symbols')
            except Exception as e:
                print(f'Fundamental quality query failed: {e}')
                fund_quality_dict = {}
        
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
                        
                        # Add dark pool signals for this symbol
                        if symbol in darkpool_signals_dict:
                            stocks[symbol][f'{pattern}_darkpool_signals'] = darkpool_signals_dict[symbol]
                        else:
                            stocks[symbol][f'{pattern}_darkpool_signals'] = []
                        
                        # Add options walls for this symbol
                        if symbol in options_walls_dict:
                            stocks[symbol][f'{pattern}_options_walls'] = options_walls_dict[symbol]
                            
                            # Calculate OMS (Options Market Share)
                            # OMS = Max(OI at key strikes) × 100 ÷ Equity Volume
                            walls = options_walls_dict[symbol]
                            volume = stocks[symbol].get(f'{pattern}_volume', 0)
                            
                            if volume and volume > 0:
                                # Get max OI from all walls (calls and puts)
                                oi_values = [
                                    walls.get('call_wall_1', {}).get('oi') or 0,
                                    walls.get('call_wall_2', {}).get('oi') or 0,
                                    walls.get('call_wall_3', {}).get('oi') or 0,
                                    walls.get('put_wall_1', {}).get('oi') or 0,
                                    walls.get('put_wall_2', {}).get('oi') or 0,
                                    walls.get('put_wall_3', {}).get('oi') or 0,
                                ]
                                max_oi = max(oi_values)
                                
                                # OMS = (max_oi * 100) / volume
                                # Each options contract controls 100 shares
                                if max_oi > 0:
                                    oms = (max_oi * 100) / volume
                                    stocks[symbol][f'{pattern}_oms'] = round(oms, 2)
                                else:
                                    stocks[symbol][f'{pattern}_oms'] = None
                            else:
                                stocks[symbol][f'{pattern}_oms'] = None
                        else:
                            stocks[symbol][f'{pattern}_options_walls'] = None
                            stocks[symbol][f'{pattern}_oms'] = None
                        
                        # Add fundamental quality scores
                        if symbol in fund_quality_dict:
                            stocks[symbol]['fund_quality'] = fund_quality_dict[symbol]
                        else:
                            stocks[symbol]['fund_quality'] = None
                        
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
        
        # Get a connection for ticker search
        conn = get_db_connection(DUCKDB_PATH)
        
        try:
            # Query all scanners that have this ticker
            ticker_query = '''
                SELECT scanner_name,
                       signal_type,
                       COALESCE(signal_strength, 75) as signal_strength,
                       COALESCE(setup_stage, 'N/A') as quality_placeholder,
                       scan_date,
                       metadata
                FROM scanner_results
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
    
    # Get available sectors for dropdown (cached)
    available_sectors = []
    try:
        cached_metadata = get_cached_symbol_metadata()
        sectors_set = set()
        for meta in cached_metadata.values():
            if meta.get('sector'):
                sectors_set.add(meta['sector'])
        available_sectors = sorted(list(sectors_set))
    except Exception as e:
        print(f'Could not get sectors: {e}')
    
    # Get available pre-calculated scanners from database (cached)
    available_scanners = get_cached_available_scanners(selected_scan_date)
    
    # Get scanner names from database for the dropdown with counts
    try:
        # Get a connection for dropdown data
        dropdown_conn = get_db_connection(DUCKDB_PATH)
        
        # Get scanner counts based on selected date
        if selected_scan_date:
            # Use the selected date
            date_to_use = selected_scan_date
            print(f"INFO: Using selected date: {date_to_use}")
        else:
            # Get the latest scan date
            latest_date_result = dropdown_conn.execute("""
                SELECT MAX(CAST(scan_date AS DATE))
                FROM scanner_results
            """).fetchone()
            date_to_use = str(latest_date_result[0]) if latest_date_result and latest_date_result[0] else None
            print(f"INFO: Using latest scan date: {date_to_use}")
        
        if date_to_use:
            scanner_counts_query = """
                SELECT scanner_name, COUNT(*) as count
                FROM scanner_results
                WHERE CAST(scan_date AS DATE) = ?
                GROUP BY scanner_name
                ORDER BY scanner_name
            """
            scanner_counts = dropdown_conn.execute(scanner_counts_query, [date_to_use]).fetchall()
            print(f"INFO: Found {len(scanner_counts)} scanners for date {date_to_use}")
        else:
            # Fallback if no date available
            print("WARNING: No date available, getting all scanner counts")
            scanner_counts_query = """
                SELECT scanner_name, COUNT(*) as count
                FROM scanner_results
                GROUP BY scanner_name
                ORDER BY scanner_name
            """
            scanner_counts = dropdown_conn.execute(scanner_counts_query).fetchall()
        
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
        dates = dropdown_conn.execute("""
            SELECT CAST(scan_date AS DATE) as date, COUNT(*) as count
            FROM scanner_results
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
        latest_timestamp = dropdown_conn.execute("""
            SELECT MAX(scan_date)
            FROM scanner_results
        """).fetchone()
        if latest_timestamp and latest_timestamp[0]:
            # Format as datetime string with date and time
            ts = latest_timestamp[0]
            if hasattr(ts, 'strftime'):
                last_updated = ts.strftime('%Y-%m-%d %H:%M')
            else:
                # If it's a string, try to parse and format it
                ts_str = str(ts)
                last_updated = ts_str[:16] if len(ts_str) >= 16 else ts_str
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
            FROM scanner_results
            WHERE scan_date >= CURRENT_DATE - INTERVAL '30 days'
                AND scanner_name NOT IN ('bullish', 'Candlestick Bullish', 'Fundamental Swing')
            GROUP BY CAST(scan_date AS DATE), scanner_name
            ORDER BY date ASC, scanner_name
        """
        history_results = dropdown_conn.execute(history_query).fetchall()
        
        # Get unique symbols per day
        unique_symbols_query = """
            SELECT 
                CAST(scan_date AS DATE) as date,
                COUNT(DISTINCT symbol) as unique_symbols
            FROM scanner_results
            WHERE scan_date >= CURRENT_DATE - INTERVAL '30 days'
                AND scanner_name NOT IN ('bullish', 'Candlestick Bullish', 'Fundamental Swing')
            GROUP BY CAST(scan_date AS DATE)
            ORDER BY date ASC
        """
        unique_symbols_results = dropdown_conn.execute(unique_symbols_query).fetchall()
        unique_symbols_by_date = {str(row[0]): row[1] for row in unique_symbols_results}
        
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
            'scanners': {},
            'unique_symbols': [unique_symbols_by_date.get(date, 0) for date in dates]
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
        scanner_history = {'dates': [], 'scanners': {}, 'unique_symbols': []}

    # If a ticker is selected, reorder stocks dict to put it first
    if selected_ticker and selected_ticker in stocks:
        # Create new ordered dict with selected ticker first
        ordered_stocks = {selected_ticker: stocks[selected_ticker]}
        # Add all other stocks
        for symbol in stocks:
            if symbol != selected_ticker:
                ordered_stocks[symbol] = stocks[symbol]
        stocks = ordered_stocks

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
