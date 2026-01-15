"""
Critical Feature Tests - Run these after each deployment
Tests key functionality to catch regressions like missing confirmations, dark pool signals, etc.
"""
import pytest
import sys
from app import app, get_db_connection, DUCKDB_PATH, SCANNER_DATA_PATH
import duckdb

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def db():
    """Get database connection"""
    return get_db_connection(DUCKDB_PATH)

@pytest.fixture
def scanner_db():
    """Get scanner_data database connection"""
    return get_db_connection(SCANNER_DATA_PATH)


class TestCriticalFeatures:
    """Test critical features that must work"""
    
    def test_homepage_loads(self, client):
        """Verify homepage loads successfully"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Scanner' in response.data or b'scanner' in response.data
    
    def test_scanner_results_load(self, client):
        """Verify scanner results page loads"""
        # Test with a known scanner
        response = client.get('/?scanner=supertrend')
        assert response.status_code == 200
        # Should have results table or empty state message
        assert b'Vol:' in response.data or b'No results' in response.data
    
    def test_confirmations_data_available(self, db):
        """Verify confirmations query works and returns data"""
        try:
            # Get a sample symbol that should have multiple scanner detections
            result = db.execute("""
                SELECT symbol, COUNT(DISTINCT scanner_name) as scanner_count
                FROM scanner_results.scanner_results
                WHERE scan_date >= CURRENT_DATE - INTERVAL 7 DAY
                GROUP BY symbol
                HAVING COUNT(DISTINCT scanner_name) > 1
                ORDER BY scanner_count DESC
                LIMIT 1
            """).fetchone()
            
            if result:
                symbol, scanner_count = result
                print(f"Test symbol {symbol} has {scanner_count} scanners")
                
                # Verify confirmations query works
                confirmations = db.execute("""
                    SELECT scanner_name, scan_date
                    FROM scanner_results.scanner_results
                    WHERE symbol = ?
                    AND scan_date >= CURRENT_DATE - INTERVAL 30 DAY
                    ORDER BY scan_date DESC
                """, [symbol]).fetchall()
                
                assert len(confirmations) > 0, f"No confirmations found for {symbol}"
                print(f"✓ Found {len(confirmations)} confirmations for {symbol}")
            else:
                pytest.skip("No multi-scanner symbols found in recent data")
        except Exception as e:
            pytest.fail(f"Confirmations query failed: {e}")
    
    def test_confirmations_appear_in_ui(self, client, db):
        """Verify confirmations block appears when symbol has multiple scanner detections"""
        # Find a symbol with multiple scanners
        result = db.execute("""
            SELECT symbol, scanner_name
            FROM scanner_results.scanner_results
            WHERE scan_date >= CURRENT_DATE - INTERVAL 7 DAY
            GROUP BY symbol, scanner_name
            HAVING COUNT(*) > 1
            LIMIT 1
        """).fetchone()
        
        if result:
            symbol, scanner = result
            response = client.get(f'/?scanner={scanner}&ticker={symbol}')
            assert response.status_code == 200
            
            # Check for confirmations block
            assert b'Confirmed by other scanners' in response.data, \
                f"Confirmations block missing for {symbol} in {scanner}"
            print(f"✓ Confirmations block present for {symbol}")
        else:
            pytest.skip("No suitable test symbol found")
    
    def test_dark_pool_signals_data(self, client):
        """Verify dark pool signals load"""
        response = client.get('/darkpool-signals')
        assert response.status_code == 200
        assert b'Dark Pool' in response.data
    
    def test_options_signals_data(self, client):
        """Verify options signals load"""
        response = client.get('/options-signals')
        assert response.status_code == 200
        assert b'Options' in response.data
    
    def test_fundamental_data_loads(self, scanner_db):
        """Verify fundamental data (Beta, sector, industry) is accessible"""
        result = scanner_db.execute("""
            SELECT symbol, sector, industry, beta, market_cap
            FROM scanner_data.main.fundamental_cache
            WHERE sector IS NOT NULL
            LIMIT 5
        """).fetchall()
        
        assert len(result) > 0, "No fundamental data found"
        
        for row in result:
            symbol, sector, industry, beta, market_cap = row[:5]
            print(f"✓ {symbol}: Sector={sector}, Industry={industry}, Beta={beta}")
            assert sector is not None, f"{symbol} missing sector"
    
    def test_beta_in_metadata(self, scanner_db):
        """Verify Beta is included in cached metadata"""
        result = scanner_db.execute("""
            SELECT symbol, beta
            FROM scanner_data.main.fundamental_cache
            WHERE beta IS NOT NULL
            LIMIT 10
        """).fetchall()
        
        assert len(result) > 0, "No symbols with Beta found"
        print(f"✓ Found {len(result)} symbols with Beta data")
    
    def test_fund_quality_scores_exist(self, scanner_db):
        """Verify fundamental quality scores are available"""
        result = scanner_db.execute("""
            SELECT symbol, fund_score, bar_blocks, bar_bucket, dot_state
            FROM scanner_data.main.fundamental_quality_scores
            WHERE score_version = 'fq_v2'
            LIMIT 5
        """).fetchall()
        
        assert len(result) > 0, "No fundamental quality scores found"
        print(f"✓ Found {len(result)} symbols with FQ scores")
    
    def test_news_sentiment_scores_exist(self, scanner_db):
        """Verify news sentiment scores are available"""
        result = scanner_db.execute("""
            SELECT symbol, news_score, bar_direction, article_count_5d
            FROM scanner_data.main.news_sentiment_pressure_scores
            WHERE score_version = 'nsp_v1'
            LIMIT 5
        """).fetchall()
        
        assert len(result) > 0, "No news sentiment scores found"
        print(f"✓ Found {len(result)} symbols with news sentiment")
    
    def test_focus_list_loads(self, client):
        """Verify focus list page loads"""
        response = client.get('/focus-list')
        assert response.status_code == 200
        assert b'Focus List' in response.data or b'focus' in response.data
    
    def test_scanner_stats_accessible(self, db):
        """Verify scanner stats are calculable"""
        result = db.execute("""
            SELECT scanner_name, COUNT(*) as count
            FROM scanner_results.scanner_results
            WHERE scan_date >= CURRENT_DATE - INTERVAL 7 DAY
            GROUP BY scanner_name
            ORDER BY count DESC
            LIMIT 5
        """).fetchall()
        
        assert len(result) > 0, "No recent scanner results found"
        for scanner, count in result:
            print(f"✓ {scanner}: {count} results in last 7 days")


class TestDataIntegrity:
    """Test data integrity and consistency"""
    
    def test_no_duplicate_scanner_results(self, db):
        """Check for duplicate scanner results on same day"""
        duplicates = db.execute("""
            SELECT symbol, scanner_name, scan_date, COUNT(*) as dup_count
            FROM scanner_results.scanner_results
            WHERE scan_date >= CURRENT_DATE - INTERVAL 7 DAY
            GROUP BY symbol, scanner_name, scan_date
            HAVING COUNT(*) > 1
        """).fetchall()
        
        if duplicates:
            print("WARNING: Found duplicate scanner results:")
            for dup in duplicates[:5]:
                print(f"  {dup}")
        # Don't fail - just warn
        assert len(duplicates) < 100, "Excessive duplicates found"
    
    def test_volume_data_recent(self, scanner_db):
        """Verify daily_cache has recent data"""
        result = scanner_db.execute("""
            SELECT MAX(date) as latest_date, COUNT(DISTINCT symbol) as symbol_count
            FROM scanner_data.main.daily_cache
        """).fetchone()
        
        latest_date, symbol_count = result
        print(f"✓ Latest daily_cache data: {latest_date}, {symbol_count} symbols")
        assert symbol_count > 100, "Too few symbols in daily_cache"


if __name__ == '__main__':
    # Run tests and output results
    pytest.main([__file__, '-v', '--tb=short'])
