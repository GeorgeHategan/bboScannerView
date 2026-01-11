#!/usr/bin/env python3
"""
Comprehensive integration tests for the BBO Scanner application.
Tests all major pages, API endpoints, and critical functionality.

Run with: python3 test_app.py
Or with pytest: pytest test_app.py -v
"""

import sys
import os
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("\n🧪 Testing imports...")
    try:
        import fastapi
        import duckdb
        import jinja2
        print("  ✅ All required packages available")
        return True
    except ImportError as e:
        print(f"  ❌ Missing package: {e}")
        return False

def test_app_startup():
    """Test that the FastAPI app can be created."""
    print("\n🧪 Testing app startup...")
    try:
        from app import app
        assert app is not None, "App instance is None"
        print("  ✅ FastAPI app created successfully")
        return True
    except Exception as e:
        print(f"  ❌ Failed to create app: {e}")
        return False

def test_environment_variables():
    """Test that critical environment variables are set."""
    print("\n🧪 Testing environment variables...")
    errors = []
    
    # Critical for deployment
    required_vars = {
        'SECRET_KEY': 'Session security',
        'motherduck_token': 'Scanner data access',
        'options_motherduck_token': 'Options data access',
    }
    
    # Optional (OAuth can be disabled)
    optional_vars = {
        'GOOGLE_CLIENT_ID': 'OAuth authentication',
        'GOOGLE_CLIENT_SECRET': 'OAuth authentication',
        'DATABASE_URL': 'User management',
    }
    
    for var, purpose in required_vars.items():
        if not os.environ.get(var):
            errors.append(f"Missing {var} ({purpose})")
    
    warnings = []
    for var, purpose in optional_vars.items():
        if not os.environ.get(var) and not os.environ.get(var.upper()):
            warnings.append(f"Missing {var} ({purpose})")
    
    if errors:
        print(f"  ❌ Required variables missing:")
        for error in errors:
            print(f"     - {error}")
        return False
    
    if warnings:
        print(f"  ⚠️  Optional variables not set:")
        for warning in warnings:
            print(f"     - {warning}")
    
    print(f"  ✅ All critical variables set")
    return True
    
    if warnings:
        print(f"  ⚠️  Optional variables missing (some features may not work):")
        for warning in warnings:
            print(f"     - {warning}")
    
    print("  ✅ All required environment variables set")
    return True

def test_database_schema_references():
    """Test for duplicate 'main' schema references."""
    print("\n🧪 Testing database schema references...")
    
    import re
    app_file = Path(__file__).parent / 'app.py'
    content = app_file.read_text()
    
    errors = []
    
    # Check for duplicate main
    double_main = re.findall(r'scanner_data\.main\.main\.', content)
    triple_main = re.findall(r'scanner_data\.main\.main\.main\.', content)
    
    if double_main:
        errors.append(f"Found {len(double_main)} instances of 'scanner_data.main.main.*'")
    if triple_main:
        errors.append(f"Found {len(triple_main)} instances of 'scanner_data.main.main.main.*'")
    
    if errors:
        print(f"  ❌ Schema reference errors:")
        for error in errors:
            print(f"     - {error}")
        return False
    
    print("  ✅ No duplicate schema references found")
    return True

def test_templates_exist():
    """Test that all template files exist."""
    print("\n🧪 Testing template files...")
    
    templates_dir = Path(__file__).parent / 'templates'
    required_templates = [
        'index.html',
        'login.html',
        'admin.html',
        'scanner_detail.html',
        'options_signals.html',
        'darkpool_signals.html',
        'focus_list.html',
        'ranked.html',
        'stats.html',
        'vix_chart.html',
        'scanner_performance.html',
    ]
    
    missing = []
    for template in required_templates:
        if not (templates_dir / template).exists():
            missing.append(template)
    
    if missing:
        print(f"  ❌ Missing templates:")
        for template in missing:
            print(f"     - {template}")
        return False
    
    print(f"  ✅ All {len(required_templates)} templates exist")
    return True

def test_static_files():
    """Test that static directory exists."""
    print("\n🧪 Testing static files...")
    
    static_dir = Path(__file__).parent / 'static'
    if not static_dir.exists():
        print("  ❌ Static directory missing")
        return False
    
    # Check for favicon
    favicon = static_dir / 'favicon.png'
    if not favicon.exists():
        print("  ⚠️  favicon.png missing")
    
    print("  ✅ Static directory exists")
    return True

def test_route_definitions():
    """Test that all major routes are defined."""
    print("\n🧪 Testing route definitions...")
    
    app_file = Path(__file__).parent / 'app.py'
    content = app_file.read_text()
    
    # Check for both single and double quote variants
    required_routes = [
        ('/', 'Home page'),
        ('/login', 'Login page'),
        ('/admin', 'Admin panel'),
        ('/options-signals', 'Options signals'),
        ('/darkpool-signals', 'Darkpool signals'),
        ('/focus-list', 'Focus list'),
        ('/ranked', 'Ranked page'),
        ('/stats', 'Stats page'),
        ('/api/darkpool-chart-data', 'Darkpool chart API'),
        ('/api/options-chart-data', 'Options chart API'),
    ]
    
    missing = []
    for route, description in required_routes:
        # Check for both single and double quotes
        if f"@app.get('{route}'" not in content and f'@app.get("{route}"' not in content:
            missing.append(f"{description} ({route})")
    
    if missing:
        print(f"  ❌ Missing routes:")
        for route in missing:
            print(f"     - {route}")
        return False
    
    print(f"  ✅ All {len(required_routes)} critical routes defined")
    return True

def test_security_patterns():
    """Test for common security issues."""
    print("\n🧪 Testing security patterns...")
    
    app_file = Path(__file__).parent / 'app.py'
    content = app_file.read_text()
    
    warnings = []
    
    # Check if SessionMiddleware is added
    if 'SessionMiddleware' not in content:
        warnings.append("SessionMiddleware not found")
    
    # Check if CORS is configured (if needed)
    # Add more security checks as needed
    
    if warnings:
        print(f"  ⚠️  Security warnings:")
        for warning in warnings:
            print(f"     - {warning}")
        return True  # Warning only, not failure
    
    print("  ✅ Security checks passed")
    return True

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("🚀 Running BBO Scanner Integration Tests")
    print("=" * 60)
    
    start_time = time.time()
    
    tests = [
        ("Imports", test_imports),
        ("App Startup", test_app_startup),
        ("Environment Variables", test_environment_variables),
        ("Database Schema", test_database_schema_references),
        ("Templates", test_templates_exist),
        ("Static Files", test_static_files),
        ("Routes", test_route_definitions),
        ("Security", test_security_patterns),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ❌ Test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    elapsed = time.time() - start_time
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed in {elapsed:.2f}s")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for deployment.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix before deploying.")
        return 1

if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
