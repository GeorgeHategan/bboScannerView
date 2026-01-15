# Testing Guide

## Overview
This project uses automated tests to verify critical functionality and prevent regressions like missing confirmations, broken Beta display, etc.

## Test Types

### 1. Critical Feature Tests (`test_critical_features.py`)
Tests core functionality:
- ✅ Homepage loads
- ✅ Scanner results display  
- ✅ Confirmations block appears
- ✅ Dark pool signals load
- ✅ Options signals load
- ✅ Fundamental data (Beta, sector, industry)
- ✅ Fund quality scores
- ✅ News sentiment scores
- ✅ Data integrity checks

**Run locally:**
```bash
pytest test_critical_features.py -v
```

### 2. E2E Browser Tests (`test_e2e_browser.py`)
Tests UI elements with real browser:
- ✅ Beta displays and tooltips
- ✅ Confirmations block visible
- ✅ Dark pool charts render
- ✅ Fund quality tooltips work
- ✅ Timeline displays
- ✅ Page navigation

**Run locally:**
```bash
# Install Playwright first
pip install playwright
playwright install chromium

# Start app
python app.py

# Run tests (in another terminal)
APP_URL=http://localhost:5000 pytest test_e2e_browser.py -v
```

### 3. Schema Reference Test

Ensures database schema references don't have duplicate 'main' prefixes that would break queries.

```bash
python3 test_schema_refs.py
```

**What it checks:**
- ❌ `scanner_data.main.main.*` (should be `scanner_data.main.*`)
- ❌ `scanner_data.main.main.main.*` (should be `scanner_data.main.*`)
- ✅ `scanner_data.main.*` (correct)

**Exit codes:**
- `0` = Pass (no issues found)
- `1` = Fail (duplicate references found)

## Running Tests

### Run All Tests
```bash
pytest -v
```

### Run Specific Test
```bash
pytest test_critical_features.py::TestCriticalFeatures::test_confirmations_appear_in_ui -v
```

## CI/CD Integration

Tests run automatically on every push to `main` branch via GitHub Actions.

See `.github/workflows/test.yml` for configuration.

### Required Secrets
Add to GitHub repository settings → Secrets:
- `MOTHERDUCK_TOKEN`
- `OPTIONS_MOTHERDUCK_TOKEN`

### Running All Tests

```bash
# Run schema test
python3 test_schema_refs.py

# Add more tests here as they are created
```

## Before Committing

It's good practice to run tests before committing changes that modify database queries:

```bash
python3 test_schema_refs.py && git add . && git commit -m "Your message"
```

This ensures you don't accidentally introduce schema reference bugs.
