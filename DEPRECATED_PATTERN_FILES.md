# Deprecated Pattern Detection Files

This app has been converted from an active scanner to a **VIEW-ONLY** app for MotherDuck results.

The following files are no longer used and contain pattern detection logic:

## Scanner Files (DEPRECATED)
- `qullamaggie_scanner.py` - Qullamaggie breakout detection
- `momentum_burst_scanner.py` - Momentum burst pattern detection  
- `supertrend_scanner.py` - SuperTrend indicator patterns
- `explosive_volume_scanner.py` - Volume explosion detection
- `save_scanner_results_to_db.py` - Scanner execution and database saving

## Pattern Detection Files (DEPRECATED)  
- `patterns.py` - Candlestick pattern definitions
- `custom_patterns.py` - Custom chart patterns (cup/handle, flags, etc.)
- `pattern_scoring.py` - Pattern strength scoring
- `pattern_detect.py` - Pattern detection testing (already disabled)

## Other Analysis Files (DEPRECATED)
- `chartlib.py` - Chart analysis utilities
- `cleanup_delisted.py` - Database cleanup
- `bulk_scan.py` - Bulk scanning operations
- `fix_motherduck_remote.py` - Database fixes
- `fix_motherduck_schema.py` - Schema fixes
- `upload_to_motherduck.py` - Data upload utilities

## Current Active Files
- `app.py` - FastAPI web viewer (ACTIVE)
- `requirements.txt` - Dependencies (ACTIVE)
- `templates/` - HTML templates (ACTIVE)
- `datasets/` - Static data (if needed)

## What This App Does Now
- Displays pre-computed scanner results from MotherDuck
- Provides filtering and search capabilities
- Shows statistics and documentation
- NO pattern detection or API calls