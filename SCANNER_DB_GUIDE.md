# Scanner Results Database System

## Overview
The web app now reads **pre-calculated** scanner results from the database instead of calculating them on-the-fly. This makes the app **much faster** and reduces server load.

## How It Works

### 1. **Run Scanners and Save to Database**
Execute this script to scan all stocks and save results to database:

```bash
python save_scanner_results_to_db.py
```

**What it does:**
- Scans all stocks in the database
- Runs 11 different scanners (Qullamaggie, Momentum Burst, Supertrend, Volume)
- Saves results to `scanner_data.scanner_results` table
- Takes ~5-10 minutes for ~2000 stocks

**Output:**
```
✅ Scan complete! Found 507 signals

📈 Summary by scanner:
  QULLAMAGGIE_BREAKOUT: 37 signals
  MOMENTUM_BURST_1D: 37 signals
  EXPLOSIVE_VOLUME_10X: 4 signals
  ...
```

### 2. **Web App Reads from Database**
The Flask app (`app.py`) now:
- Queries `scanner_data.scanner_results` table
- Filters by scanner name and date
- Displays results instantly (no calculation needed)

### 3. **Database Schema**

**Table: `scanner_data.scanner_results`**
```sql
symbol VARCHAR          -- Stock symbol (e.g., "AAPL")
scanner_name VARCHAR    -- Scanner name (e.g., "QULLAMAGGIE_BREAKOUT")
signal VARCHAR          -- Signal type ("bullish", "bearish", etc.)
strength DOUBLE         -- Pattern strength score (0-100)
quality VARCHAR         -- Quality rating ("strong", "good", "moderate", etc.)
scan_date DATE         -- Date of scan
```

## Supported Scanners

These scanners are pre-calculated and stored in the database:

- `QULLAMAGGIE_BREAKOUT` - Mark Minervini style breakouts
- `MOMENTUM_BURST_1D` - 1-day momentum burst
- `MOMENTUM_BURST_3D` - 3-day momentum burst
- `MOMENTUM_BURST_5D` - 5-day momentum burst
- `SUPERTREND_BULLISH` - Supertrend bullish signals
- `SUPERTREND_FRESH` - Fresh supertrend signals
- `SUPERTREND_RECENT` - Recent supertrend signals
- `EXPLOSIVE_VOLUME_3X` - 3x volume spike
- `EXPLOSIVE_VOLUME_5X` - 5x volume spike
- `EXPLOSIVE_VOLUME_10X` - 10x volume spike
- `VOLUME_SURGE_WITH_PRICE` - Volume surge with price movement

## Setup for Production

### Option 1: Manual Daily Update
Run the scanner script once per day:
```bash
# Add to crontab for daily 6 AM run
0 6 * * * cd /path/to/app && python save_scanner_results_to_db.py
```

### Option 2: GitHub Actions (Automated)
Create `.github/workflows/daily_scan.yml`:
```yaml
name: Daily Scanner Update
on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC
  workflow_dispatch:  # Manual trigger

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run scanner
        env:
          DUCKDB_PATH: ${{ secrets.DUCKDB_PATH }}
        run: python save_scanner_results_to_db.py
```

### Option 3: Render Cron Job
In Render dashboard:
1. Add a "Cron Job" service
2. Schedule: `0 6 * * *` (daily at 6 AM)
3. Command: `python save_scanner_results_to_db.py`
4. Environment: Same as web service

## Benefits

✅ **Faster**: Web app loads instantly (no calculation)
✅ **Scalable**: Handles many concurrent users
✅ **Consistent**: All users see same results
✅ **Scheduled**: Run once, serve many times
✅ **Cost-effective**: Reduces compute on web server

## Troubleshooting

### "Table scanner_results does not exist"
Run the scanner script at least once to create the table:
```bash
python save_scanner_results_to_db.py
```

### "No results found"
Check if scanner results are recent:
```sql
SELECT MAX(scan_date) FROM scanner_data.scanner_results;
```

### Update MotherDuck Connection
Set environment variable:
```bash
export DUCKDB_PATH="md:?motherduck_token=YOUR_TOKEN"
```

## Migration Notes

**Old System:**
- App calculated patterns on every page load
- Slow for large stock universes
- High CPU usage
- Timeout issues with 2000+ stocks

**New System:**
- Pre-calculated results in database
- Fast page loads (<1 second)
- Low CPU usage
- No timeouts
- Requires periodic scanner runs
