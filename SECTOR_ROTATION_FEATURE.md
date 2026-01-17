# 🔄 Sector & Industry Rotation Analysis (RRG)

## Overview
A new frontend page to visualize **Relative Rotation Graphs (RRG)** for sector and industry analysis using the `rotation_metrics` table in MotherDuck.

## ✅ Implementation Complete

### Backend (app.py)
- **Route:** `/sector-rotation`
- **Query Parameters:**
  - `view` - 'sector' (default) or 'industry'
  - `days` - tail length: 5, 10, 20, 30 (default), or 60 days
- **Data Sources:**
  - `v_rrg_latest` - Latest snapshot with quadrant positions
  - `v_rrg_sector_tail` - Historical sector movements
  - `v_rrg_industry_tail` - Historical industry movements

### Frontend (templates/sector_rotation.html)
Interactive RRG visualization featuring:
- **Plotly scatter chart** with quadrant layout
- **Tail visualization** showing movement over selected timeframe
- **Quadrant breakdown:** Leading, Weakening, Lagging, Improving
- **Detailed table** with RS Ratio, RS Momentum, and 5-day changes
- **Toggle between sectors (11) and industries (102)**

### Navigation Updates
Added "🔄 Sector Rotation" links to:
- ✅ index.html (main scanners page)
- ✅ darkpool_signals.html
- ✅ options_signals.html
- ✅ vix_chart.html

## 📊 Database Schema

### Main Table: `rotation_metrics`
```sql
date            DATE        -- Trading date
group_type      VARCHAR     -- 'sector' or 'industry'
group_name      VARCHAR     -- Name (e.g., "TECHNOLOGY", "SOFTWARE")
members_count   INTEGER     -- Number of stocks in group
rs_ratio        DOUBLE      -- Relative Strength vs SPY
rs_momentum     DOUBLE      -- RS Momentum (rate of change)
quadrant        VARCHAR     -- "Leading", "Weakening", "Lagging", "Improving"
rs_ratio_5d     DOUBLE      -- 5-day change in RS ratio
rs_momentum_5d  DOUBLE      -- 5-day change in RS momentum
created_at      TIMESTAMP   -- Insert timestamp
```

### Available Data
- **Date Range:** Nov 18, 2025 → Jan 16, 2026 (60 days)
- **Sectors:** 11 groups × 60 days = 451 rows
- **Industries:** 102 groups × 60 days = 4,182 rows
- **Total:** 4,633 rows

## 🎯 Quadrants Explained

### 🚀 Leading (RS Ratio > 100, RS Momentum > 100)
- Outperforming benchmark AND accelerating
- **Best for:** Long positions, high conviction trades

### ⚠️ Weakening (RS Ratio > 100, RS Momentum < 100)
- Still outperforming BUT losing momentum
- **Best for:** Take profits, reduce exposure

### ❌ Lagging (RS Ratio < 100, RS Momentum < 100)
- Underperforming AND decelerating
- **Best for:** Avoid, look elsewhere

### 💪 Improving (RS Ratio < 100, RS Momentum > 100)
- Underperforming BUT gaining momentum
- **Best for:** Early entries, sector rotation plays

## 📈 Trading Strategy

1. **Focus on Leading sectors/industries** for highest probability trades
2. **Watch Improving quadrant** for early rotation opportunities
3. **Avoid Lagging groups** - no reason to fight the trend
4. **Exit Weakening positions** before they rotate to Lagging
5. **Clockwise rotation** indicates strengthening (Improving → Leading → Weakening → Lagging)

## 🔧 Features

### Interactive Controls
- **View Toggle:** Switch between Sectors (11) and Industries (102)
- **Tail Length:** Adjust from 5 to 60 days to see movement history
- **Hover Details:** See exact RS Ratio, RS Momentum, and group info
- **Live Date Range:** Shows data availability

### Visualization
- **Scatter Plot:** Position = current state, tail = recent movement
- **Color Coding:**
  - 🟢 Leading: Green (#27ae60)
  - 🟡 Weakening: Orange (#f39c12)
  - 🔴 Lagging: Red (#e74c3c)
  - 🔵 Improving: Blue (#3498db)
- **Bubble Size:** Scaled by member count (more stocks = larger bubble)

### Stats Summary
- Count of groups in each quadrant
- 5-day change indicators (positive/negative)
- Detailed breakdown table with all metrics

## 🚀 Next Steps

### Optional Enhancements
1. **Stock Drill-Down:** Click on sector/industry to see member stocks
2. **Historical Playback:** Animate rotation over time
3. **Alerts:** Email when sectors rotate between quadrants
4. **Momentum Scoring:** Rank groups by rotation velocity
5. **Correlation Analysis:** Show inter-sector relationships

### Testing
```bash
# Test data availability
python3 -c "import duckdb; conn = duckdb.connect('md:scanner_data'); \
result = conn.execute('SELECT COUNT(*) FROM rotation_metrics').fetchone(); \
print(f'Total rows: {result[0]}')"

# Test views
python3 -c "import duckdb; conn = duckdb.connect('md:scanner_data'); \
result = conn.execute('SELECT * FROM v_rrg_latest LIMIT 5').fetchall(); \
[print(row) for row in result]"
```

### Access
- **URL:** `/sector-rotation`
- **Default View:** Sectors with 30-day tails
- **Example:** `/sector-rotation?view=industry&days=60`

## 📝 Notes

- Data updates daily as new rotation metrics are calculated
- Uses pre-built MotherDuck views for optimal performance
- Responsive design works on desktop and tablet
- No authentication required (inherits from app-level auth)

---

**Status:** ✅ Ready for production
**Date:** January 17, 2026
**Author:** Claude Sonnet 4.5
