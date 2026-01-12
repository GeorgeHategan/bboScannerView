# URGENT: Fix yoy_eps_growth Data in fundamental_quality_scores (fq_v2)

## Issue
The `yoy_eps_growth` values in the `score_components` JSON field are stored as **percentages** instead of **decimals**, causing the frontend to display incorrect values.

**Example (KO symbol):**
- Stored value: `-15.972222328186035`
- Frontend multiplies by 100: `-15.972 * 100 = -1597.2%` ❌
- Should display: `-15.97%` ✓

## Root Cause
All other raw_inputs fields (`quarterly_earnings_growth`, `operating_margin`, `profit_margin`, etc.) are stored as decimals:
- `quarterly_earnings_growth: 0.301` → displayed as `30.1%` ✓
- `yoy_eps_growth: -15.972` → displayed as `-1597.2%` ❌

The `yoy_eps_growth` field is inconsistent with the rest of the data format.

## Required Fix
Update all `yoy_eps_growth` values in `fundamental_quality_scores` table where `score_version = 'fq_v2'`:

**Divide yoy_eps_growth by 100 to convert percentage → decimal**

### Example Fix Query (DuckDB/MotherDuck)
```sql
-- Update score_components JSON to fix yoy_eps_growth
UPDATE scanner_data.main.fundamental_quality_scores
SET score_components = json_set(
    score_components,
    '$.raw_inputs.yoy_eps_growth',
    CASE 
        WHEN json_extract(score_components, '$.raw_inputs.yoy_eps_growth') IS NOT NULL 
        THEN json_extract(score_components, '$.raw_inputs.yoy_eps_growth') / 100.0
        ELSE NULL
    END
)
WHERE score_version = 'fq_v2'
AND json_extract(score_components, '$.raw_inputs.yoy_eps_growth') IS NOT NULL;
```

## Verification Steps
1. **Before fix - check current values:**
```python
import duckdb
import json
conn = duckdb.connect('md:scanner_data')

result = conn.execute("""
    SELECT symbol, score_components 
    FROM main.fundamental_quality_scores 
    WHERE score_version = 'fq_v2' AND symbol = 'KO'
""").fetchone()

components = json.loads(result[1])
print(f"KO yoy_eps_growth: {components['raw_inputs']['yoy_eps_growth']}")
# Expected BEFORE: -15.972222328186035
# Expected AFTER: -0.15972222328186035
```

2. **After fix - verify corrected values:**
```python
# KO should show:
# yoy_eps_growth: -0.1597 (displayed as -15.97%)
# quarterly_earnings_growth: 0.301 (displayed as 30.1%)
```

3. **Test multiple symbols:**
```python
results = conn.execute("""
    SELECT symbol, score_components 
    FROM main.fundamental_quality_scores 
    WHERE score_version = 'fq_v2' 
    LIMIT 5
""").fetchall()

for row in results:
    components = json.loads(row[1])
    yoy = components['raw_inputs'].get('yoy_eps_growth')
    if yoy is not None:
        # Values should be between -1.0 and 10.0 (decimals)
        # NOT between -100 and 1000 (percentages)
        assert abs(yoy) < 100, f"{row[0]}: yoy_eps_growth still stored as percentage: {yoy}"
```

## Impact
- Affects **1833 symbols** with fq_v2 scores
- Frontend display shows values 100x too large
- Does not affect scoring calculation (only display issue)

## Priority
**HIGH** - User-facing display bug showing completely wrong values

## Files Using This Data
- `templates/index.html` - scanner results tooltips
- `templates/focus_list.html` - focus list tooltips
- `app.py` lines 3285-3295, 7217-7227 - fund_quality_dict construction

## Notes
- Do NOT fix the frontend to divide by 100 - fix the data to match the standard format
- All other metrics in raw_inputs are decimals (0.0 to 1.0 range for percentages)
- This maintains consistency across all fundamental metrics
