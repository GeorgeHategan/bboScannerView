# Testing Guide

## Running Tests

### Schema Reference Test

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
