#!/usr/bin/env python3
"""
Fix all scanner_data table references to include main schema.
This script will update app.py to use scanner_data.main.table instead of scanner_data.table
"""

import re

# Read the file
with open('app.py', 'r') as f:
    content = f.read()

# Replace scanner_data.{table} with scanner_data.main.{table}
# But skip lines that already have .main. or are variable names like scanner_data.items()
pattern = r'scanner_data\.([a-z_]+)'

def replace_func(match):
    table_name = match.group(1)
    # Skip if it's a Python method/attribute (like items, keys, values, etc.)
    python_methods = ['items', 'keys', 'values', 'get', 'pop', 'update', 'clear', 'copy']
    if table_name in python_methods:
        return match.group(0)
    # Skip if already has main
    return f'scanner_data.main.{table_name}'

# Apply replacement
new_content = re.sub(pattern, replace_func, content)

# Write back
with open('app.py', 'w') as f:
    f.write(new_content)

print("✅ Fixed scanner_data table references")

# Show what changed
original_count = content.count('scanner_data.')
new_count = new_content.count('scanner_data.main.')
print(f"   scanner_data.main. references: {new_count}")
