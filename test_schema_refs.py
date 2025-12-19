#!/usr/bin/env python3
"""
Test to ensure no duplicate 'main' schema references exist in the codebase.
This prevents queries from breaking due to incorrect paths like:
- scanner_data.main.main.* (should be scanner_data.main.*)
- scanner_data.main.main.main.* (should be scanner_data.main.*)
"""

import re
import sys
from pathlib import Path

def test_schema_references():
    """Check for duplicate 'main' in schema references."""
    
    errors = []
    app_file = Path(__file__).parent / 'app.py'
    
    if not app_file.exists():
        print(f"ERROR: {app_file} not found!")
        return False
    
    content = app_file.read_text()
    
    # Pattern 1: scanner_data.main.main.* (double main)
    double_main_pattern = r'scanner_data\.main\.main\.'
    double_main_matches = re.finditer(double_main_pattern, content)
    
    for match in double_main_matches:
        # Get line number
        line_num = content[:match.start()].count('\n') + 1
        line_start = content.rfind('\n', 0, match.start()) + 1
        line_end = content.find('\n', match.end())
        line_content = content[line_start:line_end].strip()
        
        errors.append({
            'line': line_num,
            'pattern': 'scanner_data.main.main.*',
            'content': line_content
        })
    
    # Pattern 2: scanner_data.main.main.main.* (triple main)
    triple_main_pattern = r'scanner_data\.main\.main\.main\.'
    triple_main_matches = re.finditer(triple_main_pattern, content)
    
    for match in triple_main_matches:
        # Get line number
        line_num = content[:match.start()].count('\n') + 1
        line_start = content.rfind('\n', 0, match.start()) + 1
        line_end = content.find('\n', match.end())
        line_content = content[line_start:line_end].strip()
        
        errors.append({
            'line': line_num,
            'pattern': 'scanner_data.main.main.main.*',
            'content': line_content
        })
    
    # Report results
    if errors:
        print("❌ FAILED: Found duplicate 'main' schema references!\n")
        print("These should be fixed to use 'scanner_data.main.*' instead:\n")
        
        for error in errors:
            print(f"Line {error['line']}: {error['pattern']}")
            print(f"  {error['content'][:100]}...")
            print()
        
        return False
    else:
        print("✅ PASSED: No duplicate 'main' schema references found!")
        return True

if __name__ == '__main__':
    success = test_schema_references()
    sys.exit(0 if success else 1)
