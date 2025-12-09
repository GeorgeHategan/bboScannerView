#!/usr/bin/env python3
"""
Migrate user management data from SQLite to MotherDuck.
Run this once to migrate existing users and login logs.
"""
import sqlite3
import duckdb
import os

# Get MotherDuck token
USER_DB_TOKEN = os.environ.get('options_motherduck_token') or os.environ.get('OPTIONS_MOTHERDUCK_TOKEN', '')

if not USER_DB_TOKEN:
    print("ERROR: No MotherDuck token found!")
    print("Set OPTIONS_MOTHERDUCK_TOKEN environment variable")
    exit(1)

# Connect to SQLite
sqlite_conn = sqlite3.connect('users.db')
sqlite_cursor = sqlite_conn.cursor()

# Connect to MotherDuck
md_conn = duckdb.connect(f'md:options_data?motherduck_token={USER_DB_TOKEN}')

print("=" * 60)
print("Migrating User Management Data to MotherDuck")
print("=" * 60)

# Migrate allowed_users
print("\n1. Migrating allowed_users...")
sqlite_cursor.execute('SELECT id, email, added_by, added_at, is_active FROM allowed_users')
users = sqlite_cursor.fetchall()
print(f"Found {len(users)} users in SQLite")

for user in users:
    user_id, email, added_by, added_at, is_active = user
    try:
        md_conn.execute('''
            INSERT INTO allowed_users (id, email, added_by, added_at, is_active)
            SELECT ?, ?, ?, ?, ?
            WHERE NOT EXISTS (SELECT 1 FROM allowed_users WHERE email = ?)
        ''', (user_id, email, added_by, added_at, bool(is_active), email))
        print(f"  ✓ Migrated: {email}")
    except Exception as e:
        print(f"  ✗ Error migrating {email}: {e}")

# Migrate login_logs
print("\n2. Migrating login_logs...")
sqlite_cursor.execute('''
    SELECT id, email, login_time, ip_address, user_agent, success, 
           failure_reason, country, country_code, city 
    FROM login_logs
''')
logs = sqlite_cursor.fetchall()
print(f"Found {len(logs)} login logs in SQLite")

migrated = 0
for log in logs:
    try:
        md_conn.execute('''
            INSERT INTO login_logs (id, email, login_time, ip_address, user_agent,
                                   success, failure_reason, country, country_code, city)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', log)
        migrated += 1
    except Exception as e:
        print(f"  ✗ Error migrating log {log[0]}: {e}")

print(f"  ✓ Migrated {migrated} login logs")

# Update sequences
print("\n3. Updating sequences...")
max_user_id = md_conn.execute('SELECT MAX(id) FROM allowed_users').fetchone()[0] or 0
max_log_id = md_conn.execute('SELECT MAX(id) FROM login_logs').fetchone()[0] or 0

md_conn.execute(f"ALTER SEQUENCE allowed_users_seq RESTART WITH {max_user_id + 1}")
md_conn.execute(f"ALTER SEQUENCE login_logs_seq RESTART WITH {max_log_id + 1}")
print(f"  ✓ Set allowed_users_seq to {max_user_id + 1}")
print(f"  ✓ Set login_logs_seq to {max_log_id + 1}")

# Verify
print("\n4. Verification...")
md_users = md_conn.execute('SELECT COUNT(*) FROM allowed_users').fetchone()[0]
md_logs = md_conn.execute('SELECT COUNT(*) FROM login_logs').fetchone()[0]
print(f"  MotherDuck has {md_users} users")
print(f"  MotherDuck has {md_logs} login logs")

sqlite_conn.close()
md_conn.close()

print("\n" + "=" * 60)
print("Migration Complete!")
print("=" * 60)
print("\nYou can now deploy to Render and the user data will persist.")
