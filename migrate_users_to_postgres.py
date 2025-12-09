#!/usr/bin/env python3
"""
Migrate user management data from SQLite to PostgreSQL on Render.
Run this once after setting up PostgreSQL database on Render.
"""
import sqlite3
import psycopg2
import os

# Get PostgreSQL connection string
DATABASE_URL = os.environ.get('DATABASE_URL', '')

if not DATABASE_URL:
    print("ERROR: No DATABASE_URL found!")
    print("Set DATABASE_URL environment variable to your Render PostgreSQL connection string")
    exit(1)

# Fix URL format (Render uses postgresql://, psycopg2 needs postgres://)
db_url = DATABASE_URL.replace('postgresql://', 'postgres://')

# Connect to SQLite
sqlite_conn = sqlite3.connect('users.db')
sqlite_cursor = sqlite_conn.cursor()

# Connect to PostgreSQL
pg_conn = psycopg2.connect(db_url)
pg_cursor = pg_conn.cursor()

print("=" * 60)
print("Migrating User Management Data to PostgreSQL")
print("=" * 60)

# Migrate allowed_users
print("\n1. Migrating allowed_users...")
sqlite_cursor.execute('SELECT email, added_by, added_at, is_active FROM allowed_users')
users = sqlite_cursor.fetchall()
print(f"Found {len(users)} users in SQLite")

migrated_users = 0
for user in users:
    email, added_by, added_at, is_active = user
    try:
        pg_cursor.execute('''
            INSERT INTO allowed_users (email, added_by, added_at, is_active)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (email) DO NOTHING
        ''', (email, added_by, added_at, bool(is_active)))
        migrated_users += 1
        print(f"  ✓ Migrated: {email}")
    except Exception as e:
        print(f"  ✗ Error migrating {email}: {e}")

pg_conn.commit()
print(f"  ✓ Migrated {migrated_users} users")

# Migrate login_logs
print("\n2. Migrating login_logs...")
sqlite_cursor.execute('''
    SELECT email, login_time, ip_address, user_agent, success, 
           failure_reason, country, country_code, city 
    FROM login_logs
''')
logs = sqlite_cursor.fetchall()
print(f"Found {len(logs)} login logs in SQLite")

migrated_logs = 0
for log in logs:
    try:
        pg_cursor.execute('''
            INSERT INTO login_logs (email, login_time, ip_address, user_agent,
                                   success, failure_reason, country, country_code, city)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', log)
        migrated_logs += 1
    except Exception as e:
        print(f"  ✗ Error migrating log: {e}")

pg_conn.commit()
print(f"  ✓ Migrated {migrated_logs} login logs")

# Verify
print("\n3. Verification...")
pg_cursor.execute('SELECT COUNT(*) FROM allowed_users')
pg_users = pg_cursor.fetchone()[0]
pg_cursor.execute('SELECT COUNT(*) FROM login_logs')
pg_logs = pg_cursor.fetchone()[0]
print(f"  PostgreSQL has {pg_users} users")
print(f"  PostgreSQL has {pg_logs} login logs")

sqlite_conn.close()
pg_cursor.close()
pg_conn.close()

print("\n" + "=" * 60)
print("Migration Complete!")
print("=" * 60)
print("\nYou can now deploy to Render with DATABASE_URL set.")
print("User data will persist across deployments.")
