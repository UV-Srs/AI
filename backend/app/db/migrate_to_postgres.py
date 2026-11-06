"""
Migration Utility: Copies existing SQLite data (employees, biometric face templates, events)
into PostgreSQL.
Usage:
  python backend/app/db/migrate_to_postgres.py
"""

import os
import sys
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

SQLITE_PATH = PROJECT_ROOT / "data" / "ppe_compliance.db"

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("[ERROR] psycopg2-binary not installed.")
    sys.exit(1)

host = os.getenv("POSTGRES_HOST", "localhost")
port = int(os.getenv("POSTGRES_PORT", "5432"))
dbname = os.getenv("POSTGRES_DB", "ppe_compliance")
user = os.getenv("POSTGRES_USER", "postgres")
password = os.getenv("POSTGRES_PASSWORD", "")
database_url = os.getenv("DATABASE_URL", "")

if not SQLITE_PATH.exists():
    print(f"[INFO] No SQLite database found at {SQLITE_PATH}. Nothing to migrate.")
    sys.exit(0)

print("=" * 65)
print("MIGRATING SQLITE DATA TO POSTGRESQL")
print(f"Source: {SQLITE_PATH}")
print(f"Target: PostgreSQL database '{dbname}' on {host}:{port}")
print("=" * 65)

try:
    if database_url:
        pg_conn = psycopg2.connect(database_url)
    else:
        pg_conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
    pg_conn.autocommit = False
    pg_cursor = pg_conn.cursor()

    # Ensure tables exist in Postgres
    from backend.app.db.database import init_db
    # Temporarily force postgres
    import backend.app.db.database as db_mod
    db_mod._ACTIVE_BACKEND = "postgres"
    init_db()

    # Connect to SQLite
    sq_conn = sqlite3.connect(str(SQLITE_PATH))
    sq_conn.row_factory = sqlite3.Row
    sq_cursor = sq_conn.cursor()

    # 1. Migrate Employees
    sq_cursor.execute("SELECT * FROM employees")
    employees = sq_cursor.fetchall()
    emp_count = 0
    for e in employees:
        pg_cursor.execute("""
        INSERT INTO employees (id, name, department, designation, required_ppe, registration_date, is_active, consent_status, consent_date, photo_path, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            department = EXCLUDED.department,
            designation = EXCLUDED.designation,
            required_ppe = EXCLUDED.required_ppe,
            is_active = EXCLUDED.is_active,
            consent_status = EXCLUDED.consent_status,
            consent_date = EXCLUDED.consent_date,
            photo_path = EXCLUDED.photo_path,
            updated_at = EXCLUDED.updated_at;
        """, (
            e["id"], e["name"], e["department"], e["designation"], e["required_ppe"],
            e["registration_date"], e["is_active"], e["consent_status"], e["consent_date"],
            e["photo_path"], e["created_at"], e["updated_at"]
        ))
        emp_count += 1
    print(f"[OK] Migrated {emp_count} employee records.")

    # 2. Migrate Biometric Face Templates
    sq_cursor.execute("SELECT * FROM biometric_templates")
    templates = sq_cursor.fetchall()
    tmpl_count = 0
    for t in templates:
        pg_cursor.execute("""
        INSERT INTO biometric_templates (employee_id, embedding, samples_count, quality_score, enrolled_at)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (employee_id) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            samples_count = EXCLUDED.samples_count,
            quality_score = EXCLUDED.quality_score,
            enrolled_at = EXCLUDED.enrolled_at;
        """, (
            t["employee_id"], psycopg2.Binary(t["embedding"]),
            t["samples_count"], t["quality_score"], t["enrolled_at"]
        ))
        tmpl_count += 1
    print(f"[OK] Migrated {tmpl_count} biometric face templates (BYTEA embeddings).")

    # 3. Migrate Inspection Events
    sq_cursor.execute("SELECT * FROM inspection_events")
    events = sq_cursor.fetchall()
    evt_count = 0
    for ev in events:
        pg_cursor.execute("""
        INSERT INTO inspection_events (id, timestamp, camera_id, employee_id, employee_name, identity_state, match_score, ppe_summary, missing_ppe, overall_result, model_id, snapshot_path)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING;
        """, (
            ev["id"], ev["timestamp"], ev["camera_id"], ev["employee_id"],
            ev["employee_name"], ev["identity_state"], ev["match_score"],
            ev["ppe_summary"], ev["missing_ppe"], ev["overall_result"],
            ev["model_id"], ev["snapshot_path"]
        ))
        evt_count += 1
    print(f"[OK] Migrated {evt_count} inspection events.")

    pg_conn.commit()
    sq_conn.close()
    pg_conn.close()
    print("\n[SUCCESS] Migration to PostgreSQL completed successfully!")

except Exception as e:
    print(f"\n[ERROR] Migration failed: {e}")
    sys.exit(1)

