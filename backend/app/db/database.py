"""
Enterprise Database Layer - PostgreSQL (PSQL) Dedicated Engine.
Provides persistent storage for:
- Registered employees (`employees`)
- Biometric profiles (`employee_biometric_profiles`)
- 3-Angle Face Captures (`employee_face_captures`: FRONT, LEFT, RIGHT)
- Model metadata registry (`face_recognition_models`)
- Audit recognition & PPE inspection events (`face_recognition_events`)
- Multi-camera channel configurations (`camera_channels`)
- Continuous learning media history (`learned_media_samples`)
- System settings (`system_settings`)
"""

import os
import sys
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError as err:
    PSYCOPG2_AVAILABLE = False
    RealDictCursor = None
    print(f"[Database ERROR] psycopg2 is not installed: {err}. Please run 'pip install psycopg2-binary'", flush=True)

# PostgreSQL Configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "PPE_Detect")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "yuvraj9090M")
DATABASE_URL = os.getenv("DATABASE_URL", "")

_ACTIVE_BACKEND: str = "postgres"


def get_active_backend() -> str:
    """Returns active database backend name (strictly PostgreSQL)."""
    return "postgres"


def _create_postgres_connection():
    """Establishes connection to PostgreSQL database."""
    if not PSYCOPG2_AVAILABLE:
        raise RuntimeError("psycopg2 is not installed. PostgreSQL is required.")

    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor, connect_timeout=8)
    else:
        return psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            cursor_factory=RealDictCursor,
            connect_timeout=8
        )


def _test_postgres_connection() -> bool:
    try:
        conn = _create_postgres_connection()
        conn.close()
        return True
    except Exception as e:
        print(f"[Database] PostgreSQL connection check error: {e}", flush=True)
        return False


# Verify connection on module load
if _test_postgres_connection():
    print(f"[Database] Active backend: PostgreSQL (database='{POSTGRES_DB}', host='{POSTGRES_HOST}:{POSTGRES_PORT}')", flush=True)
else:
    print(f"[Database WARNING] PostgreSQL database '{POSTGRES_DB}' at {POSTGRES_HOST}:{POSTGRES_PORT} is not responding yet.", flush=True)


@contextmanager
def get_connection():
    """Context manager for PostgreSQL database connections with automatic commit and rollback."""
    conn = None
    try:
        conn = _create_postgres_connection()
        yield conn
        conn.commit()
    except Exception as e:
        if conn and not conn.closed:
            conn.rollback()
        raise RuntimeError(
            f"[Database] PostgreSQL connection failed: {e}. "
            f"Please ensure PostgreSQL service is running on {POSTGRES_HOST}:{POSTGRES_PORT} with database '{POSTGRES_DB}'."
        ) from e
    finally:
        if conn and not conn.closed:
            conn.close()


def init_db():
    """Initializes tables and schemas in PostgreSQL."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # 1. Employees table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id VARCHAR(100) PRIMARY KEY,
            employee_id VARCHAR(100) UNIQUE NOT NULL,
            name VARCHAR(255),
            full_name VARCHAR(255) NOT NULL,
            department VARCHAR(100) NOT NULL,
            designation VARCHAR(100) NOT NULL,
            required_ppe JSONB NOT NULL DEFAULT '["helmet", "safety_vest"]'::jsonb,
            profile_image_reference TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            consent_status VARCHAR(50) DEFAULT 'PENDING',
            consent_date TIMESTAMP WITH TIME ZONE,
            photo_path TEXT,
            registration_date TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # 2. Biometric Profiles table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS employee_biometric_profiles (
            id VARCHAR(100) PRIMARY KEY,
            employee_id VARCHAR(100) NOT NULL REFERENCES employees(employee_id) ON DELETE CASCADE,
            consent_status VARCHAR(50) NOT NULL DEFAULT 'PENDING',
            consent_timestamp TIMESTAMP WITH TIME ZONE,
            enrollment_status VARCHAR(50) NOT NULL DEFAULT 'PENDING',
            enrollment_model_id VARCHAR(100) NOT NULL DEFAULT 'sface_v1',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_employee_biometric_profile UNIQUE (employee_id)
        );
        """)

        # 3. Face Captures (3 Angles)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS employee_face_captures (
            id VARCHAR(100) PRIMARY KEY,
            biometric_profile_id VARCHAR(100) NOT NULL REFERENCES employee_biometric_profiles(id) ON DELETE CASCADE,
            employee_id VARCHAR(100) NOT NULL REFERENCES employees(employee_id) ON DELETE CASCADE,
            capture_angle VARCHAR(20) NOT NULL CHECK (capture_angle IN ('FRONT', 'LEFT', 'RIGHT')),
            embedding BYTEA NOT NULL,
            embedding_dimension INTEGER NOT NULL DEFAULT 128,
            embedding_model VARCHAR(100) NOT NULL DEFAULT 'OpenCV SFace',
            embedding_model_version VARCHAR(50) NOT NULL DEFAULT '2021dec',
            facial_landmarks JSONB NOT NULL,
            facial_vertices JSONB NOT NULL,
            facial_geometry JSONB NOT NULL,
            face_pose JSONB NOT NULL,
            face_quality JSONB NOT NULL,
            capture_image_reference TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_employee_capture_angle UNIQUE (employee_id, capture_angle)
        );
        """)

        # 4. Face Recognition & PPE Inspection Audit Events
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_recognition_events (
            id VARCHAR(100) PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            camera_id VARCHAR(50) NOT NULL DEFAULT 'CAM-01',
            candidate_employee_id VARCHAR(100),
            candidate_name VARCHAR(255),
            best_match_score REAL NOT NULL DEFAULT 0.0,
            second_best_match_score REAL NOT NULL DEFAULT 0.0,
            match_margin REAL NOT NULL DEFAULT 0.0,
            identity_status VARCHAR(50) NOT NULL,
            model_version VARCHAR(50) NOT NULL DEFAULT '2021dec',
            review_status VARCHAR(50) NOT NULL DEFAULT 'AUTO_VERIFIED',
            overall_ppe_result VARCHAR(50) NOT NULL DEFAULT 'COMPLIANT',
            ppe_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
            missing_ppe JSONB NOT NULL DEFAULT '[]'::jsonb,
            snapshot_path TEXT
        );
        """)

        # 5. Multi-Camera Channels
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS camera_channels (
            id VARCHAR(50) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            source_type VARCHAR(20) NOT NULL DEFAULT 'webcam',
            device_id TEXT,
            media_filename TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Insert Default Main Camera if empty
        cursor.execute("""
        INSERT INTO camera_channels (id, name, source_type, is_active)
        VALUES ('CAM-01', 'Camera 1 (Main Gate)', 'webcam', 1)
        ON CONFLICT (id) DO NOTHING;
        """)

        # 6. Self-Learning Samples History
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS learned_media_samples (
            id VARCHAR(100) PRIMARY KEY,
            camera_id VARCHAR(50) NOT NULL,
            media_type VARCHAR(20) NOT NULL,
            filename TEXT NOT NULL,
            samples_count INTEGER NOT NULL DEFAULT 1,
            classes_learned TEXT NOT NULL DEFAULT '[]',
            employee_id VARCHAR(50),
            employee_name TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # 7. System Settings
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_settings (
            key VARCHAR(100) PRIMARY KEY,
            value TEXT NOT NULL
        );
        """)

        default_settings = [
            ("face_match_threshold", "0.363"),
            ("face_match_margin", "0.050"),
            ("face_min_quality", "0.60"),
            ("event_cooldown_seconds", "3"),
            ("retention_days", "90"),
            ("default_required_ppe", json.dumps(["helmet", "safety_vest"])),
            ("class_thresholds", json.dumps({
                "helmet": 0.15,
                "safety_vest": 0.15,
                "gloves": 0.15,
                "safety_shoes": 0.15,
                "goggles": 0.15,
                "mask": 0.15,
                "ear_protection": 0.15,
                "person": 0.25
            }))
        ]
        for k, v in default_settings:
            cursor.execute("""
            INSERT INTO system_settings (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO NOTHING;
            """, (k, v))


# Initialize tables on load
try:
    init_db()
except Exception as e:
    print(f"[Database] Initialization check: {e}", flush=True)


# ==========================================
# Employee Management
# ==========================================

def get_all_employees(active_only: bool = False) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        query = """
        SELECT e.*, 
               p.consent_status, 
               p.consent_timestamp, 
               p.enrollment_status
        FROM employees e
        LEFT JOIN employee_biometric_profiles p ON e.employee_id = p.employee_id
        """
        if active_only:
            query += " WHERE e.is_active = 1"
        query += " ORDER BY e.full_name ASC"

        cursor.execute(query)
        rows = cursor.fetchall()

        res = []
        for r in rows:
            d = dict(r)
            emp_id = d["employee_id"]

            if isinstance(d.get("required_ppe"), str):
                try:
                    d["required_ppe"] = json.loads(d["required_ppe"])
                except Exception:
                    d["required_ppe"] = ["helmet", "safety_vest"]
            d["is_active"] = bool(d["is_active"])
            d["name"] = d.get("full_name") or d.get("name", "")
            d["id"] = emp_id

            # Query completed face captures
            cursor.execute("SELECT capture_angle FROM employee_face_captures WHERE employee_id = %s", (emp_id,))
            captures = cursor.fetchall()
            completed_angles = [c["capture_angle"] for c in captures]
            d["completed_angles"] = completed_angles
            d["has_biometric_template"] = len(completed_angles) == 3
            d["enrollment_status"] = d.get("enrollment_status") or ("ACTIVE" if len(completed_angles) == 3 else "PENDING")
            d["consent_status"] = d.get("consent_status") or "PENDING"
            res.append(d)
        return res


def get_employee_by_id(employee_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        query = """
        SELECT e.*, 
               p.id as profile_id,
               p.consent_status, 
               p.consent_timestamp, 
               p.enrollment_status,
               p.enrollment_model_id
        FROM employees e
        LEFT JOIN employee_biometric_profiles p ON e.employee_id = p.employee_id
        WHERE e.employee_id = %s OR e.id = %s
        """
        cursor.execute(query, (employee_id, employee_id))
        row = cursor.fetchone()
        if not row:
            return None

        d = dict(row)
        emp_id = d["employee_id"]
        if isinstance(d.get("required_ppe"), str):
            try:
                d["required_ppe"] = json.loads(d["required_ppe"])
            except Exception:
                d["required_ppe"] = ["helmet", "safety_vest"]
        d["is_active"] = bool(d["is_active"])
        d["name"] = d.get("full_name") or d.get("name", "")
        d["id"] = emp_id

        cursor.execute("SELECT capture_angle, created_at FROM employee_face_captures WHERE employee_id = %s", (emp_id,))
        captures = cursor.fetchall()
        completed_angles = [c["capture_angle"] for c in captures]
        d["completed_angles"] = completed_angles
        d["has_biometric_template"] = len(completed_angles) == 3
        d["enrollment_status"] = d.get("enrollment_status") or ("ACTIVE" if len(completed_angles) == 3 else "PENDING")
        d["consent_status"] = d.get("consent_status") or "PENDING"
        return d


def create_or_update_employee(
    emp_id: str,
    name: str,
    department: str,
    designation: str,
    required_ppe: List[str],
    consent_status: str = "CONSENTED",
    is_active: bool = True,
    profile_image_reference: Optional[str] = None
) -> Dict[str, Any]:
    ppe_json = json.dumps(required_ppe)

    with get_connection() as conn:
        cursor = conn.cursor()

        # 1. Upsert employee
        cursor.execute("""
        INSERT INTO employees (
            id, employee_id, name, full_name, department, designation, required_ppe,
            profile_image_reference, is_active, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (employee_id) DO UPDATE SET
            name = EXCLUDED.name,
            full_name = EXCLUDED.full_name,
            department = EXCLUDED.department,
            designation = EXCLUDED.designation,
            required_ppe = EXCLUDED.required_ppe,
            is_active = EXCLUDED.is_active,
            profile_image_reference = COALESCE(EXCLUDED.profile_image_reference, employees.profile_image_reference),
            updated_at = CURRENT_TIMESTAMP;
        """, (
            emp_id, emp_id, name, name, department, designation, ppe_json,
            profile_image_reference, int(is_active)
        ))

        # 2. Upsert biometric profile record
        profile_id = f"PROF-{emp_id}"
        cursor.execute("""
        INSERT INTO employee_biometric_profiles (
            id, employee_id, consent_status, consent_timestamp,
            enrollment_status, enrollment_model_id, created_at, updated_at
        ) VALUES (%s, %s, %s, CURRENT_TIMESTAMP, 'PENDING', 'sface_v1', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (employee_id) DO UPDATE SET
            consent_status = EXCLUDED.consent_status,
            consent_timestamp = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP;
        """, (profile_id, emp_id, consent_status))

    return get_employee_by_id(emp_id)


def set_employee_active_status(emp_id: str, is_active: bool) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE employees SET is_active = %s, updated_at = CURRENT_TIMESTAMP WHERE employee_id = %s OR id = %s",
            (int(is_active), emp_id, emp_id)
        )
        return cursor.rowcount > 0


def delete_employee_and_biometrics(emp_id: str) -> bool:
    """Deletes employee record and cascades deletion to biometric profiles & face captures."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM employee_face_captures WHERE employee_id = %s", (emp_id,))
        cursor.execute("DELETE FROM employee_biometric_profiles WHERE employee_id = %s", (emp_id,))
        cursor.execute("DELETE FROM employees WHERE employee_id = %s OR id = %s", (emp_id, emp_id))
        return cursor.rowcount > 0


def withdraw_biometric_consent(emp_id: str) -> bool:
    """Purges face captures and sets profile status to WITHDRAWN."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM employee_face_captures WHERE employee_id = %s", (emp_id,))
        cursor.execute("""
        UPDATE employee_biometric_profiles 
        SET consent_status = 'WITHDRAWN', enrollment_status = 'REVOKED', updated_at = CURRENT_TIMESTAMP
        WHERE employee_id = %s;
        """, (emp_id,))
        return True


# ==========================================
# 3-Angle Biometric Face Capture Operations
# ==========================================

def save_face_capture(
    emp_id: str,
    capture_angle: str,
    embedding_bytes: bytes,
    landmarks: Dict[str, Any],
    vertices: Dict[str, Any],
    geometry: Dict[str, Any],
    pose: Dict[str, Any],
    quality: Dict[str, Any],
    image_reference: Optional[str] = None
) -> Dict[str, Any]:
    angle = capture_angle.upper()
    if angle not in ("FRONT", "LEFT", "RIGHT"):
        raise ValueError(f"Invalid capture angle '{capture_angle}'. Must be FRONT, LEFT, or RIGHT.")

    capture_id = f"CAP-{emp_id}-{angle}"
    profile_id = f"PROF-{emp_id}"

    with get_connection() as conn:
        cursor = conn.cursor()

        # Ensure biometric profile exists
        cursor.execute("""
        INSERT INTO employee_biometric_profiles (
            id, employee_id, consent_status, consent_timestamp,
            enrollment_status, enrollment_model_id, created_at, updated_at
        ) VALUES (%s, %s, 'CONSENTED', CURRENT_TIMESTAMP, 'PENDING', 'sface_v1', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (employee_id) DO NOTHING;
        """, (profile_id, emp_id))

        # Upsert face capture record
        cursor.execute("""
        INSERT INTO employee_face_captures (
            id, biometric_profile_id, employee_id, capture_angle, embedding,
            embedding_dimension, embedding_model, embedding_model_version,
            facial_landmarks, facial_vertices, facial_geometry, face_pose, face_quality,
            capture_image_reference, created_at
        ) VALUES (
            %s, %s, %s, %s, %s, 128, 'OpenCV SFace', '2021dec',
            %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
        )
        ON CONFLICT (employee_id, capture_angle) DO UPDATE SET
            biometric_profile_id = EXCLUDED.biometric_profile_id,
            embedding = EXCLUDED.embedding,
            facial_landmarks = EXCLUDED.facial_landmarks,
            facial_vertices = EXCLUDED.facial_vertices,
            facial_geometry = EXCLUDED.facial_geometry,
            face_pose = EXCLUDED.face_pose,
            face_quality = EXCLUDED.face_quality,
            capture_image_reference = COALESCE(EXCLUDED.capture_image_reference, employee_face_captures.capture_image_reference),
            created_at = CURRENT_TIMESTAMP;
        """, (
            capture_id, profile_id, emp_id, angle, psycopg2.Binary(embedding_bytes),
            json.dumps(landmarks), json.dumps(vertices), json.dumps(geometry),
            json.dumps(pose), json.dumps(quality), image_reference
        ))

        # Check if all 3 angles exist
        cursor.execute("SELECT capture_angle FROM employee_face_captures WHERE employee_id = %s;", (emp_id,))
        rows = cursor.fetchall()
        completed_angles = [r["capture_angle"] for r in rows]
        is_complete = set(completed_angles) >= {"FRONT", "LEFT", "RIGHT"}

        if is_complete:
            cursor.execute("""
            UPDATE employee_biometric_profiles
            SET enrollment_status = 'ACTIVE', updated_at = CURRENT_TIMESTAMP
            WHERE employee_id = %s;
            """, (emp_id,))

    return {
        "employee_id": emp_id,
        "capture_angle": angle,
        "completed_angles": completed_angles,
        "is_complete": is_complete,
        "enrollment_status": "ACTIVE" if is_complete else "PENDING"
    }


def get_employee_biometric_details(emp_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT p.*, e.full_name
        FROM employee_biometric_profiles p
        JOIN employees e ON p.employee_id = e.employee_id
        WHERE p.employee_id = %s;
        """, (emp_id,))
        prof = cursor.fetchone()
        if not prof:
            return None

        p_data = dict(prof)

        cursor.execute("""
        SELECT id, capture_angle, embedding_dimension, embedding_model, embedding_model_version,
               facial_geometry, face_pose, face_quality, created_at
        FROM employee_face_captures
        WHERE employee_id = %s
        ORDER BY capture_angle ASC;
        """, (emp_id,))
        captures = cursor.fetchall()

        parsed_captures = []
        for c in captures:
            cd = dict(c)
            if isinstance(cd.get("facial_geometry"), str):
                try:
                    cd["facial_geometry"] = json.loads(cd["facial_geometry"])
                except Exception:
                    pass
            if isinstance(cd.get("face_pose"), str):
                try:
                    cd["face_pose"] = json.loads(cd["face_pose"])
                except Exception:
                    pass
            if isinstance(cd.get("face_quality"), str):
                try:
                    cd["face_quality"] = json.loads(cd["face_quality"])
                except Exception:
                    pass
            parsed_captures.append(cd)

        p_data["captures"] = parsed_captures
        p_data["completed_angles"] = [c["capture_angle"] for c in parsed_captures]
        p_data["is_complete"] = len(parsed_captures) == 3
        return p_data


def get_all_active_biometric_templates() -> Dict[str, Dict[str, bytes]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT c.employee_id, c.capture_angle, c.embedding
        FROM employee_face_captures c
        JOIN employee_biometric_profiles p ON c.employee_id = p.employee_id
        JOIN employees e ON c.employee_id = e.employee_id
        WHERE e.is_active = 1 
          AND p.consent_status = 'CONSENTED'
          AND p.enrollment_status = 'ACTIVE';
        """)
        rows = cursor.fetchall()

        templates: Dict[str, Dict[str, bytes]] = {}
        for r in rows:
            emp_id = r["employee_id"]
            angle = r["capture_angle"]
            raw_emb = r["embedding"]
            emb_bytes = bytes(raw_emb)

            if emp_id not in templates:
                templates[emp_id] = {}
            templates[emp_id][angle] = emb_bytes

        return templates


# ==========================================
# Face Recognition & Inspection Events Audit Log
# ==========================================

def log_inspection_event(
    event_id: str,
    camera_id: str,
    candidate_employee_id: Optional[str],
    candidate_name: Optional[str],
    best_match_score: float,
    second_best_match_score: float,
    match_margin: float,
    identity_status: str,
    ppe_summary: Dict[str, str],
    missing_ppe: List[str],
    overall_result: str,
    model_version: str = "2021dec",
    review_status: str = "AUTO_VERIFIED",
    snapshot_path: Optional[str] = None
) -> Dict[str, Any]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO face_recognition_events (
            id, timestamp, camera_id, candidate_employee_id, candidate_name,
            best_match_score, second_best_match_score, match_margin, identity_status,
            model_version, review_status, overall_ppe_result, ppe_summary, missing_ppe, snapshot_path
        ) VALUES (
            %s, CURRENT_TIMESTAMP, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (id) DO NOTHING;
        """, (
            event_id, camera_id, candidate_employee_id, candidate_name,
            round(best_match_score, 4), round(second_best_match_score, 4), round(match_margin, 4),
            identity_status, model_version, review_status, overall_result,
            json.dumps(ppe_summary), json.dumps(missing_ppe), snapshot_path
        ))

    return {
        "id": event_id,
        "camera_id": camera_id,
        "candidate_employee_id": candidate_employee_id,
        "candidate_name": candidate_name,
        "best_match_score": best_match_score,
        "second_best_match_score": second_best_match_score,
        "match_margin": match_margin,
        "identity_status": identity_status,
        "overall_result": overall_result
    }


def get_inspection_events(
    limit: int = 100,
    employee_id: Optional[str] = None,
    overall_result: Optional[str] = None,
    identity_state: Optional[str] = None
) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM face_recognition_events WHERE 1=1"
        params = []
        if employee_id:
            query += " AND candidate_employee_id = %s"
            params.append(employee_id)
        if overall_result:
            query += " AND overall_ppe_result = %s"
            params.append(overall_result)
        if identity_state:
            query += " AND identity_status = %s"
            params.append(identity_state)
        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        events = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("ppe_summary"), str):
                try:
                    d["ppe_summary"] = json.loads(d["ppe_summary"])
                except Exception:
                    pass
            if isinstance(d.get("missing_ppe"), str):
                try:
                    d["missing_ppe"] = json.loads(d["missing_ppe"])
                except Exception:
                    pass
            d["employee_id"] = d.get("candidate_employee_id")
            d["employee_name"] = d.get("candidate_name")
            d["identity_state"] = d.get("identity_status")
            d["match_score"] = d.get("best_match_score", 0.0)
            d["overall_result"] = d.get("overall_ppe_result")
            d["model_id"] = f"YOLO26n + SFace {d.get('model_version', '2021dec')}"
            events.append(d)
        return events


def get_compliance_stats() -> Dict[str, Any]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM face_recognition_events;")
        total_scans = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM face_recognition_events WHERE overall_ppe_result = 'COMPLIANT';")
        compliant_scans = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM face_recognition_events WHERE overall_ppe_result = 'NON_COMPLIANT';")
        violations = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM employees WHERE is_active = 1;")
        active_employees = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM employee_biometric_profiles WHERE enrollment_status = 'ACTIVE';")
        enrolled_faces = cursor.fetchone()["count"]

        compliance_rate = round((compliant_scans / total_scans) * 100, 1) if total_scans > 0 else 100.0

        return {
            "total_scans": total_scans,
            "compliant_scans": compliant_scans,
            "violations": violations,
            "compliance_rate": compliance_rate,
            "active_employees": active_employees,
            "enrolled_faces": enrolled_faces
        }


def get_settings() -> Dict[str, str]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM system_settings;")
        rows = cursor.fetchall()
        return {r["key"]: r["value"] for r in rows}


def update_setting(key: str, value: str):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO system_settings (key, value)
        VALUES (%s, %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
        """, (key, value))


# ==========================================
# Multi-Camera Channel Management
# ==========================================

def get_camera_channels() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM camera_channels ORDER BY created_at ASC;")
        rows = cursor.fetchall()
        result = []
        for r in rows:
            keys = r.keys() if hasattr(r, "keys") else []
            result.append({
                "id": r["id"],
                "name": r["name"],
                "source_type": r["source_type"],
                "device_id": r["device_id"] if "device_id" in keys and r["device_id"] else "",
                "media_filename": r["media_filename"] if "media_filename" in keys and r["media_filename"] else "",
                "is_active": bool(r["is_active"] if "is_active" in keys and r["is_active"] is not None else 1),
                "created_at": str(r["created_at"] if "created_at" in keys and r["created_at"] else "")
            })
        if not result:
            result = [{
                "id": "CAM-01",
                "name": "Camera 1 (Main Gate)",
                "source_type": "webcam",
                "device_id": "",
                "media_filename": "",
                "is_active": True,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }]
        return result


def save_camera_channel(
    cam_id: str,
    name: str,
    source_type: str = "webcam",
    device_id: Optional[str] = "",
    media_filename: Optional[str] = "",
    is_active: int = 1
) -> Dict[str, Any]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO camera_channels (id, name, source_type, device_id, media_filename, is_active, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            source_type = EXCLUDED.source_type,
            device_id = EXCLUDED.device_id,
            media_filename = EXCLUDED.media_filename,
            is_active = EXCLUDED.is_active;
        """, (cam_id, name, source_type, device_id or "", media_filename or "", is_active))

    return {
        "id": cam_id,
        "name": name,
        "source_type": source_type,
        "device_id": device_id or "",
        "media_filename": media_filename or "",
        "is_active": bool(is_active)
    }


def delete_camera_channel(cam_id: str) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM camera_channels WHERE id = %s;", (cam_id,))
    return True


# ==========================================
# Self-Learning & Ingested Sample History
# ==========================================

def record_learned_sample(
    sample_id: str,
    camera_id: str,
    media_type: str,
    filename: str,
    samples_count: int = 1,
    classes_learned: Optional[List[str]] = None,
    employee_id: Optional[str] = None,
    employee_name: Optional[str] = None
) -> Dict[str, Any]:
    classes_json = json.dumps(classes_learned or [])
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO learned_media_samples (id, camera_id, media_type, filename, samples_count, classes_learned, employee_id, employee_name, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (id) DO NOTHING;
        """, (sample_id, camera_id, media_type, filename, samples_count, classes_json, employee_id, employee_name))

    return {
        "id": sample_id,
        "camera_id": camera_id,
        "media_type": media_type,
        "filename": filename,
        "samples_count": samples_count,
        "classes_learned": classes_learned or [],
        "employee_id": employee_id,
        "employee_name": employee_name
    }


def get_learned_samples(limit: int = 50) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM learned_media_samples ORDER BY created_at DESC LIMIT %s;", (limit,))
        rows = cursor.fetchall()
        result = []
        for r in rows:
            keys = r.keys() if hasattr(r, "keys") else []
            classes = []
            try:
                raw_cls = r["classes_learned"] if "classes_learned" in keys else "[]"
                classes = json.loads(raw_cls or "[]")
            except Exception:
                pass
            result.append({
                "id": r["id"],
                "camera_id": r["camera_id"],
                "media_type": r["media_type"],
                "filename": r["filename"],
                "samples_count": r["samples_count"] if "samples_count" in keys else 1,
                "classes_learned": classes,
                "employee_id": r["employee_id"] if "employee_id" in keys else None,
                "employee_name": r["employee_name"] if "employee_name" in keys else None,
                "created_at": str(r["created_at"] if "created_at" in keys and r["created_at"] else "")
            })
        return result
