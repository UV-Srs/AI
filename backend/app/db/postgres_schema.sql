-- =====================================================================
-- PostgreSQL Enterprise Biometric & PPE Compliance Schema
-- Database: PPE_Detect
-- Compatible with PostgreSQL 12+ (including PostgreSQL 18)
-- =====================================================================

-- Optional: Enable pgvector extension if available on the system
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'vector') THEN
        CREATE EXTENSION IF NOT EXISTS vector;
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        NULL; -- Continue with BYTEA storage if extension is not installed
END $$;

-- 1. Employees Table
CREATE TABLE IF NOT EXISTS employees (
    id VARCHAR(100) PRIMARY KEY,
    employee_id VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    department VARCHAR(100) NOT NULL,
    designation VARCHAR(100) NOT NULL,
    required_ppe JSONB NOT NULL DEFAULT '["helmet", "safety_vest"]'::jsonb,
    profile_image_reference TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_employees_emp_id ON employees(employee_id);
CREATE INDEX IF NOT EXISTS idx_employees_active ON employees(is_active);

-- 2. Employee Biometric Profiles Table
CREATE TABLE IF NOT EXISTS employee_biometric_profiles (
    id VARCHAR(100) PRIMARY KEY,
    employee_id VARCHAR(100) NOT NULL REFERENCES employees(employee_id) ON DELETE CASCADE,
    consent_status VARCHAR(50) NOT NULL DEFAULT 'PENDING', -- CONSENTED, WITHDRAWN, PENDING
    consent_timestamp TIMESTAMP WITH TIME ZONE,
    enrollment_status VARCHAR(50) NOT NULL DEFAULT 'PENDING', -- PENDING, ACTIVE, REVOKED
    enrollment_model_id VARCHAR(100) NOT NULL DEFAULT 'sface_v1',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_employee_biometric_profile UNIQUE (employee_id)
);

CREATE INDEX IF NOT EXISTS idx_biometric_profiles_emp ON employee_biometric_profiles(employee_id);
CREATE INDEX IF NOT EXISTS idx_biometric_profiles_status ON employee_biometric_profiles(enrollment_status);

-- 3. Employee Face Captures Table (Three angles required: FRONT, LEFT, RIGHT)
CREATE TABLE IF NOT EXISTS employee_face_captures (
    id VARCHAR(100) PRIMARY KEY,
    biometric_profile_id VARCHAR(100) NOT NULL REFERENCES employee_biometric_profiles(id) ON DELETE CASCADE,
    employee_id VARCHAR(100) NOT NULL REFERENCES employees(employee_id) ON DELETE CASCADE,
    capture_angle VARCHAR(20) NOT NULL CHECK (capture_angle IN ('FRONT', 'LEFT', 'RIGHT')),
    embedding BYTEA NOT NULL, -- 128 float32 values (512 bytes)
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

CREATE INDEX IF NOT EXISTS idx_face_captures_emp ON employee_face_captures(employee_id);
CREATE INDEX IF NOT EXISTS idx_face_captures_profile ON employee_face_captures(biometric_profile_id);
CREATE INDEX IF NOT EXISTS idx_face_captures_angle ON employee_face_captures(capture_angle);

-- 4. Face Recognition Models Table
CREATE TABLE IF NOT EXISTS face_recognition_models (
    id VARCHAR(100) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_weight_identifier VARCHAR(255) NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    landmark_model VARCHAR(100) NOT NULL,
    mesh_model VARCHAR(100) NOT NULL,
    preprocessing_version VARCHAR(50) NOT NULL,
    license_information VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. Face Recognition Events Table (Audit logs)
CREATE TABLE IF NOT EXISTS face_recognition_events (
    id VARCHAR(100) PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    camera_id VARCHAR(50) NOT NULL DEFAULT 'CAM-01',
    candidate_employee_id VARCHAR(100),
    candidate_name VARCHAR(255),
    best_match_score REAL NOT NULL DEFAULT 0.0,
    second_best_match_score REAL NOT NULL DEFAULT 0.0,
    match_margin REAL NOT NULL DEFAULT 0.0,
    identity_status VARCHAR(50) NOT NULL, -- REGISTERED, UNREGISTERED, UNCERTAIN, LIVENESS_REVIEW
    model_version VARCHAR(50) NOT NULL DEFAULT '2021dec',
    review_status VARCHAR(50) NOT NULL DEFAULT 'AUTO_VERIFIED', -- AUTO_VERIFIED, FLAGGED_FOR_REVIEW, MANUAL_OVERRIDE
    overall_ppe_result VARCHAR(50) NOT NULL DEFAULT 'COMPLIANT', -- COMPLIANT, NON_COMPLIANT, NEEDS_REVIEW
    ppe_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    missing_ppe JSONB NOT NULL DEFAULT '[]'::jsonb,
    snapshot_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_face_events_time ON face_recognition_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_face_events_emp ON face_recognition_events(candidate_employee_id);
CREATE INDEX IF NOT EXISTS idx_face_events_status ON face_recognition_events(identity_status);
CREATE INDEX IF NOT EXISTS idx_face_events_ppe ON face_recognition_events(overall_ppe_result);

-- 6. System Settings Table
CREATE TABLE IF NOT EXISTS system_settings (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL
);

-- Seed initial model registration if not present
INSERT INTO face_recognition_models (
    id, model_name, model_version, model_weight_identifier,
    embedding_dimension, landmark_model, mesh_model,
    preprocessing_version, license_information
) VALUES (
    'sface_v1',
    'OpenCV SFace',
    '2021dec',
    'face_recognition_sface_2021dec.onnx',
    128,
    'OpenCV YuNet 2023mar',
    'YuNet 5-Point Anthropometric Surface Polygon',
    'align_crop_112x112',
    'Apache-2.0'
) ON CONFLICT (id) DO UPDATE SET
    model_weight_identifier = EXCLUDED.model_weight_identifier,
    license_information = EXCLUDED.license_information;

-- Seed system settings if not present
INSERT INTO system_settings (key, value) VALUES
    ('face_match_threshold', '0.363'),
    ('face_match_margin', '0.050'),
    ('face_min_quality', '0.60'),
    ('event_cooldown_seconds', '20'),
    ('retention_days', '90'),
    ('default_required_ppe', '["helmet", "safety_vest"]')
ON CONFLICT (key) DO NOTHING;

