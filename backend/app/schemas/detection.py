from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# ==========================================
# Detection & Inference Schemas
# ==========================================

class DetectionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class DetectionItem(BaseModel):
    box: List[float] = Field(..., description="[x1, y1, x2, y2] bounding box coordinates")
    class_id: int
    class_name: str
    confidence: float

class ModelInfo(BaseModel):
    id: str
    name: str
    filename: str
    path: str
    size_mb: float
    classes_count: int
    classes: List[str]
    is_active: bool
    exists: bool

class ModelSelectRequest(BaseModel):
    model_id: str

class HealthResponse(BaseModel):
    status: str
    app_name: str
    device: str
    device_name: Optional[str] = None
    cuda_available: bool = False
    active_model: Optional[str] = None
    active_model_id: Optional[str] = None
    model_loaded: bool
    classes: List[str]
    face_engine_ready: bool = True

# ==========================================
# Employee & Biometric Schemas
# ==========================================

class EmployeeCreateRequest(BaseModel):
    id: str
    name: str
    department: str
    designation: str
    required_ppe: List[str] = ["helmet", "safety_vest"]
    consent_status: str = "CONSENTED"
    is_active: bool = True

class FaceEnrollmentRequest(BaseModel):
    employee_id: str
    capture_angle: str = "FRONT" # 'FRONT' | 'LEFT' | 'RIGHT'
    image_base64: str # Data URI or raw base64 JPEG/PNG

class FaceAngleEnrollmentRequest(BaseModel):
    employee_id: str
    capture_angle: str # 'FRONT' | 'LEFT' | 'RIGHT'
    image_base64: str

class BiometricCaptureSummary(BaseModel):
    capture_angle: str
    embedding_dimension: int
    embedding_model: str
    embedding_model_version: str
    facial_geometry: Dict[str, Any]
    face_pose: Dict[str, Any]
    face_quality: Dict[str, Any]
    created_at: Any

class BiometricDetailsResponse(BaseModel):
    employee_id: str
    full_name: Optional[str] = None
    consent_status: str
    enrollment_status: str
    completed_angles: List[str]
    is_complete: bool
    captures: List[BiometricCaptureSummary]

class EmployeeResponse(BaseModel):
    id: str
    name: str
    department: str
    designation: str
    required_ppe: List[str]
    registration_date: Optional[str] = None
    is_active: bool
    consent_status: str
    consent_date: Optional[str] = None
    photo_path: Optional[str] = None
    enrollment_status: str = "PENDING"
    completed_angles: List[str] = []
    has_biometric_template: bool = False

# ==========================================
# Inspection Event & Stats Schemas
# ==========================================

class InspectionEventResponse(BaseModel):
    id: str
    timestamp: str
    camera_id: str
    employee_id: Optional[str] = None
    employee_name: Optional[str] = None
    identity_state: str
    match_score: float
    ppe_summary: Dict[str, str]
    missing_ppe: List[str]
    overall_result: str
    model_id: str

class ComplianceStatsResponse(BaseModel):
    total_scans: int
    compliant_scans: int
    violations: int
    compliance_rate: float
    active_employees: int
    enrolled_faces: int

class SettingsUpdateRequest(BaseModel):
    face_match_threshold: Optional[float] = None
    event_cooldown_seconds: Optional[int] = None
    default_required_ppe: Optional[List[str]] = None
    class_thresholds: Optional[Dict[str, float]] = None
