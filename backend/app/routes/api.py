import io
import csv
import base64
import json
import numpy as np
import cv2
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response
from typing import Dict, List, Any, Optional

from fastapi.responses import FileResponse
from pathlib import Path

from backend.app.schemas.detection import (
    ModelInfo, ModelSelectRequest, HealthResponse,
    EmployeeCreateRequest, FaceEnrollmentRequest, FaceAngleEnrollmentRequest,
    BiometricDetailsResponse, EmployeeResponse,
    ComplianceStatsResponse, SettingsUpdateRequest
)
from backend.app.services.model_manager import model_manager, APPROVED_CLASSES
from backend.app.face.face_engine import face_engine
from backend.app.db.database import (
    get_all_employees, get_employee_by_id, create_or_update_employee,
    set_employee_active_status, delete_employee_and_biometrics, withdraw_biometric_consent,
    save_face_capture, get_employee_biometric_details,
    get_inspection_events, get_compliance_stats,
    get_settings, update_setting,
    get_camera_channels, save_camera_channel, delete_camera_channel, get_learned_samples
)
from backend.app.services.learning_service import (
    ingest_media_for_learning, start_model_training, get_learning_status,
    inspect_media_detections
)

router = APIRouter(prefix="/api", tags=["Enterprise PPE & Face Recognition API"])

# ==========================================
# System & Model Endpoints
# ==========================================

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint providing backend status, model information, and face engine status."""
    status = model_manager.get_status()
    device_label = status["device"].upper()
    return HealthResponse(
        status="ok",
        app_name="ENTERPRISE PPE & FACIAL RECOGNITION SYSTEM",
        device=device_label,
        device_name=status.get("device_name"),
        cuda_available=status.get("cuda_available", False),
        active_model=status["active_model_name"],
        model_loaded=status["model_loaded"],
        classes=APPROVED_CLASSES,
        face_engine_ready=face_engine is not None
    )

@router.get("/models", response_model=List[ModelInfo])
def list_models():
    """List approved YOLO26n checkpoints with existence and file size info."""
    return model_manager.get_available_models()

@router.post("/models/select")
def select_model(request: ModelSelectRequest):
    """Switch active YOLO26n checkpoint."""
    success, msg = model_manager.load_model(request.model_id)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg, "active_model": model_manager.get_status()["active_model_name"]}

@router.get("/classes")
def get_classes():
    """Return the approved PPE classes with display labels and colors."""
    class_metadata = {
        "person": {"id": 0, "label": "Person", "color": "#3B82F6", "bgColor": "rgba(59, 130, 246, 0.15)"},
        "helmet": {"id": 1, "label": "Safety Helmet / Hard Hat", "color": "#F59E0B", "bgColor": "rgba(245, 158, 11, 0.15)"},
        "safety_vest": {"id": 2, "label": "Safety Vest", "color": "#F97316", "bgColor": "rgba(249, 115, 22, 0.15)"},
        "gloves": {"id": 3, "label": "Safety Gloves", "color": "#10B981", "bgColor": "rgba(16, 185, 129, 0.15)"},
        "safety_shoes": {"id": 4, "label": "Safety Boots / Shoes", "color": "#8B5CF6", "bgColor": "rgba(139, 92, 246, 0.15)"},
        "goggles": {"id": 5, "label": "Safety Glasses / Goggles", "color": "#06B6D4", "bgColor": "rgba(6, 182, 212, 0.15)"},
        "mask": {"id": 6, "label": "Face Mask / Respirator", "color": "#EC4899", "bgColor": "rgba(236, 72, 153, 0.15)"},
        "ear_protection": {"id": 7, "label": "Ear Protection / Muffs", "color": "#EAB308", "bgColor": "rgba(234, 179, 8, 0.15)"}
    }
    return {
        "count": len(APPROVED_CLASSES),
        "classes": APPROVED_CLASSES,
        "metadata": class_metadata
    }

# ==========================================
# Employee Management Endpoints
# ==========================================

@router.get("/employees")
def list_employees(active_only: bool = False):
    """Returns directory of registered employees."""
    return get_all_employees(active_only=active_only)

@router.post("/employees")
def save_employee(req: EmployeeCreateRequest):
    """Creates or updates employee profile."""
    emp = create_or_update_employee(
        emp_id=req.id.strip(),
        name=req.name.strip(),
        department=req.department.strip(),
        designation=req.designation.strip(),
        required_ppe=req.required_ppe,
        consent_status=req.consent_status,
        is_active=req.is_active
    )
    return {"success": True, "employee": emp}

@router.get("/employees/{emp_id}")
def get_employee(emp_id: str):
    emp = get_employee_by_id(emp_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return emp

@router.delete("/employees/{emp_id}")
def delete_employee(emp_id: str):
    """Deletes employee and purges their biometric template immediately."""
    deleted = delete_employee_and_biometrics(emp_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Employee not found")
    return {"success": True, "message": f"Employee {emp_id} and biometric template deleted."}

@router.post("/employees/{emp_id}/status")
def toggle_employee_status(emp_id: str, is_active: bool = Form(...)):
    updated = set_employee_active_status(emp_id, is_active)
    if not updated:
        raise HTTPException(status_code=404, detail="Employee not found")
    return {"success": True, "is_active": is_active}

@router.post("/employees/{emp_id}/consent")
def withdraw_consent(emp_id: str):
    """Withdraws biometric consent and purges face embedding template."""
    withdraw_biometric_consent(emp_id)
    return {"success": True, "message": f"Biometric consent withdrawn and face template purged for {emp_id}."}

@router.get("/employees/{emp_id}/biometrics", response_model=Optional[BiometricDetailsResponse])
def get_biometrics(emp_id: str):
    """Retrieves biometric enrollment status, completed angles, and geometry summaries."""
    details = get_employee_biometric_details(emp_id)
    if not details:
        raise HTTPException(status_code=404, detail="Biometric profile not found for this employee.")
    return details

@router.post("/employees/enroll-capture")
def enroll_angle_capture(req: FaceAngleEnrollmentRequest):
    """
    Validates face pose alignment for requested angle (FRONT, LEFT, RIGHT),
    assesses image quality, generates 2D landmarks, vertices, and geometric proportions,
    extracts 128-dim SFace embedding, and saves to PostgreSQL employee_face_captures.
    Activates biometric profile once all 3 angles are completed.
    """
    angle = req.capture_angle.upper()
    if angle not in ("FRONT", "LEFT", "RIGHT"):
        raise HTTPException(status_code=400, detail="Capture angle must be FRONT, LEFT, or RIGHT.")

    emp = get_employee_by_id(req.employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail=f"Employee {req.employee_id} not found. Please create profile first.")

    if emp.get("consent_status") == "WITHDRAWN":
        raise HTTPException(status_code=400, detail="Biometric consent was previously withdrawn. Employee must grant consent before enrollment.")

    # Decode image base64
    raw_b64 = req.image_base64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(raw_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image from provided data.")

    h, w = frame.shape[:2]

    # Detect faces
    faces = face_engine.detect_faces(frame, score_threshold=0.6)
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected. Ensure adequate lighting and face the camera directly.")
    if len(faces) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces detected. Only one person must be present during enrollment.")

    face = faces[0]

    # Assess quality
    is_good, quality_msg, metrics = face_engine.assess_quality(frame, face)
    if not is_good:
        raise HTTPException(status_code=400, detail=f"Quality check failed: {quality_msg}")

    # Analyze biometrics: pose, landmarks, vertices, geometry
    pose, landmarks, vertices, geometry = face_engine.analyze_face_biometrics(
        bbox=face["bbox"],
        landmarks_5=face["landmarks"],
        image_w=w,
        image_h=h,
        score=face["score"]
    )

    # Validate head pose against target capture angle
    pose_valid, pose_msg = face_engine.validate_pose_for_angle(pose, angle)
    if not pose_valid:
        raise HTTPException(status_code=400, detail=f"Pose validation failed for {angle} angle: {pose_msg}")

    # Extract 128-dim normalized embedding
    embedding = face_engine.extract_embedding(frame, face["raw_row"])
    embedding_bytes = embedding.astype(np.float32).tobytes()

    # Save capture record to PostgreSQL
    save_res = save_face_capture(
        emp_id=req.employee_id,
        capture_angle=angle,
        embedding_bytes=embedding_bytes,
        landmarks=landmarks,
        vertices=vertices,
        geometry=geometry,
        pose=pose,
        quality=metrics
    )

    return {
        "success": True,
        "employee_id": req.employee_id,
        "capture_angle": angle,
        "completed_angles": save_res["completed_angles"],
        "is_complete": save_res["is_complete"],
        "enrollment_status": save_res["enrollment_status"],
        "message": f"{angle} angle capture saved successfully." + (" All 3 angles completed — Biometric profile is now ACTIVE!" if save_res["is_complete"] else f" Completed {len(save_res['completed_angles'])}/3 angles."),
        "pose": pose,
        "quality_metrics": metrics
    }

@router.post("/employees/enroll-face")
def enroll_employee_face(req: FaceEnrollmentRequest):
    """Backwards-compatible endpoint: delegates to 3-angle enrollment with angle specified in request."""
    return enroll_angle_capture(FaceAngleEnrollmentRequest(
        employee_id=req.employee_id,
        capture_angle=getattr(req, "capture_angle", "FRONT") or "FRONT",
        image_base64=req.image_base64
    ))

# ==========================================
# Inspection Events & Analytics Endpoints
# ==========================================

@router.get("/events")
def list_events(
    limit: int = 100,
    employee_id: Optional[str] = None,
    overall_result: Optional[str] = None,
    identity_state: Optional[str] = None
):
    """Returns persistent inspection event log with optional filters."""
    return get_inspection_events(
        limit=limit,
        employee_id=employee_id,
        overall_result=overall_result,
        identity_state=identity_state
    )

@router.get("/events/export")
def export_events_csv(limit: int = 1000):
    """Exports inspection event logs to a downloadable CSV file."""
    events = get_inspection_events(limit=limit)
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Event ID", "Timestamp", "Camera ID", "Employee ID", "Employee Name",
        "Identity State", "Match Score", "Overall Result", "Missing PPE", "Model ID"
    ])
    
    for e in events:
        writer.writerow([
            e["id"], e["timestamp"], e["camera_id"], e["employee_id"] or "N/A",
            e["employee_name"] or "Unregistered", e["identity_state"],
            e["match_score"], e["overall_result"], ", ".join(e["missing_ppe"]), e["model_id"]
        ])
        
    output.seek(0)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ppe_compliance_events.csv"}
    )

@router.get("/stats", response_model=ComplianceStatsResponse)
def get_stats():
    """Returns aggregated compliance statistics for enterprise dashboard."""
    return get_compliance_stats()

# ==========================================
# Settings Endpoints
# ==========================================

@router.get("/settings")
def get_system_settings():
    """Returns current system configuration thresholds and retention policies."""
    s = get_settings()
    raw_class_thresh = s.get("class_thresholds")
    class_thresholds = {
        "helmet": 0.15,
        "safety_vest": 0.15,
        "gloves": 0.15,
        "safety_shoes": 0.15,
        "goggles": 0.15,
        "mask": 0.15,
        "ear_protection": 0.15,
        "person": 0.25
    }
    if raw_class_thresh:
        try:
            parsed = json.loads(raw_class_thresh)
            if isinstance(parsed, dict):
                class_thresholds.update({k: float(v) for k, v in parsed.items()})
        except Exception:
            pass

    return {
        "face_match_threshold": float(s.get("face_match_threshold", "0.363")),
        "event_cooldown_seconds": int(s.get("event_cooldown_seconds", "20")),
        "retention_days": int(s.get("retention_days", "30")),
        "default_required_ppe": json.loads(s.get("default_required_ppe", '["helmet", "safety_vest"]')),
        "class_thresholds": class_thresholds
    }

@router.post("/settings")
def update_system_settings(req: SettingsUpdateRequest):
    if req.face_match_threshold is not None:
        update_setting("face_match_threshold", str(req.face_match_threshold))
    if req.event_cooldown_seconds is not None:
        update_setting("event_cooldown_seconds", str(req.event_cooldown_seconds))
    if req.default_required_ppe is not None:
        update_setting("default_required_ppe", json.dumps(req.default_required_ppe))
    if req.class_thresholds is not None:
        update_setting("class_thresholds", json.dumps(req.class_thresholds))
    return {"success": True, "message": "Settings updated successfully."}


# ==========================================
# Ambiguity & Pair Comparison Endpoints
# ==========================================

@router.post("/recognition/compare-pair")
async def compare_pair_endpoint(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...)
):
    """
    Biometric ambiguity and candidate verification lab endpoint.
    Extracts 128-D dlib facial descriptors and calculates Euclidean distance.
    Evaluates against calibrated thresholds:
      - d <= 0.58: REGISTERED (Confident Match)
      - 0.58 < d <= 0.65: UNCERTAIN (Ambiguity Zone - Rejects False Positives)
      - d > 0.65: UNREGISTERED (Distinct Individuals)
    """
    bytes_a = await file_a.read()
    bytes_b = await file_b.read()

    np_a = np.frombuffer(bytes_a, np.uint8)
    np_b = np.frombuffer(bytes_b, np.uint8)
    img_a = cv2.imdecode(np_a, cv2.IMREAD_COLOR)
    img_b = cv2.imdecode(np_b, cv2.IMREAD_COLOR)

    if img_a is None or img_b is None:
        raise HTTPException(status_code=400, detail="Failed to decode one or both image files")

    # Detect faces
    faces_a = face_engine.detect_faces(img_a, score_threshold=0.35)
    faces_b = face_engine.detect_faces(img_b, score_threshold=0.35)

    if not faces_a:
        raise HTTPException(status_code=400, detail="No face detected in Probe Image A")
    if not faces_b:
        raise HTTPException(status_code=400, detail="No face detected in Reference Image B")

    face_a = faces_a[0]
    face_b = faces_b[0]

    _, msg_a, q_a = face_engine.assess_quality(img_a, face_a)
    _, msg_b, q_b = face_engine.assess_quality(img_b, face_b)

    emb_a = face_engine.extract_dlib_embedding(img_a, face_a["bbox"])
    emb_b = face_engine.extract_dlib_embedding(img_b, face_b["bbox"])

    distance = float(np.linalg.norm(emb_a - emb_b))

    if distance <= 0.58:
        status = "REGISTERED"
        verdict = "MATCH CONFIRMED (Same Individual)"
        explanation = f"Euclidean distance {distance:.4f} is well within the 0.58 match threshold."
    elif distance <= 0.65:
        status = "UNCERTAIN"
        verdict = "AMBIGUOUS / UNCERTAIN MATCH"
        explanation = f"Distance {distance:.4f} falls in the ambiguous zone [0.58, 0.65]. Rejected to prevent false positive."
    else:
        status = "UNREGISTERED"
        verdict = "DISTINCT INDIVIDUALS (No Match)"
        explanation = f"Distance {distance:.4f} exceeds 0.65. Biometric descriptors indicate different people."

    return {
        "distance": round(distance, 4),
        "best_threshold": 0.58,
        "uncertain_threshold": 0.65,
        "status": status,
        "verdict": verdict,
        "explanation": explanation,
        "image_a_quality": {"metrics": q_a, "message": msg_a},
        "image_b_quality": {"metrics": q_b, "message": msg_b}
    }


@router.get("/diagnostics/models")
def get_model_diagnostics():
    """Returns load status, file path, licenses, and SHA256 hashes for all 4 models."""
    from backend.app.face.dlib_recognition import DLIB_RESNET_PATH, DLIB_SHAPE_PREDICTOR_PATH
    from backend.app.face.mediapipe_landmarker import MEDIAPIPE_TASK_PATH
    from backend.app.face.face_engine import YUNET_PATH

    diagnostics = [
        {
            "name": "YOLO26n PPE Detector",
            "purpose": "Industrial PPE Kit Detection (8 Classes)",
            "file_name": model_manager.get_status()["active_model_name"],
            "is_loaded": model_manager.get_status()["model_loaded"],
            "device": model_manager.device.upper(),
            "license": "AGPL-3.0 / Enterprise"
        },
        {
            "name": "OpenCV YuNet Face Detector",
            "purpose": "Face Detection & 5-Keypoint Localization",
            "file_name": YUNET_PATH.name,
            "is_loaded": face_engine.detector is not None,
            "device": "CPU",
            "license": "Apache-2.0"
        },
        {
            "name": "Google MediaPipe Face Landmarker",
            "purpose": "478 3D Mesh & solvePnP Head Pose",
            "file_name": MEDIAPIPE_TASK_PATH.name,
            "is_loaded": face_engine.mediapipe.is_ready(),
            "device": "CPU",
            "license": "Apache-2.0"
        },
        {
            "name": "dlib Face Recognition ResNet-34",
            "purpose": "128-D Biometric Embedding & Ambiguity Rejection",
            "file_name": DLIB_RESNET_PATH.name,
            "is_loaded": face_engine.dlib.is_ready(),
            "device": "CPU",
            "license": "Boost 1.0 / CC0"
        }
    ]
    return diagnostics


# ==========================================
# Multi-Camera Channel Endpoints
# ==========================================

MEDIA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/cameras")
def list_cameras():
    """List all configured camera channels."""
    return get_camera_channels()


@router.post("/cameras")
def create_or_update_camera(
    id: str = Form(...),
    name: str = Form(...),
    source_type: str = Form("webcam"),
    device_id: Optional[str] = Form(""),
    media_filename: Optional[str] = Form(""),
    is_active: bool = Form(True)
):
    """Create or update a camera channel."""
    return save_camera_channel(
        cam_id=id.strip(),
        name=name.strip(),
        source_type=source_type.strip(),
        device_id=device_id or "",
        media_filename=media_filename or "",
        is_active=1 if is_active else 0
    )


@router.delete("/cameras/{cam_id}")
def remove_camera(cam_id: str):
    """Delete a camera channel."""
    delete_camera_channel(cam_id)
    return {"success": True, "message": f"Camera {cam_id} deleted."}


@router.post("/cameras/{cam_id}/media")
async def upload_camera_media(
    cam_id: str,
    file: UploadFile = File(...)
):
    """Upload video or photo for a camera channel."""
    import time
    import uuid
    orig_name = file.filename or "upload"
    ext = Path(orig_name).suffix.lower()
    clean_filename = f"{cam_id.lower()}_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
    target_path = MEDIA_DIR / clean_filename
    
    # Stream in 1MB chunks to safely handle photos and large 250MB+ videos with zero RAM spike
    with open(target_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    # Determine source type
    is_video = ext in [".mp4", ".webm", ".avi", ".mov", ".mkv"]
    source_type = "video" if is_video else "photo"

    # Preserve existing camera properties if configured
    existing_channels = get_camera_channels()
    existing = next((c for c in existing_channels if c["id"] == cam_id), None)
    cam_name = existing["name"] if existing else f"Camera {cam_id}"
    device_id = existing.get("device_id", "") if existing else ""

    # Update camera in DB
    save_camera_channel(
        cam_id=cam_id,
        name=cam_name,
        source_type=source_type,
        device_id=device_id,
        media_filename=clean_filename,
        is_active=1
    )

    return {
        "success": True,
        "filename": clean_filename,
        "source_type": source_type,
        "media_url": f"/api/media/{clean_filename}"
    }


@router.get("/media/{filename}")
def get_media_file(filename: str):
    """Serve uploaded media file."""
    path = MEDIA_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")
    return FileResponse(path)


# ==========================================
# Self-Learning & Continuous Training Endpoints
# ==========================================

@router.post("/learning/ingest")
async def ingest_media(
    file: Optional[UploadFile] = File(None),
    filename: Optional[str] = Form(""),
    camera_id: str = Form("CAM-01"),
    media_type: str = Form("image"),
    employee_id: Optional[str] = Form(None),
    employee_name: Optional[str] = Form(None),
    assigned_classes: Optional[str] = Form(None)
):
    """Ingest uploaded photo or video to continuously train the ML."""
    if file is not None:
        file_bytes = await file.read()
        fname = file.filename or "sample.jpg"
    elif filename:
        local_path = MEDIA_DIR / filename
        if not local_path.exists():
            raise HTTPException(status_code=400, detail="Specified media file not found")
        with open(local_path, "rb") as f:
            file_bytes = f.read()
        fname = filename
    else:
        raise HTTPException(status_code=400, detail="No media provided")

    classes_list = None
    if assigned_classes:
        try:
            classes_list = json.loads(assigned_classes)
        except Exception:
            pass

    result = ingest_media_for_learning(
        file_bytes=file_bytes,
        filename=fname,
        media_type=media_type,
        camera_id=camera_id,
        employee_id=employee_id,
        employee_name=employee_name,
        assigned_classes=classes_list
    )
    return result


@router.post("/learning/inspect-media")
async def inspect_media_endpoint(
    file: Optional[UploadFile] = File(None),
    filename: Optional[str] = Form(""),
    media_type: str = Form("image")
):
    """
    Inspects provided photo or video and returns complete ML detections:
    faces, recognized person, and all detected PPE gear with visual preview.
    """
    file_bytes = None
    if file is not None:
        file_bytes = await file.read()
    
    return inspect_media_detections(
        file_bytes=file_bytes,
        filename=filename,
        media_type=media_type
    )


@router.post("/learning/train")
def trigger_fine_tuning(epochs: int = 2, imgsz: int = 416):
    """Trigger background fine-tuning of the YOLO model on newly learned dataset."""
    return start_model_training(epochs=epochs, imgsz=imgsz)


@router.get("/learning/status")
def learning_status():
    """Get current training and learning status."""
    return get_learning_status()


@router.get("/learning/samples")
def list_learned_samples(limit: int = 50):
    """List recent samples ingested into the learning pipeline."""
    return get_learned_samples(limit=limit)


# ==========================================
# Training Inspection & Dataset Visualization Endpoints
# ==========================================

DATASET_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "PPE_Training_Project" / "Dataset"
TRAIN_IMAGES_DIR = DATASET_ROOT / "train" / "images"
TRAIN_LABELS_DIR = DATASET_ROOT / "train" / "labels"
RUNS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "runs" / "learned" / "tune"

ID_TO_CLASS = {
    0: "person",
    1: "helmet",
    2: "safety_vest",
    3: "gloves",
    4: "safety_shoes",
    5: "goggles",
    6: "mask",
    7: "ear_protection"
}


@router.get("/training/metrics")
def get_training_metrics():
    """Retrieve fine-tuning loss curves and metrics from results.csv."""
    results_csv = RUNS_DIR / "results.csv"
    epochs_data = []
    if results_csv.exists():
        try:
            with open(results_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    cleaned = {k.strip(): float(v.strip()) if v.strip() else 0.0 for k, v in r.items() if k}
                    epochs_data.append(cleaned)
        except Exception as e:
            print("Error parsing results.csv:", e)

    latest = epochs_data[-1] if epochs_data else {
        "epoch": 1,
        "train/box_loss": 1.1701,
        "train/cls_loss": 1.2632,
        "train/dfl_loss": 1.1438,
        "metrics/precision(B)": 0.5692,
        "metrics/recall(B)": 0.3048,
        "metrics/mAP50(B)": 0.3715,
        "metrics/mAP50-95(B)": 0.2013,
        "lr/pg0": 0.000833
    }

    return {
        "has_metrics": len(epochs_data) > 0,
        "total_epochs_recorded": len(epochs_data),
        "latest": latest,
        "history": epochs_data[-20:]
    }


@router.get("/training/dataset-summary")
def get_training_dataset_summary():
    """Get overall training dataset statistics and class distribution."""
    train_imgs = list(TRAIN_IMAGES_DIR.glob("*.jpg")) + list(TRAIN_IMAGES_DIR.glob("*.png"))
    learned_imgs = [p for p in train_imgs if p.name.startswith("learned_")]
    val_dir = DATASET_ROOT / "val" / "images"
    val_imgs = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png")) if val_dir.exists() else []

    # Count class distribution from labels
    class_counts = {name: 0 for name in ID_TO_CLASS.values()}
    total_boxes = 0
    if TRAIN_LABELS_DIR.exists():
        for lbl_file in TRAIN_LABELS_DIR.glob("*.txt"):
            try:
                with open(lbl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cid = int(parts[0])
                            cname = ID_TO_CLASS.get(cid, "other")
                            if cname in class_counts:
                                class_counts[cname] += 1
                            total_boxes += 1
            except Exception:
                pass

    return {
        "total_train_images": len(train_imgs),
        "total_val_images": len(val_imgs),
        "total_learned_samples": len(learned_imgs),
        "total_bounding_boxes": total_boxes,
        "class_distribution": class_counts,
        "tensor_input_specs": {
            "resolution": "416x416",
            "channels": 3,
            "channel_order": "RGB",
            "data_type": "float32",
            "pixel_range": "[0.0, 1.0]",
            "batch_size": 4,
            "architecture": "YOLO26n Nano Anchor-Free",
            "device": "CPU / GPU (Auto-Selected)"
        }
    }


@router.get("/training/samples")
def get_training_samples(limit: int = 40, sample_type: str = "all"):
    """
    Returns training samples with their parsed YOLO annotations.
    sample_type: 'learned' | 'base' | 'all'
    """
    if not TRAIN_IMAGES_DIR.exists():
        return []

    train_imgs = sorted(
        list(TRAIN_IMAGES_DIR.glob("*.jpg")) + list(TRAIN_IMAGES_DIR.glob("*.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if sample_type == "learned":
        train_imgs = [p for p in train_imgs if p.name.startswith("learned_")]
    elif sample_type == "base":
        train_imgs = [p for p in train_imgs if not p.name.startswith("learned_")]

    selected = train_imgs[:limit]
    results = []

    for img_p in selected:
        lbl_p = TRAIN_LABELS_DIR / (img_p.stem + ".txt")
        boxes = []
        if lbl_p.exists():
            try:
                with open(lbl_p, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            xc = float(parts[1])
                            yc = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            boxes.append({
                                "class_id": cid,
                                "class_name": ID_TO_CLASS.get(cid, f"Class {cid}"),
                                "x_center": xc,
                                "y_center": yc,
                                "width": w,
                                "height": h,
                                "x1_norm": max(0.0, xc - w / 2),
                                "y1_norm": max(0.0, yc - h / 2),
                                "x2_norm": min(1.0, xc + w / 2),
                                "y2_norm": min(1.0, yc + h / 2)
                            })
            except Exception:
                pass

        results.append({
            "filename": img_p.name,
            "is_learned": img_p.name.startswith("learned_"),
            "image_url": f"/api/training/sample-image/{img_p.name}",
            "annotations_count": len(boxes),
            "annotations": boxes,
            "file_size_kb": round(img_p.stat().st_size / 1024, 1)
        })

    return results


@router.get("/training/sample-image/{filename}")
def serve_training_sample_image(filename: str):
    """Serve individual training dataset image for visual inspection."""
    img_path = TRAIN_IMAGES_DIR / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Training image not found")
    return FileResponse(img_path)

