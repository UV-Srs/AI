"""
Integrated High-Performance Multimodal Inference Pipeline:
Combines YOLO26n PPE Detection + OpenCV YuNet Face Detection + SFace Facial Recognition.
Optimized for zero-lag real-time CPU streaming with in-memory caching and robust
multi-person spatial PPE Kit assignment.
"""

import time
import json
import uuid
import numpy as np
import cv2
import torch
from typing import Dict, List, Optional, Tuple, Any

from backend.app.services.model_manager import model_manager, APPROVED_CLASSES
from backend.app.face.face_engine import face_engine
from backend.app.face.face_tracking import face_tracker
from backend.app.db.database import (
    get_all_active_biometric_templates,
    get_all_employees,
    get_employee_by_id,
    log_inspection_event,
    get_settings
)

# Optimize CPU thread allocation to prevent thread thrashing
torch.set_num_threads(4)
cv2.setNumThreads(2)

# ==========================================
# In-Memory Environment Cache (Prevents DB queries on every frame)
# ==========================================
_cache_last_refresh: float = 0.0
_cached_active_templates: Dict[str, Dict[str, bytes]] = {}
_cached_settings: Dict[str, str] = {
    "face_match_threshold": "0.363",
    "face_match_margin": "0.050",
    "event_cooldown_seconds": "20",
    "default_required_ppe": '["helmet", "safety_vest"]'
}
_cached_employees: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECS = 8.0

# Cooldown cache: (entity_key, camera_id) -> last_event_timestamp
_cooldown_cache: Dict[Tuple[str, str], float] = {}

def get_cached_environment():
    global _cache_last_refresh, _cached_active_templates, _cached_settings, _cached_employees
    now = time.time()
    if now - _cache_last_refresh > CACHE_TTL_SECS:
        try:
            _cached_settings = get_settings()
            _cached_active_templates = get_all_active_biometric_templates()
            all_emps = get_all_employees()
            _cached_employees = {e["id"]: e for e in all_emps}
            _cache_last_refresh = now
        except Exception as e:
            # Fall back to existing cached values if transient DB issue
            pass
    return _cached_settings, _cached_active_templates, _cached_employees

def invalidate_pipeline_cache():
    global _cache_last_refresh
    _cache_last_refresh = 0.0


# ==========================================
# Geometric Utilities for Multi-Person Association
# ==========================================

def compute_box_center(box: List[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

def compute_box_area(box: List[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

def compute_intersection_area(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    return interW * interH

def is_point_inside(pt: Tuple[float, float], box: List[float], y_min_ratio: float = 0.0, y_max_ratio: float = 1.0) -> bool:
    x, y = pt
    bx1, by1, bx2, by2 = box
    bh = by2 - by1
    zone_y1 = by1 + bh * y_min_ratio
    zone_y2 = by1 + bh * y_max_ratio
    return (bx1 <= x <= bx2) and (zone_y1 <= y <= zone_y2)


def validate_ppe_candidate(
    cname: str,
    frame: np.ndarray,
    box: List[float],
    conf: float,
    face_box: Optional[List[float]] = None
) -> Tuple[bool, str]:
    """
    Validates whether a detected candidate is a genuine industrial PPE equipment item
    or an everyday false positive (e.g. tissue paper for helmet, normal shirt for vest,
    bare hands for gloves, slippers/bare feet for safety shoes, bare eyes for goggles).
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    frame_h, frame_w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)

    bw = x2 - x1
    bh = y2 - y1
    if bw < 4 or bh < 4:
        return False, "Box too small"

    aspect_ratio = bh / max(1.0, float(bw))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, "Empty crop"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h = float(np.mean(hsv[:, :, 0]))
    mean_s = float(np.mean(hsv[:, :, 1]))
    mean_v = float(np.mean(hsv[:, :, 2]))

    # 1. HELMET: Hard hat vs tissue paper / napkin / flat covering / hair
    if cname == "helmet":
        # Broad dome aspect ratio supporting standard, wide-brim, low-profile and bump cap hard hats
        if aspect_ratio < 0.20:
            return False, f"Aspect ratio too flat ({aspect_ratio:.2f} < 0.20) - likely paper/tissue"
        if aspect_ratio > 2.80:
            return False, f"Aspect ratio too tall ({aspect_ratio:.2f} > 2.80)"

        if face_box is not None:
            fx1, fy1, fx2, fy2 = face_box
            fw = max(1.0, float(fx2 - fx1))
            fh = max(1.0, float(fy2 - fy1))
            if bw / fw < 0.35 or bw / fw > 4.2:
                return False, f"Helmet width ratio ({bw/fw:.2f}) invalid for head"
            h_cx = (x1 + x2) / 2.0
            f_cx = (fx1 + fx2) / 2.0
            if abs(h_cx - f_cx) / fw > 0.95:
                return False, "Helmet not centered on head"
            if y1 > fy1 + fh * 0.50:
                return False, "Helmet top is below forehead"
            h_cy = (y1 + y2) / 2.0
            f_cy = (fy1 + fy2) / 2.0
            if h_cy >= f_cy:
                return False, "Helmet center is below face center"

        return True, "Valid industrial helmet"

    # 2. SAFETY VEST: High-vis vest vs everyday clothes / t-shirts / hoodies
    elif cname == "safety_vest":
        # Plain black/grey/white everyday clothes have low saturation (S < 30)
        if mean_s < 30.0 and mean_v < 180.0:
            return False, f"Lacks high-visibility safety coloration (saturation {mean_s:.1f} < 30.0) - ordinary clothing"

        # Check for standard high-vis hue (fluorescent yellow-green: 20-75, neon orange/red: 0-24 or 170-180)
        is_fluorescent_hue = (20 <= mean_h <= 75) or (mean_h <= 24) or (mean_h >= 170)
        if not is_fluorescent_hue:
            # Non-fluorescent clothing (e.g. casual blue shirt, dark jacket)
            # Require higher confidence (>= 0.18) to prevent ordinary shirts from false triggering
            if conf < 0.18:
                return False, f"Non-fluorescent clothing confidence ({conf:.2f} < 0.18) too low for safety vest"

        if aspect_ratio < 0.55:
            return False, f"Vest aspect ratio too flat ({aspect_ratio:.2f} < 0.55)"
        if aspect_ratio > 2.60:
            return False, f"Vest aspect ratio too tall ({aspect_ratio:.2f} > 2.60)"

        if conf < 0.08:
            return False, f"Safety vest confidence too low ({conf:.2f} < 0.08)"

        return True, "Valid high-vis safety vest"

    # 3. GLOVES: Industrial work gloves vs bare human hands
    elif cname == "gloves":
        if conf < 0.12:
            return False, f"Gloves confidence too low ({conf:.2f} < 0.12)"

        # Check for bare skin in YCrCb: human skin tone clusters tightly
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        skin_ratio = float(np.mean(skin_mask))

        if skin_ratio > 0.65:
            return False, f"Detected bare human skin ({skin_ratio:.1%}) - not a safety glove"

        return True, "Valid industrial safety glove"

    # 4. SAFETY SHOES: Heavy work boots vs bare feet / flat slippers
    elif cname == "safety_shoes":
        if conf < 0.12:
            return False, f"Safety shoes confidence too low ({conf:.2f} < 0.12)"

        # Flat house slippers or bare feet have flat aspect ratio < 0.35
        if aspect_ratio < 0.35:
            return False, f"Shoe aspect ratio too flat ({aspect_ratio:.2f} < 0.35) - likely slippers/bare foot"

        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        skin_ratio = float(np.mean(skin_mask))
        if skin_ratio > 0.65:
            return False, f"Detected bare human foot skin ({skin_ratio:.1%}) - not a safety shoe"

        return True, "Valid industrial safety shoe"

    # 5. GOGGLES: Protective eyewear vs bare eyes / reading glasses
    elif cname == "goggles":
        if conf < 0.12:
            return False, f"Goggles confidence too low ({conf:.2f} < 0.12)"

        if face_box is not None:
            fx1, fy1, fx2, fy2 = face_box
            fw = max(1.0, float(fx2 - fx1))
            fh = max(1.0, float(fy2 - fy1))
            g_cx = (x1 + x2) / 2.0
            g_cy = (y1 + y2) / 2.0
            f_cx = (fx1 + fx2) / 2.0

            # Must sit across the eye region (10% to 60% down from face top)
            y_rel = (g_cy - fy1) / fh
            if y_rel < 0.10 or y_rel > 0.60:
                return False, f"Goggles not at eye line (relative Y {y_rel:.2f} outside 0.10-0.60)"

            if abs(g_cx - f_cx) / fw > 0.40:
                return False, "Goggles not centered across eyes"

            if bw / fw < 0.40 or bw / fw > 1.25:
                return False, f"Goggles width ratio ({bw/fw:.2f}) invalid for eyes"

        return True, "Valid protective safety goggles"

    # 6. MASK: Protective facial mask vs bare mouth / beard
    elif cname == "mask":
        if conf < 0.10:
            return False, f"Mask confidence too low ({conf:.2f} < 0.10)"

        if face_box is not None:
            fx1, fy1, fx2, fy2 = face_box
            fh = max(1.0, float(fy2 - fy1))
            m_cy = (y1 + y2) / 2.0
            # Must sit across lower face / nose & mouth (30% to 105% down from face top)
            y_rel = (m_cy - fy1) / fh
            if y_rel < 0.30 or y_rel > 1.05:
                return False, f"Mask not over mouth/nose (relative Y {y_rel:.2f} outside 0.30-1.05)"

        return True, "Valid protective face mask"

    # 7. EAR PROTECTION: Industrial earmuffs / noise protectors
    elif cname == "ear_protection":
        if conf < 0.10:
            return False, f"Ear protection confidence too low ({conf:.2f} < 0.10)"

        return True, "Valid industrial ear protection"

    return True, "Valid detection"


def validate_helmet_candidate(
    frame: np.ndarray,
    helmet_box: List[float],
    conf: float,
    matched_face_box: Optional[List[float]] = None
) -> Tuple[bool, str]:
    """Compatibility alias for helmet validation."""
    return validate_ppe_candidate("helmet", frame, helmet_box, conf, face_box=matched_face_box)


# ==========================================
# Core Multimodal Pipeline
# ==========================================

def run_integrated_pipeline(
    image_bytes: bytes,
    ppe_conf_threshold: float = 0.08,
    face_match_threshold: Optional[float] = None,
    class_thresholds: Optional[Dict[str, float]] = None,
    camera_id: str = "CAM-01",
    imgsz: int = 416
) -> Dict[str, Any]:
    """
    Executes combined live recognition & PPE inspection on a single frame.
    Supports multiple people in the camera frame simultaneously, evaluating whether
    each person has worn the required PPE kit with configurable per-class confidence thresholds.
    """
    start_time = time.perf_counter()

    # Fast cached environment lookup
    settings, active_templates, employees_map = get_cached_environment()

    if face_match_threshold is None:
        face_match_threshold = float(settings.get("face_match_threshold", "0.363"))
    match_margin_threshold = float(settings.get("face_match_margin", "0.050"))
    cooldown_secs = float(settings.get("event_cooldown_seconds", "3"))
    default_req_ppe = json.loads(settings.get("default_required_ppe", '["helmet", "safety_vest"]'))

    # Resolve per-class confidence thresholds (permits custom sensitivity for different hard hats/vests)
    default_class_thresholds = {
        "helmet": ppe_conf_threshold,
        "safety_vest": ppe_conf_threshold,
        "gloves": ppe_conf_threshold,
        "safety_shoes": ppe_conf_threshold,
        "goggles": ppe_conf_threshold,
        "mask": ppe_conf_threshold,
        "ear_protection": ppe_conf_threshold,
        "person": max(0.20, ppe_conf_threshold)
    }

    # Load persisted class thresholds from settings if available
    saved_raw_thresh = settings.get("class_thresholds")
    if saved_raw_thresh:
        try:
            saved_thresh = json.loads(saved_raw_thresh)
            if isinstance(saved_thresh, dict):
                default_class_thresholds.update({k: float(v) for k, v in saved_thresh.items()})
        except Exception:
            pass

    # Dynamic websocket payload overrides take immediate precedence
    if class_thresholds and isinstance(class_thresholds, dict):
        effective_thresholds = {**default_class_thresholds, **{k: float(v) for k, v in class_thresholds.items()}}
    else:
        effective_thresholds = default_class_thresholds

    # Ensure human person detection is always sensitive (0.12 - 0.25) so EVERY person entering the camera frame is detected
    effective_thresholds["person"] = min(0.30, max(0.12, float(effective_thresholds.get("person", 0.20))))

    # 1. Decode frame
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"success": False, "error": "Failed to decode frame", "persons": [], "fps": 0, "inference_time_ms": 0}

    frame_h, frame_w = frame.shape[:2]

    # 2. Run Face Detection (YuNet) + 478 3D Mesh (MediaPipe) + Recognition (dlib & SFace)
    # Optimized for multi-person tracking with zero CPU lag
    t_yunet_0 = time.perf_counter()
    raw_faces = face_engine.detect_faces(frame, score_threshold=0.30)
    yunet_time_ms = round((time.perf_counter() - t_yunet_0) * 1000.0, 1)

    # Filter out tiny distant noise (< 20px) and rank prominent faces first
    raw_faces = [f for f in raw_faces if (f["bbox"][2] - f["bbox"][0]) >= 20 and (f["bbox"][3] - f["bbox"][1]) >= 20]
    raw_faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
    raw_faces = raw_faces[:12]  # Detect up to 12 people simultaneously in frame

    # Associate with multi-person temporal tracker
    detected_bboxes = [f["bbox"] for f in raw_faces]
    tracked_associations = face_tracker.update(detected_bboxes)
    # Map bbox tuple to track instance for O(1) lookup
    bbox_to_track = {tuple(bbox): track for _, bbox, track in tracked_associations}

    processed_faces = []
    mediapipe_accum_ms = 0.0
    dlib_accum_ms = 0.0
    now_ts = time.time()
    dlib_budget_used = False

    for face in raw_faces:
        track = bbox_to_track.get(tuple(face["bbox"]))
        is_good_quality, quality_msg, quality_metrics = face_engine.assess_quality(frame, face)
        liveness_state, liveness_score = face_engine.check_liveness(frame, face)

        # A. Google MediaPipe 478 3D Facial Mesh & Pose (Cached across ~15 frames to prevent CPU lag)
        t_mp0 = time.perf_counter()
        if track and track.cached_mesh_data is not None and (now_ts - track.last_mesh_time) < 0.45:
            mesh_data = track.cached_mesh_data
        else:
            mesh_data = face_engine.get_face_mesh(frame, face["bbox"])
            if track:
                track.cached_mesh_data = mesh_data
                track.last_mesh_time = now_ts
        mediapipe_accum_ms += (time.perf_counter() - t_mp0) * 1000.0

        # B. dlib 128-D Biometric Embedding (Cached per tracked person; max 1 heavy extraction per frame)
        t_dl0 = time.perf_counter()
        dlib_emb = None
        if track and track.cached_dlib_descriptor is not None and (now_ts - track.last_dlib_time) < 3.0:
            # Instant 0ms reuse for tracked person
            dlib_emb = track.cached_dlib_descriptor
        else:
            # Need fresh descriptor: check frame budget to prevent multi-person CPU stutter
            if not dlib_budget_used and (is_good_quality or face["score"] >= 0.45):
                dlib_emb_arr = face_engine.extract_dlib_embedding(frame, face["bbox"])
                dlib_emb = dlib_emb_arr.tolist() if dlib_emb_arr is not None else []
                if track:
                    track.cached_dlib_descriptor = dlib_emb
                    track.last_dlib_time = now_ts
                dlib_budget_used = True
            elif track and track.cached_dlib_descriptor is not None:
                dlib_emb = track.cached_dlib_descriptor
            else:
                dlib_emb = []
        dlib_accum_ms += (time.perf_counter() - t_dl0) * 1000.0
        
        emp_id = None
        emp_name = None
        dept = None
        desig = None
        match_score = 0.0
        second_match_score = 0.0
        match_margin = 0.0
        identity_state = "UNCERTAIN"

        if (is_good_quality or face["score"] >= 0.40) and liveness_state != "FAIL":
            # Check if tracked person already has confirmed match within 2.5 seconds
            if track and track.cached_match_data is not None and (now_ts - track.last_match_time) < 2.5:
                cached_m = track.cached_match_data
                emp_id = cached_m["emp_id"]
                match_score = cached_m["match_score"]
                second_match_score = cached_m["second_match_score"]
                match_margin = cached_m["match_margin"]
                identity_state = cached_m["identity_state"]
            else:
                # Fast SFace Match against active templates (~15ms)
                embedding = face_engine.extract_embedding(frame, face["raw_row"])
                matched_id, score, second_score, margin, raw_state = face_engine.match_face(
                    embedding, active_templates, threshold=face_match_threshold, margin_threshold=match_margin_threshold
                )
                match_score = score
                second_match_score = second_score
                match_margin = margin
                
                # Confirm over consecutive frames to prevent flicker
                track_key = track.track_id if track else f"{int(face['bbox'][0]//40)}_{int(face['bbox'][1]//40)}"
                emp_id, match_score, identity_state = face_engine.confirm_multi_frame(
                    track_key, matched_id, match_score, raw_state
                )

                if track:
                    track.cached_match_data = {
                        "emp_id": emp_id,
                        "match_score": match_score,
                        "second_match_score": second_match_score,
                        "match_margin": match_margin,
                        "identity_state": identity_state
                    }
                    track.last_match_time = now_ts

            if emp_id and identity_state == "REGISTERED":
                emp_record = employees_map.get(emp_id)
                if emp_record:
                    emp_name = emp_record.get("full_name") or emp_record.get("name")
                    dept = emp_record.get("department")
                    desig = emp_record.get("designation")
        else:
            identity_state = "UNCERTAIN" if not is_good_quality else "LIVENESS_REVIEW"

        processed_faces.append({
            "bbox": face["bbox"],
            "score": face["score"],
            "quality": {"acceptable": is_good_quality, "message": quality_msg, "metrics": quality_metrics},
            "liveness": {"state": liveness_state, "score": liveness_score},
            "mesh": mesh_data,
            "pose": mesh_data.get("pose") if mesh_data else None,
            "dlib_descriptor": dlib_emb if isinstance(dlib_emb, list) else (dlib_emb.tolist() if dlib_emb is not None else []),
            "employee_id": emp_id,
            "employee_name": emp_name or ("Unregistered Subject" if identity_state == "UNREGISTERED" else "Verifying Identity"),
            "department": dept or "Operations",
            "designation": desig or "Personnel",
            "match_score": match_score,
            "second_match_score": second_match_score,
            "match_margin": match_margin,
            "identity_state": identity_state
        })

    # 3. Run YOLO26n PPE Detection
    t_yolo_0 = time.perf_counter()
    yolo_model = model_manager.get_active_model()
    person_boxes: List[Tuple[List[float], float]] = []
    ppe_detections: List[Dict[str, Any]] = []

    if yolo_model is not None:
        min_thresh = min(effective_thresholds.values()) if effective_thresholds else ppe_conf_threshold
        internal_conf = max(0.010, min(min_thresh, 0.03))
        with model_manager.model_lock, torch.inference_mode():
            yolo_res = yolo_model.predict(
                source=frame,
                conf=internal_conf,
                iou=0.65,
                imgsz=imgsz,
                device=model_manager.device,
                verbose=False
            )[0]
        yolo_time_ms = round((time.perf_counter() - t_yolo_0) * 1000.0, 1)

        raw_peak_scores: Dict[str, float] = {}
        if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
            canonical_map = {
                "helmet": "helmet", "safety-vest": "safety_vest", "vest": "safety_vest",
                "gloves": "gloves", "shoes": "safety_shoes", "boots": "safety_shoes", "safety_shoes": "safety_shoes",
                "glasses": "goggles", "goggles": "goggles", "face-mask": "mask", "mask": "mask",
                "ear-mufs": "ear_protection", "ear_protection": "ear_protection", "person": "person"
            }
            model_names = getattr(yolo_model, "names", {})
            for b in yolo_res.boxes:
                cid = int(b.cls[0].item())
                raw_cname = model_names.get(cid, APPROVED_CLASSES[cid] if 0 <= cid < len(APPROVED_CLASSES) else f"class_{cid}")
                cname = canonical_map.get(str(raw_cname).lower(), str(raw_cname).lower())
                conf = float(b.conf[0].item())
                if conf > raw_peak_scores.get(cname, 0.0):
                    raw_peak_scores[cname] = round(conf, 3)
                
                # Apply per-class confidence threshold (user-configurable per PPE class)
                target_thresh = effective_thresholds.get(cname, ppe_conf_threshold)
                if conf < target_thresh:
                    continue

                xyxy = [round(float(v), 2) for v in b.xyxy[0].tolist()]
                if cname == "person":
                    person_boxes.append((xyxy, conf))
                else:
                    # Validate all candidate PPE gear against authentic industrial standards
                    # (rejects tissue for helmets, ordinary shirts for vests, bare skin for gloves/shoes)
                    is_valid, _ = validate_ppe_candidate(cname, frame, xyxy, conf, face_box=None)
                    if not is_valid:
                        continue
                    ppe_detections.append({
                        "class_id": cid,
                        "class_name": cname,
                        "confidence": round(conf, 4),
                        "box": xyxy,
                        "assigned_to": None
                    })

    # 4. Multi-Person Candidate Construction: Face-Anchored Fusion Engine
    # Prioritizes high-confidence faces as anatomical ground truth, then harmonizes
    # with YOLO person bounding boxes to eliminate asymmetric slicing and suppress phantom background noise.
    person_candidates: List[Dict[str, Any]] = []
    assigned_yolo_indices = set()

    # Pass 1: Build candidates from detected human faces (guarantees centered, complete upper-body framing)
    for f_idx, f_data in enumerate(processed_faces):
        fx1, fy1, fx2, fy2 = f_data["bbox"]
        fw = max(1.0, float(fx2 - fx1))
        fh = max(1.0, float(fy2 - fy1))
        fcx = (fx1 + fx2) / 2.0

        # Anatomical torso anchor: symmetrical shoulder span ~2.8x face width
        half_shoulder = max(fw * 1.45, fh * 1.15)
        ideal_left = max(0.0, fcx - half_shoulder)
        ideal_right = min(float(frame_w), fcx + half_shoulder)
        ideal_top = max(0.0, fy1 - fh * 0.45)
        ideal_bot = min(float(frame_h), fy2 + fh * 3.2)

        # Look for overlapping YOLO person box
        best_pbox = None
        best_p_idx = None
        max_overlap = 0.0

        for p_idx, (pbox, p_conf) in enumerate(person_boxes):
            if p_idx in assigned_yolo_indices:
                continue
            inter = compute_intersection_area(f_data["bbox"], pbox)
            if inter > max_overlap:
                max_overlap = inter
                best_pbox = pbox
                best_p_idx = p_idx

        # If no direct overlap, check if face center sits within upper region of person box
        if best_pbox is None:
            for p_idx, (pbox, p_conf) in enumerate(person_boxes):
                if p_idx in assigned_yolo_indices:
                    continue
                if is_point_inside((fcx, (fy1 + fy2)/2.0), pbox, y_min_ratio=-0.20, y_max_ratio=0.60):
                    best_pbox = pbox
                    best_p_idx = p_idx
                    break

        if best_pbox is not None and best_p_idx is not None:
            assigned_yolo_indices.add(best_p_idx)
            # Fuse YOLO box with anatomical anchor:
            # Symmetrically balance so user's shoulders are never chopped or skewed to one side
            fused_y1 = min(best_pbox[1], ideal_top)
            fused_y2 = max(best_pbox[3], ideal_bot)
            
            # Bound horizontal shoulder span symmetrically around face center
            left_span = fcx - min(best_pbox[0], ideal_left)
            right_span = max(best_pbox[2], ideal_right) - fcx
            # Harmonize: prevent skew where one side is bloated with background objects
            balanced_span = min(max(left_span, right_span), max(half_shoulder * 1.35, 120.0))
            fused_x1 = max(0.0, fcx - balanced_span)
            fused_x2 = min(float(frame_w), fcx + balanced_span)
            final_pbox = [round(fused_x1, 1), round(fused_y1, 1), round(fused_x2, 1), round(fused_y2, 1)]
        else:
            final_pbox = [round(ideal_left, 1), round(ideal_top, 1), round(ideal_right, 1), round(ideal_bot, 1)]

        person_candidates.append({
            "id": f"person_{f_idx + 1}",
            "box": final_pbox,
            "face": f_data,
            "is_virtual": (best_pbox is None)
        })

    # Pass 2: Consider all remaining YOLO person boxes (every person visible in frame)
    for p_idx, (pbox, p_conf) in enumerate(person_boxes):
        if p_idx in assigned_yolo_indices:
            continue
        bw = pbox[2] - pbox[0]
        bh = pbox[3] - pbox[1]
        # Allow people sitting, kneeling, leaning, or visible from chest up
        if bw < 20 or bh < 25 or (bw * bh) < 600:
            continue

        person_candidates.append({
            "id": f"person_body_{p_idx + 1}",
            "box": pbox,
            "face": None,
            "is_virtual": False
        })

    # Deduplicate overlapping candidates that represent the same human (IoU > 0.60)
    deduped_candidates: List[Dict[str, Any]] = []
    for cand in person_candidates:
        cbox = cand["box"]
        merged = False
        for existing in deduped_candidates:
            ebox = existing["box"]
            inter = compute_intersection_area(cbox, ebox)
            union = compute_box_area(cbox) + compute_box_area(ebox) - inter
            iou = inter / max(1.0, union)
            if iou > 0.60:
                if cand["face"] and not existing["face"]:
                    existing["face"] = cand["face"]
                merged = True
                break
        if not merged:
            deduped_candidates.append(cand)
    person_candidates = deduped_candidates

    # If NO persons and NO faces detected, but PPE items (e.g. helmet + vest) are found:
    # Cluster them into a virtual person
    if len(person_candidates) == 0 and len(ppe_detections) > 0:
        min_x = min(p["box"][0] for p in ppe_detections)
        min_y = min(p["box"][1] for p in ppe_detections)
        max_x = max(p["box"][2] for p in ppe_detections)
        max_y = max(p["box"][3] for p in ppe_detections)
        # Expand slightly
        pad_x = (max_x - min_x) * 0.15
        pad_y = (max_y - min_y) * 0.15
        clustered_box = [
            max(0.0, min_x - pad_x),
            max(0.0, min_y - pad_y),
            min(float(frame_w), max_x + pad_x),
            min(float(frame_h), max_y + pad_y)
        ]
        person_candidates.append({
            "id": "person_ppe_cluster_1",
            "box": clustered_box,
            "face": None,
            "is_virtual": True
        })

    # 5. Multi-Person Spatial PPE Assignment
    # Assign each PPE detection to the closest/most overlapping person candidate
    person_ppe_map: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(person_candidates))}

    for ppe in ppe_detections:
        ppe_cx, ppe_cy = compute_box_center(ppe["box"])
        cname = ppe["class_name"]

        best_cand_idx = None
        min_dist = float("inf")

        for cand_idx, cand in enumerate(person_candidates):
            cbox = cand["box"]
            cb_cx, cb_cy = compute_box_center(cbox)
            cb_w = cbox[2] - cbox[0]
            cb_h = cbox[3] - cbox[1]

            # Check vertical containment zone according to PPE equipment type
            in_zone = False
            if cname == "helmet":
                # Helmets sit at the top of the person (from top -15% down to 40%)
                in_zone = is_point_inside((ppe_cx, ppe_cy), cbox, y_min_ratio=-0.20, y_max_ratio=0.42)
            elif cname == "safety_vest":
                # Vests sit on chest/torso (from 15% to 80%)
                in_zone = is_point_inside((ppe_cx, ppe_cy), cbox, y_min_ratio=0.10, y_max_ratio=0.85)
            elif cname == "gloves":
                # Gloves sit around waist/arms (from 25% to 90%)
                in_zone = is_point_inside((ppe_cx, ppe_cy), cbox, y_min_ratio=0.25, y_max_ratio=0.92)
            elif cname == "safety_shoes":
                # Shoes sit at bottom (from 65% to 110%)
                in_zone = is_point_inside((ppe_cx, ppe_cy), cbox, y_min_ratio=0.65, y_max_ratio=1.15)
            elif cname == "goggles":
                # Goggles sit near eyes (from -5% to 35%)
                in_zone = is_point_inside((ppe_cx, ppe_cy), cbox, y_min_ratio=-0.10, y_max_ratio=0.35)

            if in_zone:
                # Calculate horizontal distance to person's centerline
                dist_x = abs(ppe_cx - cb_cx)
                if dist_x < min_dist:
                    # Spatial alignment check with detected face
                    face_box = cand["face"]["bbox"] if cand.get("face") and cand["face"].get("bbox") else None
                    if face_box is not None and cname in ["helmet", "goggles"]:
                        is_valid, _ = validate_ppe_candidate(
                            cname, frame, ppe["box"], ppe["confidence"], face_box=face_box
                        )
                        if not is_valid:
                            continue
                    min_dist = dist_x
                    best_cand_idx = cand_idx

        if best_cand_idx is not None:
            ppe["assigned_to"] = best_cand_idx
            person_ppe_map[best_cand_idx].append(ppe)

    # 6. Build Final Multi-Person Output & PPE Kit Compliance Status
    persons_output = []

    for cand_idx, cand in enumerate(person_candidates):
        matched_face = cand["face"]
        associated_ppe = person_ppe_map.get(cand_idx, [])
        pbox = cand["box"]

        # Determine required PPE based on employee profile if matched
        req_list = default_req_ppe
        emp_display_name = None
        if matched_face and matched_face.get("employee_id"):
            emp_rec = employees_map.get(matched_face["employee_id"])
            if emp_rec and emp_rec.get("required_ppe"):
                req_list = emp_rec["required_ppe"]
                emp_display_name = emp_rec.get("full_name") or emp_rec.get("name")

        detected_classes = {p["class_name"] for p in associated_ppe}
        checklist = {}
        missing_ppe = []

        # Standard check for all standard PPE classes
        for item in ["helmet", "safety_vest", "gloves", "safety_shoes", "goggles", "mask", "ear_protection"]:
            if item in req_list:
                if item in detected_classes:
                    checklist[item] = "DETECTED"
                else:
                    checklist[item] = "MISSING"
                    missing_ppe.append(item)
            else:
                checklist[item] = "DETECTED" if item in detected_classes else "NOT_REQUIRED"

        # Explicit PPE Kit evaluation
        has_helmet = "helmet" in detected_classes
        has_vest = "safety_vest" in detected_classes
        ppe_kit_worn = (len(missing_ppe) == 0)

        if ppe_kit_worn:
            overall_result = "COMPLIANT"
            ppe_kit_status = "PPE KIT WORN"
        else:
            overall_result = "NON_COMPLIANT"
            ppe_kit_status = "PPE KIT NOT WORN"

        if matched_face and matched_face.get("identity_state") in ["UNCERTAIN", "LIVENESS_REVIEW"]:
            if overall_result == "COMPLIANT":
                overall_result = "NEEDS_REVIEW"

        person_label = emp_display_name or f"Person {cand_idx + 1}"

        persons_output.append({
            "person_id": f"P-{cand_idx + 1}",
            "person_label": person_label,
            "person_box": [round(v, 1) for v in pbox],
            "face": matched_face,
            "associated_ppe": associated_ppe,
            "checklist": checklist,
            "missing_ppe": missing_ppe,
            "required_ppe": req_list,
            "ppe_kit_worn": ppe_kit_worn,
            "ppe_kit_status": ppe_kit_status,
            "has_helmet": has_helmet,
            "has_vest": has_vest,
            "overall_result": overall_result
        })

    # 7. Audit Event Logging (with cooldown to prevent flooding)
    now_ts = time.time()
    for p in persons_output:
        face_info = p.get("face")
        emp_id = face_info.get("employee_id") if face_info else None
        emp_name = face_info.get("employee_name") if face_info else None
        identity_state = face_info.get("identity_state", "UNREGISTERED") if face_info else "UNREGISTERED"
        match_score = face_info.get("match_score", 0.0) if face_info else 0.0
        second_score = face_info.get("second_match_score", 0.0) if face_info else 0.0
        margin = face_info.get("match_margin", 0.0) if face_info else 0.0

        cooldown_key = (emp_id or p["person_id"], camera_id)
        last_logged = _cooldown_cache.get(cooldown_key, 0.0)

        # Log if cooldown expired
        if (now_ts - last_logged) >= cooldown_secs:
            if len(_cooldown_cache) > 1000:
                _cooldown_cache.clear()
            _cooldown_cache[cooldown_key] = now_ts
            event_id = f"EVT-{int(now_ts)}-{uuid.uuid4().hex[:6]}"
            try:
                log_inspection_event(
                    event_id=event_id,
                    camera_id=camera_id,
                    candidate_employee_id=emp_id,
                    candidate_name=emp_name,
                    best_match_score=match_score,
                    second_best_match_score=second_score,
                    match_margin=margin,
                    identity_status=identity_state,
                    ppe_summary=p["checklist"],
                    missing_ppe=p["missing_ppe"],
                    overall_result=p["overall_result"],
                    model_version=model_manager.active_model_id or "combined_sh17"
                )
            except Exception:
                pass

    inf_time_ms = round((time.perf_counter() - start_time) * 1000.0, 1)
    fps = round(1000.0 / inf_time_ms, 1) if inf_time_ms > 0 else 0.0

    return {
        "success": True,
        "persons": persons_output,
        "unassociated_ppe": [p for p in ppe_detections if p["assigned_to"] is None],
        "total_persons": len(persons_output),
        "compliant_count": sum(1 for p in persons_output if p["overall_result"] == "COMPLIANT"),
        "violation_count": sum(1 for p in persons_output if p["overall_result"] == "NON_COMPLIANT"),
        "needs_review_count": sum(1 for p in persons_output if p["overall_result"] == "NEEDS_REVIEW"),
        "inference_time_ms": inf_time_ms,
        "fps": fps,
        "telemetry": {
            "fps": fps,
            "total_time_ms": inf_time_ms,
            "yunet_time_ms": yunet_time_ms,
            "mediapipe_time_ms": round(mediapipe_accum_ms, 1),
            "dlib_time_ms": round(dlib_accum_ms, 1),
            "yolo_time_ms": locals().get("yolo_time_ms", 0.0)
        },
        "camera_id": camera_id,
        "class_thresholds": effective_thresholds,
        "peak_scores": locals().get("raw_peak_scores", {}),
        "frame_width": frame_w,
        "frame_height": frame_h
    }
