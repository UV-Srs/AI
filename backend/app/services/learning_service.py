import os
import cv2
import json
import time
import uuid
import shutil
import base64
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

from backend.app.face.face_engine import face_engine
from backend.app.services.model_manager import model_manager, APPROVED_CLASSES
from backend.app.db.database import (
    create_or_update_employee, save_face_capture, record_learned_sample, get_learned_samples
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = BASE_DIR / "Datasets" / "Learned"
TRAIN_IMG_DIR = DATASET_DIR / "train" / "images"
TRAIN_LBL_DIR = DATASET_DIR / "train" / "labels"
VAL_IMG_DIR = DATASET_DIR / "val" / "images"
VAL_LBL_DIR = DATASET_DIR / "val" / "labels"
DATA_YAML_PATH = DATASET_DIR / "data.yaml"

CLASS_MAP = {
    "person": 0,
    "helmet": 1,
    "safety_vest": 2,
    "gloves": 3,
    "safety_shoes": 4,
    "goggles": 5,
    "mask": 6,
    "ear_protection": 7
}

# Ensure directories exist
for d in [TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if not DATA_YAML_PATH.exists():
    import yaml
    with open(DATA_YAML_PATH, "w") as f:
        yaml.dump({
            "path": str(DATASET_DIR.resolve()).replace("\\", "/"),
            "train": "train/images",
            "val": "val/images",
            "names": {
                0: "person",
                1: "helmet",
                2: "safety_vest",
                3: "gloves",
                4: "safety_shoes",
                5: "goggles",
                6: "mask",
                7: "ear_protection"
            }
        }, f)

# Global background training state and multi-job queue
_training_lock = threading.Lock()
_training_queue: List[Dict[str, Any]] = []
_training_state = {
    "status": "idle",  # "idle" | "training" | "completed" | "failed"
    "progress": 0,
    "epoch": 0,
    "total_epochs": 0,
    "queue_length": 0,
    "message": "Ready to learn",
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
}


def get_learning_status() -> Dict[str, Any]:
    with _training_lock:
        state = dict(_training_state)
        state["queue_length"] = len(_training_queue)
    # Count total learned samples in dataset
    learned_files = list(TRAIN_IMG_DIR.glob("learned_*.jpg"))
    state["total_learned_samples"] = len(learned_files)
    return state


def _process_single_frame(
    frame_bgr: np.ndarray,
    camera_id: str,
    employee_id: Optional[str] = None,
    employee_name: Optional[str] = None,
    assigned_classes: Optional[List[str]] = None
) -> Tuple[bool, List[str], bool]:
    """
    Processes one frame:
    1. Extracts facial biometrics and enrolls if employee specified.
    2. Runs YOLO, predicts PPE, formats standard YOLO .txt label, and writes to train dataset.
    Returns (success, classes_found, face_enrolled)
    """
    h, w = frame_bgr.shape[:2]
    if h < 32 or w < 32:
        return False, [], False

    classes_found: List[str] = []
    face_enrolled = False

    # 1. Face Biometric Learning
    if face_engine is not None and (employee_name or employee_id):
        raw_faces = face_engine.detect_faces(frame_bgr, score_threshold=0.40)
        if raw_faces:
            best_face = max(raw_faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
            dlib_emb = face_engine.extract_dlib_embedding(frame_bgr, best_face["bbox"])
            if dlib_emb is not None:
                # Ensure employee exists
                emp_id = employee_id or f"EMP-{uuid.uuid4().hex[:6].upper()}"
                emp_name = employee_name or f"Employee {emp_id}"
                create_or_update_employee(
                    emp_id=emp_id,
                    name=emp_name,
                    department="Field Operations",
                    designation="Learned Subject",
                    required_ppe=assigned_classes or ["helmet", "safety_vest"],
                    consent_status="ACTIVE",
                    is_active=True
                )
                _, _, metrics = face_engine.assess_quality(frame_bgr, best_face)
                # Encode face crop
                x1, y1, x2, y2 = [int(v) for v in best_face["bbox"]]
                face_crop = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                crop_bytes = cv2.imencode(".jpg", face_crop)[1].tobytes() if face_crop.size > 0 else b""
                emb_bytes = dlib_emb.astype(np.float32).tobytes() if hasattr(dlib_emb, "tobytes") else bytes(dlib_emb)
                save_face_capture(
                    emp_id=emp_id,
                    capture_angle="FRONT",
                    embedding_bytes=emb_bytes,
                    landmarks={},
                    vertices={},
                    geometry={},
                    pose={},
                    quality=metrics if isinstance(metrics, dict) else {"status": "ok"},
                    image_reference=None
                )
                face_enrolled = True

    # 2. PPE Detection & YOLO Annotation Generation
    yolo_model = model_manager.get_active_model()
    if yolo_model is None:
        model_manager.load_model("combined_sh17")
        yolo_model = model_manager.get_active_model()
    yolo_lines: List[str] = []

    if yolo_model is not None:
        results = yolo_model.predict(source=frame_bgr, conf=0.10, iou=0.45, imgsz=416, device=model_manager.device, verbose=False)[0]
        canonical_map = {
            "helmet": "helmet", "safety-vest": "safety_vest", "vest": "safety_vest",
            "gloves": "gloves", "shoes": "safety_shoes", "boots": "safety_shoes", "safety_shoes": "safety_shoes",
            "glasses": "goggles", "goggles": "goggles", "face-mask": "mask", "mask": "mask",
            "ear-mufs": "ear_protection", "ear_protection": "ear_protection", "person": "person"
        }
        model_names = getattr(yolo_model, "names", {})

        if results.boxes is not None:
            for b in results.boxes:
                cid = int(b.cls[0].item())
                raw_cname = model_names.get(cid, APPROVED_CLASSES[cid] if 0 <= cid < len(APPROVED_CLASSES) else f"class_{cid}")
                cname = canonical_map.get(str(raw_cname).lower(), str(raw_cname).lower())
                if cname in CLASS_MAP:
                    target_cid = CLASS_MAP[cname]
                    # Normalized YOLO format: class_id x_center y_center width height
                    xywhn = b.xywhn[0].tolist()
                    yolo_lines.append(f"{target_cid} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")
                    if cname not in classes_found:
                        classes_found.append(cname)

    # If user explicitly specified gear (e.g. helmet), and none was detected, assume upper half has helmet
    if assigned_classes:
        for req_gear in assigned_classes:
            if req_gear not in classes_found and req_gear in CLASS_MAP:
                target_cid = CLASS_MAP[req_gear]
                # Default box in upper region
                yolo_lines.append(f"{target_cid} 0.500000 0.250000 0.350000 0.300000")
                classes_found.append(req_gear)

    if not yolo_lines:
        # Default person box covering center if completely empty
        yolo_lines.append("0 0.500000 0.500000 0.800100 0.900000")
        classes_found.append("person")

    # 3. Save Image & Label to Training Dataset
    sample_uid = f"learned_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    img_filename = f"{sample_uid}.jpg"
    lbl_filename = f"{sample_uid}.txt"

    out_img_path = TRAIN_IMG_DIR / img_filename
    out_lbl_path = TRAIN_LBL_DIR / lbl_filename

    cv2.imwrite(str(out_img_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(out_lbl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines) + "\n")

    # Mirror every 4th sample to val set for balanced evaluation
    if int(sample_uid.split("_")[1]) % 4 == 0:
        shutil.copy2(str(out_img_path), str(VAL_IMG_DIR / img_filename))
        shutil.copy2(str(out_lbl_path), str(VAL_LBL_DIR / lbl_filename))

    return True, classes_found, face_enrolled


def ingest_media_for_learning(
    file_bytes: bytes,
    filename: str,
    media_type: str,
    camera_id: str = "CAM-01",
    employee_id: Optional[str] = None,
    employee_name: Optional[str] = None,
    assigned_classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Ingests photo or video for continuous self-learning.
    Extracts frames, performs biometric face learning and PPE annotation,
    and appends to training dataset.
    """
    sample_id = f"SMP-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    is_video = media_type.lower() == "video" or any(filename.lower().endswith(ext) for ext in [".mp4", ".webm", ".avi", ".mov", ".mkv"])
    
    samples_added = 0
    total_classes_learned: List[str] = []
    face_enrolled_total = False

    if not is_video:
        # Single image
        np_arr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            ok, classes, face_enr = _process_single_frame(frame, camera_id, employee_id, employee_name, assigned_classes)
            if ok:
                samples_added += 1
                total_classes_learned.extend(classes)
                if face_enr:
                    face_enrolled_total = True
    else:
        # Video: sample up to 10 keyframes across duration
        temp_video_path = BASE_DIR / "data" / f"temp_learn_{uuid.uuid4().hex}.mp4"
        temp_video_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_video_path, "wb") as f:
            f.write(file_bytes)

        try:
            cap = cv2.VideoCapture(str(temp_video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = max(1.0, cap.get(cv2.CAP_PROP_FPS))
            step = max(1, int(fps * 1.5)) # Every 1.5s
            
            frame_idx = 0
            while cap.isOpened() and samples_added < 12:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    ok, classes, face_enr = _process_single_frame(frame, camera_id, employee_id, employee_name, assigned_classes)
                    if ok:
                        samples_added += 1
                        for c in classes:
                            if c not in total_classes_learned:
                                total_classes_learned.append(c)
                        if face_enr:
                            face_enrolled_total = True
                frame_idx += 1
            cap.release()
        finally:
            if temp_video_path.exists():
                try:
                    temp_video_path.unlink()
                except Exception:
                    pass

    # Record in database
    unique_classes = list(set(total_classes_learned))
    record_learned_sample(
        sample_id=sample_id,
        camera_id=camera_id,
        media_type="video" if is_video else "image",
        filename=filename,
        samples_count=samples_added,
        classes_learned=unique_classes,
        employee_id=employee_id,
        employee_name=employee_name
    )

    return {
        "success": samples_added > 0,
        "sample_id": sample_id,
        "samples_added": samples_added,
        "media_type": "video" if is_video else "image",
        "face_enrolled": face_enrolled_total,
        "classes_learned": unique_classes,
        "total_learned_samples": len(list(TRAIN_IMG_DIR.glob("learned_*.jpg"))),
        "message": f"Successfully ingested {samples_added} new training sample(s). Face & PPE annotations generated."
    }


def _run_training_worker(initial_epochs: int, initial_imgsz: int):
    """Background worker executing Ultralytics YOLO fine-tuning on learned dataset with auto-queue chaining."""
    global _training_state, _training_queue
    from ultralytics import YOLO

    model_path = BASE_DIR / "Model" / "PPE_Combined_SH17_Best.pt"
    if not model_path.exists():
        with _training_lock:
            _training_state["status"] = "failed"
            _training_state["progress"] = 0
            _training_state["message"] = f"Base model not found at {model_path}"
        return

    curr_epochs = initial_epochs
    curr_imgsz = initial_imgsz

    while True:
        try:
            with _training_lock:
                _training_state["status"] = "training"
                _training_state["progress"] = 15
                _training_state["epoch"] = 0
                _training_state["total_epochs"] = curr_epochs
                q_len = len(_training_queue)
                queue_note = f" ({q_len} more job(s) queued)" if q_len > 0 else ""
                _training_state["message"] = f"Fine-tuning YOLO on {curr_epochs} epochs{queue_note}..."
                _training_state["queue_length"] = q_len
                _training_state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Load YOLO model
            yolo = YOLO(str(model_path))

            # Progress callback for epoch completions
            def on_train_epoch_end(trainer):
                ep = trainer.epoch + 1
                tot = trainer.epochs
                pct = int(15 + (ep / tot) * 70)
                with _training_lock:
                    _training_state["epoch"] = ep
                    _training_state["total_epochs"] = tot
                    _training_state["progress"] = min(88, pct)
                    _training_state["message"] = f"Epoch {ep}/{tot} completed. Fine-tuning neural weights..."

            try:
                yolo.add_callback("on_train_epoch_end", on_train_epoch_end)
            except Exception:
                pass

            # Guard: ensure training dataset contains samples
            train_images = list(TRAIN_IMG_DIR.glob("*.jpg")) + list(TRAIN_IMG_DIR.glob("*.png"))
            if not train_images:
                with _training_lock:
                    _training_state["status"] = "failed"
                    _training_state["progress"] = 0
                    _training_state["message"] = "No training images found in dataset. Ingest at least one image or video first."
                    _training_state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                break

            # GPU / CUDA acceleration with CPU fallback
            is_cuda = (torch is not None and torch.cuda.is_available() and model_manager.device.startswith("cuda"))
            train_device = "0" if is_cuda else "cpu"
            train_batch = 8 if is_cuda else 4
            train_workers = 2 if is_cuda else 0

            results = yolo.train(
                data=str(DATA_YAML_PATH),
                epochs=curr_epochs,
                imgsz=curr_imgsz,
                device=train_device,
                batch=train_batch,
                workers=train_workers,
                plots=False,
                val=False,
                verbose=False,
                project=str(BASE_DIR / "runs" / "learned"),
                name="tune",
                exist_ok=True
            )

            with _training_lock:
                _training_state["progress"] = 92
                _training_state["message"] = "Fine-tuning step complete. Deploying newly learned weights..."

            # Update active model in model_manager
            best_weights = BASE_DIR / "runs" / "learned" / "tune" / "weights" / "best.pt"
            last_weights = BASE_DIR / "runs" / "learned" / "tune" / "weights" / "last.pt"
            target_weights = best_weights if best_weights.exists() else last_weights

            if target_weights.exists():
                shutil.copy2(str(target_weights), str(model_path))
                # Reload active model so live camera feed immediately benefits
                model_manager.load_model("combined_sh17")

            # Check if additional training jobs were queued while this training was running
            with _training_lock:
                if _training_queue:
                    next_job = _training_queue.pop(0)
                    curr_epochs = next_job.get("epochs", 2)
                    curr_imgsz = next_job.get("imgsz", 416)
                    _training_state["message"] = f"Continuing with next queued training on new photos ({len(_training_queue)} pending)..."
                    _training_state["progress"] = 10
                    continue
                else:
                    _training_state["status"] = "completed"
                    _training_state["progress"] = 100
                    _training_state["epoch"] = curr_epochs
                    _training_state["queue_length"] = 0
                    _training_state["message"] = f"Successfully fine-tuned model with all learned media!"
                    _training_state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    break

        except Exception as e:
            with _training_lock:
                # If error, check if another job is waiting or report failure
                if _training_queue:
                    next_job = _training_queue.pop(0)
                    curr_epochs = next_job.get("epochs", 2)
                    curr_imgsz = next_job.get("imgsz", 416)
                    continue
                _training_state["status"] = "failed"
                _training_state["progress"] = 0
                _training_state["queue_length"] = 0
                _training_state["message"] = f"Training failed: {str(e)}"
                _training_state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                break


def start_model_training(epochs: int = 2, imgsz: int = 416) -> Dict[str, Any]:
    """Starts background fine-tuning or queues it if training is already running."""
    global _training_state, _training_queue
    with _training_lock:
        if _training_state["status"] == "training":
            _training_queue.append({"epochs": epochs, "imgsz": imgsz, "time": time.time()})
            q_len = len(_training_queue)
            return {
                "success": True,
                "queued": True,
                "queue_length": q_len,
                "message": f"Training pass already active in background. New media queued (Position #{q_len}) - will automatically train next!",
                "status": _training_state
            }

        _training_state["status"] = "training"
        _training_state["progress"] = 5
        _training_state["epoch"] = 0
        _training_state["total_epochs"] = epochs
        _training_state["queue_length"] = 0
        _training_state["message"] = "Starting background training worker..."
        _training_state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    worker = threading.Thread(target=_run_training_worker, args=(epochs, imgsz), daemon=True)
    worker.start()

    return {
        "success": True,
        "queued": False,
        "queue_length": 0,
        "message": f"Started continuous model learning on {epochs} epochs in background.",
        "status": _training_state
    }


def inspect_media_detections(
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
    media_type: str = "image"
) -> Dict[str, Any]:
    """
    Inspects image or video media and runs integrated detection to show everything the ML finds:
    - People & workers detected
    - Facial biometrics & recognized employee identity
    - PPE gear (helmet, safety vest, gloves, safety boots, goggles, mask, ear protection)
    - Returns annotated image with bounding boxes and structured detection breakdown.
    """
    import base64
    from backend.app.db.database import get_all_active_biometric_templates, get_all_employees

    MEDIA_DIR = BASE_DIR / "data" / "media"
    frame_bgr: Optional[np.ndarray] = None

    t0 = time.perf_counter()

    if file_bytes is not None and len(file_bytes) > 0:
        if media_type == "video":
            temp_vid = BASE_DIR / "data" / f"temp_inspect_{uuid.uuid4().hex[:6]}.mp4"
            with open(temp_vid, "wb") as f:
                f.write(file_bytes)
            cap = cv2.VideoCapture(str(temp_vid))
            ret, frame = cap.read()
            cap.release()
            try:
                temp_vid.unlink()
            except Exception:
                pass
            if ret and frame is not None:
                frame_bgr = frame
        else:
            arr = np.frombuffer(file_bytes, np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    elif filename:
        fpath = MEDIA_DIR / filename
        if fpath.exists():
            if media_type == "video" or fpath.suffix.lower() in [".mp4", ".webm", ".avi", ".mov", ".mkv"]:
                cap = cv2.VideoCapture(str(fpath))
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    frame_bgr = frame
            else:
                frame_bgr = cv2.imread(str(fpath))

    if frame_bgr is None or frame_bgr.size == 0:
        return {
            "success": False,
            "error": "No media available to inspect.",
            "preview_image": None,
            "detections": [],
            "detected_classes": [],
            "faces": [],
            "persons_count": 0,
            "recognized_employee": None,
            "summary": "No media provided. Upload a photo or video to inspect ML detections."
        }

    h, w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()

    # 1. Face Detection & Recognition
    detected_faces = []
    recognized_name = None
    if face_engine is not None:
        try:
            active_templates = get_all_active_biometric_templates()
            employees = {e["id"]: e for e in get_all_employees()}
            raw_faces = face_engine.detect_faces(frame_bgr, score_threshold=0.35)
            for f in raw_faces:
                fx1, fy1, fx2, fy2 = [int(v) for v in f["bbox"]]
                is_good, q_msg, _ = face_engine.assess_quality(frame_bgr, f)
                emp_name = None
                emp_id = None
                match_score = 0.0
                if is_good and active_templates:
                    emb = face_engine.extract_embedding(frame_bgr, f["raw_row"])
                    matched_id, score, _, _, _ = face_engine.match_face(emb, active_templates, threshold=0.55)
                    if matched_id:
                        emp_id = matched_id
                        match_score = score
                        emp_name = employees.get(matched_id, {}).get("full_name") or employees.get(matched_id, {}).get("name")
                        if not recognized_name and emp_name:
                            recognized_name = emp_name

                face_info = {
                    "bbox": [fx1, fy1, fx2, fy2],
                    "confidence": round(float(f["score"]), 3),
                    "confidence_pct": int(float(f["score"]) * 100),
                    "employee_id": emp_id,
                    "employee_name": emp_name or "Unregistered Person",
                    "quality": q_msg or ("Good" if is_good else "Low Quality"),
                    "is_recognized": bool(emp_name)
                }
                detected_faces.append(face_info)

                # Draw face box (cyan)
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 200, 0), 2)
                tag = f"{emp_name or 'Person'} ({int(float(f['score'])*100)}%)"
                cv2.putText(annotated, tag, (fx1, max(15, fy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)
        except Exception as e:
            print(f"[LearningService] Face inspection error: {e}")

    # 2. YOLO PPE Detection
    ppe_detections = []
    detected_classes = set()
    persons_count = 0

    class_colors = {
        "person": (180, 180, 180),
        "helmet": (34, 197, 94),        # green
        "safety_vest": (234, 88, 12),   # orange/blue
        "gloves": (245, 158, 11),       # amber
        "safety_shoes": (59, 130, 246), # blue
        "goggles": (168, 85, 247),     # purple
        "mask": (236, 72, 153),         # pink
        "ear_protection": (20, 184, 166)# teal
    }

    try:
        from ultralytics import YOLO
        model = model_manager.get_active_model()
        if model is None:
            model_path = BASE_DIR / "Model" / "PPE_Combined_SH17_Best.pt"
            if model_path.exists():
                model = YOLO(str(model_path))
            else:
                model_manager.load_model("combined_sh17")
                model = model_manager.get_active_model()

        if model is not None:
            results = model.predict(source=frame_bgr, conf=0.12, imgsz=416, device=model_manager.device, verbose=False)[0]
            model_names = getattr(model, "names", {})
            for b in results.boxes:
                cid = int(b.cls[0].item())
                raw_cname = model_names.get(cid, APPROVED_CLASSES[cid] if 0 <= cid < len(APPROVED_CLASSES) else f"class_{cid}")
                cname = str(raw_cname).lower()
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]

                if cname == "person":
                    persons_count += 1
                else:
                    detected_classes.add(cname)

                ppe_detections.append({
                    "class_name": cname,
                    "confidence": round(conf, 3),
                    "confidence_pct": int(conf * 100),
                    "box": [x1, y1, x2, y2]
                })

                color = class_colors.get(cname, (0, 255, 0))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label_text = f"{cname.replace('_', ' ').title()}: {int(conf * 100)}%"
                cv2.putText(annotated, label_text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    except Exception as e:
        print(f"[LearningService] YOLO inspection error: {e}")

    # Scale annotated preview to max width 640 to keep payload lightweight
    if w > 640:
        scale = 640 / w
        annotated = cv2.resize(annotated, (640, int(h * scale)))

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    preview_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

    # Summary string
    summary_parts = []
    if persons_count > 0:
        summary_parts.append(f"{persons_count} Person{'s' if persons_count > 1 else ''}")
    for c in sorted(detected_classes):
        c_conf = max([d["confidence_pct"] for d in ppe_detections if d["class_name"] == c], default=0)
        summary_parts.append(f"{c.replace('_', ' ').title()} ({c_conf}%)")

    summary_str = "ML Detected: " + (", ".join(summary_parts) if summary_parts else "No specific PPE detected")

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    return {
        "success": True,
        "preview_image": preview_b64,
        "detections": ppe_detections,
        "detected_classes": sorted(list(detected_classes)),
        "faces": detected_faces,
        "persons_count": max(persons_count, len(detected_faces)),
        "recognized_employee": recognized_name,
        "summary": summary_str,
        "inference_time_ms": elapsed_ms
    }
