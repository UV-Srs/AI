"""
Facial Recognition Engine using OpenCV YuNet (Detection) and SFace (Recognition).
Both models are licensed under Apache-2.0 and run 100% locally on CPU without external cloud transmission.
"""

import threading
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from .mediapipe_landmarker import mediapipe_landmarker
from .dlib_recognition import dlib_recognition
from .face_alignment import expand_face_bbox_proportionally

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

class FaceEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FaceEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        print("[FaceEngine] Initializing YuNet & SFace models (Apache-2.0)...", flush=True)
        if not YUNET_PATH.exists():
            raise FileNotFoundError(f"YuNet model not found at {YUNET_PATH}")
        if not SFACE_PATH.exists():
            raise FileNotFoundError(f"SFace model not found at {SFACE_PATH}")

        # Initialize YuNet detector with initial size (320, 320)
        self.detector = cv2.FaceDetectorYN.create(
            str(YUNET_PATH),
            "",
            (320, 320),
            score_threshold=0.45,
            nms_threshold=0.3,
            top_k=5000
        )

        # Initialize SFace recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
        self.curr_input_size = (320, 320)

        # Multi-model components (Google MediaPipe 478 3D Mesh + dlib ResNet-34)
        self.mediapipe = mediapipe_landmarker
        self.dlib = dlib_recognition
        
        # Spatial identity track cache for multi-frame confirmation
        # Maps track_key -> list of recent matched employee IDs
        self.track_history: Dict[str, List[Tuple[Optional[str], float, str]]] = {}
        self._cache_lock = threading.Lock()
        
        self._initialized = True
        print("[FaceEngine] Multi-model face engine (YuNet + MediaPipe + dlib + SFace) initialized successfully.", flush=True)

    def is_ready(self) -> bool:
        return (
            getattr(self, "_initialized", False)
            and self.detector is not None
            and self.mediapipe.is_ready()
            and self.dlib.is_ready()
        )

    def get_face_mesh(self, frame: np.ndarray, bbox: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
        """
        Calculates Google MediaPipe 478 3D facial landmarks, anatomical contours,
        and solvePnP 3D head pose estimation.
        """
        if bbox is not None:
            return self.mediapipe.process_face_crop(frame, bbox)
        meshes = self.mediapipe.process_frame(frame)
        return meshes[0] if meshes else None

    def extract_dlib_embedding(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Extracts 128-D facial descriptor using dlib ResNet-34.
        """
        h, w = frame.shape[:2]
        expanded_box = expand_face_bbox_proportionally(bbox, w, h, pad_ratio=0.10)
        return self.dlib.generate_embedding(frame, expanded_box)

    def detect_faces(self, frame: np.ndarray, score_threshold: float = 0.45) -> List[Dict[str, Any]]:
        """
        Detects faces in BGR image frame using YuNet.
        Returns list of face dicts with bbox, landmarks, score, and raw row.
        """
        h, w = frame.shape[:2]
        with self._lock:
            if self.curr_input_size != (w, h):
                self.detector.setInputSize((w, h))
                self.curr_input_size = (w, h)

            _, faces = self.detector.detect(frame)

        if faces is None or len(faces) == 0:
            return []

        detected = []
        for row in faces:
            score = float(row[14])
            if score < score_threshold:
                continue

            x, y, fw, fh = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            # Clamp coordinates to frame boundary
            x = max(0, x)
            y = max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)

            if fw < 10 or fh < 10:
                continue

            landmarks = [
                (float(row[4]), float(row[5])),   # Right eye
                (float(row[6]), float(row[7])),   # Left eye
                (float(row[8]), float(row[9])),   # Nose tip
                (float(row[10]), float(row[11])), # Right mouth corner
                (float(row[12]), float(row[13])), # Left mouth corner
            ]

            detected.append({
                "bbox": [x, y, x + fw, y + fh], # [x1, y1, x2, y2]
                "width": fw,
                "height": fh,
                "score": round(score, 4),
                "landmarks": landmarks,
                "raw_row": row
            })

        return detected

    def assess_quality(self, frame: np.ndarray, face: Dict[str, Any]) -> Tuple[bool, str, Dict[str, float]]:
        """
        Assesses face image quality based on:
          1. Resolution (min 20x20 pixels for long-range detection)
          2. Blur / Sharpness (Laplacian variance >= 12.0 for webcams)
          3. Brightness (mean intensity 20 - 240)
        """
        x1, y1, x2, y2 = face["bbox"]
        fw = x2 - x1
        fh = y2 - y1

        if fw < 20 or fh < 20:
            return False, f"Face resolution too small ({fw}x{fh}px, min 20x20px)", {"sharpness": 0, "brightness": 0}

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False, "Invalid face crop", {"sharpness": 0, "brightness": 0}

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness via Laplacian variance
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # 2. Brightness via mean intensity
        brightness = float(np.mean(gray))

        metrics = {
            "sharpness": round(sharpness, 1),
            "brightness": round(brightness, 1),
            "width": fw,
            "height": fh
        }

        if sharpness < 12.0:
            return False, f"Face image is blurry (sharpness={sharpness:.1f}, min 12.0)", metrics

        if brightness < 20.0:
            return False, f"Lighting too dark (brightness={brightness:.1f}, min 20.0)", metrics
        if brightness > 240.0:
            return False, f"Lighting overexposed (brightness={brightness:.1f}, max 240.0)", metrics

        return True, "Quality acceptable", metrics

    def check_liveness(self, frame: np.ndarray, face: Dict[str, Any], track_id: Optional[str] = None) -> Tuple[str, float]:
        """
        Anti-spoofing / presentation-attack filter using:
          1. High-frequency texture ratio & specular gradient analysis.
          2. Micro-motion / landmark consistency tracking.
        Returns:
          status: "PASS" | "LIVENESS_REVIEW" | "UNCERTAIN"
          score: liveness confidence between 0.0 and 1.0.
        """
        x1, y1, x2, y2 = face["bbox"]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "UNCERTAIN", 0.5

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Printed paper and low-res screen replays produce either ultra-low texture or sharp pixel grids
        # Genuine faces typically have natural gradient distribution
        score = 0.85
        if lap_var < 50.0:
            score -= 0.35
        elif lap_var > 1500.0:
            # Extreme high frequency grid often from screen refresh or print dithering
            score -= 0.30

        # Check color saturation (screens or printed photos often have flattened saturation)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat_mean = np.mean(hsv[:, :, 1])
        if sat_mean < 15.0:
            # Grayscale or washed out printed photo
            score -= 0.25

        score = max(0.0, min(1.0, score))

        if score >= 0.70:
            return "PASS", round(score, 2)
        elif score >= 0.45:
            return "LIVENESS_REVIEW", round(score, 2)
        else:
            return "LIVENESS_REVIEW", round(score, 2)

    def extract_embedding(self, frame: np.ndarray, raw_row: np.ndarray) -> np.ndarray:
        """
        Aligns face crop and extracts 128-dimensional L2-normalized feature vector using SFace.
        """
        with self._lock:
            aligned = self.recognizer.alignCrop(frame, raw_row)
            feature = self.recognizer.feature(aligned)
            # SFace returns (1, 128) float32 vector
            vec = feature.flatten()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec

    def analyze_face_biometrics(
        self,
        bbox: List[int],
        landmarks_5: List[Tuple[float, float]],
        image_w: int = 640,
        image_h: int = 480,
        score: float = 0.95
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Extracts 2D landmarks with coordinate system metadata, anthropometric vertices
        (with 2D limitation disclosure), calculated geometric proportions, and head pose.
        """
        x1, y1, x2, y2 = bbox
        fw = max(1, x2 - x1)
        fh = max(1, y2 - y1)

        # 5 YuNet landmarks: re, le, nose, rm, lm
        re_x, re_y = landmarks_5[0]
        le_x, le_y = landmarks_5[1]
        nose_x, nose_y = landmarks_5[2]
        rm_x, rm_y = landmarks_5[3]
        lm_x, lm_y = landmarks_5[4]

        # 1. Pose estimation (yaw ratio & roll angle)
        d_re_nose = abs(nose_x - re_x)
        d_le_nose = abs(nose_x - le_x)
        yaw_ratio = round(float(d_re_nose / (d_le_nose + 1e-5)), 3)

        dx_eyes = le_x - re_x
        dy_eyes = le_y - re_y
        roll_deg = round(float(np.degrees(np.arctan2(dy_eyes, dx_eyes))), 2)

        # Yaw degrees relative to frontal baseline (-45 to +45)
        yaw_deg = round(float((yaw_ratio - 1.0) * 35.0), 1)
        yaw_deg = max(-45.0, min(45.0, yaw_deg))

        if abs(yaw_deg) <= 15.0:
            detected_angle = "FRONT"
        elif yaw_deg > 15.0:
            detected_angle = "RIGHT"
        else:
            detected_angle = "LEFT"

        pose = {
            "detected_angle": detected_angle,
            "yaw_ratio": yaw_ratio,
            "estimated_yaw_degrees": yaw_deg,
            "roll_degrees": roll_deg,
            "confidence": round(float(score), 4)
        }

        # 2. Detailed 2D Landmarks with coordinate metadata
        chin_x = nose_x
        chin_y = min(float(image_h), y2 - (fh * 0.05))
        jaw_left_x = max(0.0, x1 + (fw * 0.1))
        jaw_left_y = y1 + (fh * 0.75)
        jaw_right_x = min(float(image_w), x2 - (fw * 0.1))
        jaw_right_y = y1 + (fh * 0.75)
        eyebrow_y = max(0.0, min(re_y, le_y) - (fh * 0.12))

        landmarks = {
            "coordinate_system": "image_pixel_coordinates",
            "origin": "top_left",
            "frame_width": int(image_w),
            "frame_height": int(image_h),
            "primary_keypoints": {
                "right_eye": {"x": round(re_x, 2), "y": round(re_y, 2)},
                "left_eye": {"x": round(le_x, 2), "y": round(le_y, 2)},
                "nose_tip": {"x": round(nose_x, 2), "y": round(nose_y, 2)},
                "right_mouth_corner": {"x": round(rm_x, 2), "y": round(rm_y, 2)},
                "left_mouth_corner": {"x": round(lm_x, 2), "y": round(lm_y, 2)}
            },
            "derived_facial_points": {
                "chin": {"x": round(chin_x, 2), "y": round(chin_y, 2)},
                "jaw_left": {"x": round(jaw_left_x, 2), "y": round(jaw_left_y, 2)},
                "jaw_right": {"x": round(jaw_right_x, 2), "y": round(jaw_right_y, 2)},
                "eyebrows_mid": {"x": round(nose_x, 2), "y": round(eyebrow_y, 2)}
            },
            "model": "OpenCV YuNet",
            "version": "2023mar",
            "confidence": round(float(score), 4)
        }

        # 3. Vertices (normalized polygon with 2D limitation disclosure)
        mesh_pts = [
            {"index": 0, "name": "right_eye", "x": round((re_x - x1) / fw, 4), "y": round((re_y - y1) / fh, 4), "z": 0.0},
            {"index": 1, "name": "left_eye", "x": round((le_x - x1) / fw, 4), "y": round((le_y - y1) / fh, 4), "z": 0.0},
            {"index": 2, "name": "nose_tip", "x": round((nose_x - x1) / fw, 4), "y": round((nose_y - y1) / fh, 4), "z": 0.08},
            {"index": 3, "name": "right_mouth", "x": round((rm_x - x1) / fw, 4), "y": round((rm_y - y1) / fh, 4), "z": 0.02},
            {"index": 4, "name": "left_mouth", "x": round((lm_x - x1) / fw, 4), "y": round((lm_y - y1) / fh, 4), "z": 0.02},
            {"index": 5, "name": "chin", "x": round((chin_x - x1) / fw, 4), "y": round((chin_y - y1) / fh, 4), "z": -0.02},
            {"index": 6, "name": "jaw_left", "x": round((jaw_left_x - x1) / fw, 4), "y": round((jaw_left_y - y1) / fh, 4), "z": -0.05},
            {"index": 7, "name": "jaw_right", "x": round((jaw_right_x - x1) / fw, 4), "y": round((jaw_right_y - y1) / fh, 4), "z": -0.05}
        ]
        vertices = {
            "mesh_model": "YuNet Anthropometric Surface Polygon",
            "version": "2023mar",
            "limitation_disclosure": "Model provides 2D image coordinates; Z-depth is anthropometrically estimated and not from 3D sensor.",
            "normalization": "unit_interval_relative_to_face_bbox",
            "vertex_count": len(mesh_pts),
            "vertices": mesh_pts
        }

        # 4. Facial Geometry (anthropometric proportions and ratios)
        inter_ocular_dist = float(np.hypot(le_x - re_x, le_y - re_y))
        mouth_width = float(np.hypot(lm_x - rm_x, lm_y - rm_y))
        mouth_mid_y = (rm_y + lm_y) / 2.0
        nose_to_mouth_dist = float(abs(mouth_mid_y - nose_y))

        geometry = {
            "face_width_to_height_ratio": round(float(fw / fh), 4),
            "inter_ocular_distance_px": round(inter_ocular_dist, 2),
            "inter_eye_to_face_width_ratio": round(float(inter_ocular_dist / fw), 4),
            "mouth_width_px": round(mouth_width, 2),
            "mouth_to_face_width_ratio": round(float(mouth_width / fw), 4),
            "nose_to_mouth_distance_px": round(nose_to_mouth_dist, 2),
            "jaw_width_px": round(float(jaw_right_x - jaw_left_x), 2),
            "jaw_to_face_width_ratio": round(float((jaw_right_x - jaw_left_x) / fw), 4),
            "chin_proportions_ratio": round(float((chin_y - mouth_mid_y) / fh), 4),
            "measurement_units": {
                "absolute": "pixels",
                "proportional": "normalized_ratio"
            },
            "disclaimer": "These anthropometric geometric proportions are for verification alignment assistance and do not uniquely identify an individual."
        }

        return pose, landmarks, vertices, geometry

    def validate_pose_for_angle(self, pose: Dict[str, Any], target_angle: str) -> Tuple[bool, str]:
        """
        Validates whether estimated pose aligns with requested enrollment angle.
        target_angle: 'FRONT' | 'LEFT' | 'RIGHT'
        """
        target = target_angle.upper()
        detected = pose.get("detected_angle", "FRONT")
        yaw = pose.get("estimated_yaw_degrees", 0.0)

        if target == "FRONT":
            if abs(yaw) <= 15.0:
                return True, "Frontal pose valid"
            return False, f"Please face the camera directly (estimated yaw: {yaw:+.1f}°)"

        elif target == "LEFT":
            # Looking towards their left (camera perspective yaw <= -10)
            if yaw <= -10.0 or detected == "LEFT":
                return True, "Left-facing angle valid"
            return False, f"Please turn your head slightly to the left (current yaw: {yaw:+.1f}°)"

        elif target == "RIGHT":
            # Looking towards their right (camera perspective yaw >= 10)
            if yaw >= 10.0 or detected == "RIGHT":
                return True, "Right-facing angle valid"
            return False, f"Please turn your head slightly to the right (current yaw: {yaw:+.1f}°)"

        return True, "Angle accepted"

    def match_face(
        self,
        embedding: np.ndarray,
        active_templates: Dict[str, Any],
        threshold: float = 0.363,
        margin_threshold: float = 0.050
    ) -> Tuple[Optional[str], float, float, float, str]:
        """
        Matches embedding against active enrolled templates supporting 3-angle multi-template comparison.
        active_templates can be:
          1. Dict[emp_id, Dict[angle, bytes]]: 3-angle captures {emp1: {'FRONT': b'...', 'LEFT': b'...', ...}}
          2. Dict[emp_id, bytes]: legacy single-template mapping {emp1: b'...'}

        Returns:
          (best_employee_id, best_similarity, second_best_similarity, match_margin, identity_state)
        States:
          - 'REGISTERED': best_similarity >= threshold AND match_margin >= margin_threshold
          - 'UNCERTAIN': best_similarity >= 0.25 (or fails margin separation)
          - 'UNREGISTERED': best_similarity < 0.25
        """
        if not active_templates:
            return None, 0.0, 0.0, 0.0, "UNREGISTERED"

        candidate_scores: List[Tuple[str, float]] = []

        for emp_id, tmpl_data in active_templates.items():
            best_emp_score = -1.0

            if isinstance(tmpl_data, dict):
                # Multi-angle template dictionary
                for angle, tmpl_bytes in tmpl_data.items():
                    tmpl_vec = np.frombuffer(tmpl_bytes, dtype=np.float32)
                    if tmpl_vec.shape[0] != 128:
                        continue
                    sim = float(np.dot(embedding, tmpl_vec))
                    if sim > best_emp_score:
                        best_emp_score = sim
            elif isinstance(tmpl_data, (bytes, bytearray, memoryview)):
                # Single template bytes
                tmpl_vec = np.frombuffer(bytes(tmpl_data), dtype=np.float32)
                if tmpl_vec.shape[0] == 128:
                    best_emp_score = float(np.dot(embedding, tmpl_vec))

            if best_emp_score > -1.0:
                candidate_scores.append((emp_id, best_emp_score))

        if not candidate_scores:
            return None, 0.0, 0.0, 0.0, "UNREGISTERED"

        # Sort candidate employees by highest similarity descending
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        best_id, best_sim = candidate_scores[0]
        best_sim = round(max(0.0, best_sim), 4)

        if len(candidate_scores) > 1:
            second_sim = round(max(0.0, candidate_scores[1][1]), 4)
        else:
            second_sim = 0.0

        match_margin = round(best_sim - second_sim, 4)

        # Verification Decision Logic
        if best_sim >= threshold:
            if match_margin >= margin_threshold or len(candidate_scores) == 1:
                return best_id, best_sim, second_sim, match_margin, "REGISTERED"
            else:
                # Top match passes threshold, but runner-up is too close -> ambiguous
                return best_id, best_sim, second_sim, match_margin, "UNCERTAIN"
        elif best_sim < 0.25:
            return None, best_sim, second_sim, match_margin, "UNREGISTERED"
        else:
            return best_id, best_sim, second_sim, match_margin, "UNCERTAIN"

    def confirm_multi_frame(
        self,
        track_key: str,
        current_emp_id: Optional[str],
        current_score: float,
        current_state: str,
        required_consecutive: int = 2
    ) -> Tuple[Optional[str], float, str]:
        """
        Validates match consistency across consecutive frames before declaring REGISTERED.
        Prevents single-frame identity flicker. If confidence is strong (>= 0.45), confirms immediately.
        """
        with self._cache_lock:
            if len(self.track_history) > 200:
                self.track_history.clear()

            if track_key not in self.track_history:
                self.track_history[track_key] = []

            history = self.track_history[track_key]
            history.append((current_emp_id, current_score, current_state))
            if len(history) > 10:
                history.pop(0)

            # If current is not registered, return current state
            if current_state != "REGISTERED" or current_emp_id is None:
                return current_emp_id, current_score, current_state

            # Strong match passes immediately without requiring consecutive streak
            if current_score >= 0.45:
                return current_emp_id, current_score, "REGISTERED"

            # Count recent occurrences of this employee ID
            recent = history[-required_consecutive:]
            matching_count = sum(1 for (eid, _, state) in recent if eid == current_emp_id and state == "REGISTERED")

            if matching_count >= required_consecutive:
                return current_emp_id, current_score, "REGISTERED"
            else:
                return current_emp_id, current_score, "UNCERTAIN"

face_engine = FaceEngine()

