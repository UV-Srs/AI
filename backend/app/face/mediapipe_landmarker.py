import threading
import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MEDIAPIPE_TASK_PATH = MODELS_DIR / "face_landmarker.task"

# Key landmark indices in the 478 MediaPipe face mesh
NOSE_TIP_IDX = 1
CHIN_IDX = 152
LEFT_EYE_OUTER_IDX = 33
RIGHT_EYE_OUTER_IDX = 263
LEFT_MOUTH_IDX = 61
RIGHT_MOUTH_IDX = 291
FOREHEAD_IDX = 10

class MediaPipeLandmarker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MediaPipeLandmarker, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        task_path = str(MEDIAPIPE_TASK_PATH)
        if not MEDIAPIPE_TASK_PATH.exists():
            raise FileNotFoundError(f"MediaPipe task model not found at {task_path}")

        print(f"[MediaPipeLandmarker] Initializing modern MediaPipe FaceLandmarker from {task_path}...", flush=True)
        base_options = mp_python.BaseOptions(model_asset_path=task_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=4
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._initialized = True
        print("[MediaPipeLandmarker] MediaPipe FaceLandmarker initialized successfully.", flush=True)

    def is_ready(self) -> bool:
        return getattr(self, "_initialized", False) and self.landmarker is not None

    def process_frame(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs MediaPipe Face Landmarker on BGR frame.
        Returns list of detected face mesh structures with 478 landmarks, contours, and estimated 3D head pose.
        """
        if not self.is_ready():
            return []

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        with self._lock:
            results = self.landmarker.detect(mp_image)

        if not results.face_landmarks:
            return []

        mesh_results = []
        for face_landmarks in results.face_landmarks:
            landmarks_list = []
            for lm in face_landmarks:
                landmarks_list.append({
                    "x": round(lm.x, 4),
                    "y": round(lm.y, 4),
                    "z": round(lm.z, 4),
                    "px": int(lm.x * w),
                    "py": int(lm.y * h)
                })

            # Calculate Bounding Box of the mesh
            xs = [lm["px"] for lm in landmarks_list]
            ys = [lm["py"] for lm in landmarks_list]
            bbox = [max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys))]

            # Estimate 3D head pose orientation (Yaw, Pitch, Roll)
            pose = self._estimate_head_pose(landmarks_list, w, h)

            # Extract anatomical contour groupings for UI rendering
            contours = self._extract_contours(landmarks_list)

            mesh_results.append({
                "bbox": bbox,
                "landmarks_count": len(landmarks_list),
                "landmarks": landmarks_list,
                "contours": contours,
                "pose": pose
            })

        return mesh_results

    def process_face_crop(self, frame_bgr: np.ndarray, bbox: List[int], pad_ratio: float = 0.20) -> Optional[Dict[str, Any]]:
        """
        Runs MediaPipe Face Landmarker on an expanded face crop and projects 
        the 478 3D landmarks and contours back to full frame coordinates.
        Ensures distant or small faces detected by YuNet get accurate mesh representation.
        """
        if not self.is_ready():
            return None

        h, w = frame_bgr.shape[:2]
        bx1, by1, bx2, by2 = bbox
        bw = bx2 - bx1
        bh = by2 - by1

        px = int(bw * pad_ratio)
        py = int(bh * pad_ratio)
        cx1 = max(0, bx1 - px)
        cy1 = max(0, by1 - py)
        cx2 = min(w, bx2 + px)
        cy2 = min(h, by2 + py)

        crop = frame_bgr[cy1:cy2, cx1:cx2]
        ch, cw = crop.shape[:2]
        if cw < 40 or ch < 40:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

        with self._lock:
            results = self.landmarker.detect(mp_image)

        if not results.face_landmarks:
            return None

        face_landmarks = results.face_landmarks[0]
        landmarks_list = []
        for lm in face_landmarks:
            # Crop pixel coords
            crop_px = lm.x * cw
            crop_py = lm.y * ch
            # Full frame pixel coords
            full_px = int(cx1 + crop_px)
            full_py = int(cy1 + crop_py)
            landmarks_list.append({
                "x": round(full_px / w, 4),
                "y": round(full_py / h, 4),
                "z": round(lm.z, 4),
                "px": full_px,
                "py": full_py
            })

        xs = [lm["px"] for lm in landmarks_list]
        ys = [lm["py"] for lm in landmarks_list]
        full_bbox = [max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys))]

        pose = self._estimate_head_pose(landmarks_list, w, h)
        contours = self._extract_contours(landmarks_list)

        return {
            "bbox": full_bbox,
            "landmarks_count": len(landmarks_list),
            "landmarks": landmarks_list,
            "contours": contours,
            "pose": pose
        }

    def _estimate_head_pose(self, landmarks: List[Dict[str, Any]], w: int, h: int) -> Dict[str, Any]:
        """
        Estimates yaw, pitch, roll angles using Perspective-n-Point (solvePnP)
        with canonical 3D facial model points.
        """
        if len(landmarks) < 468:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "angle": "FRONT"}

        # 3D Model generic reference coordinates (in mm)
        model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip (index 1)
            (0.0, -330.0, -65.0),     # Chin (index 152)
            (-225.0, 170.0, -135.0),  # Left eye outer corner (index 33)
            (225.0, 170.0, -135.0),   # Right eye outer corner (index 263)
            (-150.0, -150.0, -125.0), # Left mouth corner (index 61)
            (150.0, -150.0, -125.0),  # Right mouth corner (index 291)
        ], dtype=np.float64)

        image_points = np.array([
            (landmarks[NOSE_TIP_IDX]["px"], landmarks[NOSE_TIP_IDX]["py"]),
            (landmarks[CHIN_IDX]["px"], landmarks[CHIN_IDX]["py"]),
            (landmarks[LEFT_EYE_OUTER_IDX]["px"], landmarks[LEFT_EYE_OUTER_IDX]["py"]),
            (landmarks[RIGHT_EYE_OUTER_IDX]["px"], landmarks[RIGHT_EYE_OUTER_IDX]["py"]),
            (landmarks[LEFT_MOUTH_IDX]["px"], landmarks[LEFT_MOUTH_IDX]["py"]),
            (landmarks[RIGHT_MOUTH_IDX]["px"], landmarks[RIGHT_MOUTH_IDX]["py"]),
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "angle": "FRONT"}

        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(rmat[2, 1], rmat[2, 2])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = 0.0

        yaw_deg = round(math.degrees(yaw), 1)
        pitch_deg = round(math.degrees(pitch), 1)
        roll_deg = round(math.degrees(roll), 1)

        # Classify pose angle for registration verification
        if yaw_deg > 14.0:
            angle = "RIGHT"
        elif yaw_deg < -14.0:
            angle = "LEFT"
        elif pitch_deg > 15.0:
            angle = "TILT_DOWN"
        elif pitch_deg < -15.0:
            angle = "TILT_UP"
        else:
            angle = "FRONT"

        return {
            "yaw": yaw_deg,
            "pitch": pitch_deg,
            "roll": roll_deg,
            "angle": angle,
            "disclaimer": "3D head pose is mathematically estimated from 2D camera geometry and does not represent a physical 3D scan."
        }

    def _extract_contours(self, landmarks: List[Dict[str, Any]]) -> Dict[str, List[List[int]]]:
        """Groups key landmark coordinates into semantic contour paths for smooth UI overlay rendering."""
        oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
        left_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33]
        right_eye_indices = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249, 263]
        lips_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
        left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        right_eyebrow = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

        def get_pts(indices):
            return [[landmarks[i]["px"], landmarks[i]["py"]] for i in indices if i < len(landmarks)]

        return {
            "face_oval": get_pts(oval_indices),
            "left_eye": get_pts(left_eye_indices),
            "right_eye": get_pts(right_eye_indices),
            "lips": get_pts(lips_indices),
            "left_eyebrow": get_pts(left_eyebrow),
            "right_eyebrow": get_pts(right_eyebrow)
        }

mediapipe_landmarker = MediaPipeLandmarker()
