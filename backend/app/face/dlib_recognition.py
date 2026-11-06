import threading
import numpy as np
import cv2
import dlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DLIB_SHAPE_PREDICTOR_PATH = MODELS_DIR / "shape_predictor_5_face_landmarks.dat"
DLIB_RESNET_PATH = MODELS_DIR / "dlib_face_recognition_resnet_model_v1.dat"

class DlibRecognitionEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DlibRecognitionEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        sp_path = str(DLIB_SHAPE_PREDICTOR_PATH)
        resnet_path = str(DLIB_RESNET_PATH)

        if not DLIB_SHAPE_PREDICTOR_PATH.exists():
            raise FileNotFoundError(f"dlib shape predictor not found at {sp_path}")
        if not DLIB_RESNET_PATH.exists():
            raise FileNotFoundError(f"dlib ResNet recognition model not found at {resnet_path}")

        print(f"[DlibRecognition] Loading dlib shape predictor from {sp_path}...", flush=True)
        self.shape_predictor = dlib.shape_predictor(sp_path)

        print(f"[DlibRecognition] Loading dlib ResNet recognition model from {resnet_path}...", flush=True)
        self.recognition_model = dlib.face_recognition_model_v1(resnet_path)

        self._initialized = True
        print("[DlibRecognition] dlib recognition engine initialized successfully.", flush=True)

    def is_ready(self) -> bool:
        return getattr(self, "_initialized", False) and self.shape_predictor is not None and self.recognition_model is not None

    def generate_embedding(self, frame_bgr: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Extracts 128-dimensional L2-normalized biometric embedding vector using dlib ResNet.
        frame_bgr: BGR frame as numpy array.
        bbox: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame_bgr.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        bw = x2 - x1
        bh = y2 - y1

        if bw < 10 or bh < 10:
            return np.zeros(128, dtype=np.float64)

        # Crop face with padding to avoid allocating or converting massive full frames
        pad_x = int(bw * 0.20)
        pad_y = int(bh * 0.20)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return np.zeros(128, dtype=np.float64)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rect = dlib.rectangle(x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)

        with self._lock:
            shape = self.shape_predictor(crop_rgb, rect)
            descriptor = self.recognition_model.compute_face_descriptor(crop_rgb, shape, num_jitters=1)

        embedding = np.array(descriptor, dtype=np.float64)
        # Normalize vector to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def calculate_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculates Euclidean distance between two 128D embeddings."""
        return float(np.linalg.norm(emb1 - emb2))

    def match_face(
        self,
        live_embedding: np.ndarray,
        active_templates: Dict[str, Dict[str, np.ndarray]],
        threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None
    ) -> Tuple[Optional[str], float, Optional[str], float, float, str]:
        """
        One-to-many biometric comparison against enrolled templates.
        Compares live embedding against all enrolled angles (FRONT, LEFT, RIGHT).
        
        Returns:
          (best_emp_id, best_distance, second_best_id, second_best_distance, match_margin, identity_state)
        
        Statuses:
          - 'REGISTERED': Closest distance <= threshold AND separation margin >= margin_threshold
          - 'UNCERTAIN': Closest distance <= 0.65 or separation margin too small (ambiguous)
          - 'UNREGISTERED': Closest distance > 0.65
        """
        match_thresh = threshold if threshold is not None else 0.58
        margin_thresh = margin_threshold if margin_threshold is not None else 0.08

        if not active_templates:
            return None, 999.0, None, 999.0, 0.0, "UNREGISTERED"

        candidate_scores: List[Tuple[str, float]] = []

        for emp_id, angles_dict in active_templates.items():
            min_dist_for_emp = float("inf")
            for angle, tmpl in angles_dict.items():
                d = self.calculate_distance(live_embedding, tmpl)
                if d < min_dist_for_emp:
                    min_dist_for_emp = d
            candidate_scores.append((emp_id, min_dist_for_emp))

        # Sort candidates ascending by Euclidean distance (0.0 is exact identical face)
        candidate_scores.sort(key=lambda x: x[1])

        best_id, best_dist = candidate_scores[0]
        second_id = candidate_scores[1][0] if len(candidate_scores) > 1 else None
        second_dist = candidate_scores[1][1] if len(candidate_scores) > 1 else 999.0
        match_margin = second_dist - best_dist

        # Rigorous ambiguity rejection
        if best_dist <= match_thresh:
            if len(candidate_scores) == 1 or match_margin >= margin_thresh:
                return best_id, round(best_dist, 4), second_id, round(second_dist, 4), round(match_margin, 4), "REGISTERED"
            else:
                # Ambiguous: Both candidate 1 and candidate 2 have similar low distances
                return best_id, round(best_dist, 4), second_id, round(second_dist, 4), round(match_margin, 4), "UNCERTAIN"
        elif best_dist <= 0.65:
            return best_id, round(best_dist, 4), second_id, round(second_dist, 4), round(match_margin, 4), "UNCERTAIN"
        else:
            return None, round(best_dist, 4), second_id, round(second_dist, 4), round(match_margin, 4), "UNREGISTERED"

dlib_recognition = DlibRecognitionEngine()

