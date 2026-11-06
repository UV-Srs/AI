import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "Model"

APPROVED_CLASSES = [
    "person",
    "helmet",
    "safety_vest",
    "gloves",
    "safety_shoes",
    "goggles",
    "mask",
    "ear_protection"
]

APPROVED_MODELS = {
    "combined_sh17": {
        "id": "combined_sh17",
        "name": "YOLO26n Combined SH17 (8 Classes)",
        "filename": "PPE_Combined_SH17_Best.pt",
        "rel_path": "Model/PPE_Combined_SH17_Best.pt"
    }
}

class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.active_model_id: Optional[str] = None
        self.active_model: Optional[YOLO] = None
        self.active_model_path: Optional[Path] = None
        self.model_lock = threading.Lock()
        
        # Automatically load the first available approved model
        self._auto_load_default()
        self._initialized = True

    def _auto_load_default(self):
        """Automatically load the best available approved checkpoint."""
        for mid in APPROVED_MODELS:
            info = APPROVED_MODELS[mid]
            path = PROJECT_ROOT / info["rel_path"]
            if path.exists() and path.stat().st_size > 1000000:
                try:
                    self.load_model(mid)
                    return
                except Exception as e:
                    print(f"Warning: Failed to auto-load {mid}: {e}")

    def get_available_models(self) -> List[Dict]:
        """Return metadata for all approved model checkpoints."""
        models_info = []
        for mid, meta in APPROVED_MODELS.items():
            path = PROJECT_ROOT / meta["rel_path"]
            exists = path.exists() and path.is_file()
            size_mb = round(path.stat().st_size / 1e6, 2) if exists else 0.0
            
            is_active = (mid == self.active_model_id)
            
            models_info.append({
                "id": mid,
                "name": meta["name"],
                "filename": meta["filename"],
                "path": str(path.relative_to(PROJECT_ROOT) if exists else meta["rel_path"]),
                "size_mb": size_mb,
                "classes_count": len(APPROVED_CLASSES),
                "classes": APPROVED_CLASSES,
                "is_active": is_active,
                "exists": exists
            })
        return models_info

    def load_model(self, model_id: str) -> Tuple[bool, str]:
        """Switch or load the specified approved model checkpoint."""
        if model_id not in APPROVED_MODELS:
            return False, f"Model ID '{model_id}' is not in the approved model registry."

        meta = APPROVED_MODELS[model_id]
        checkpoint_path = PROJECT_ROOT / meta["rel_path"]

        if not checkpoint_path.exists():
            return False, f"Checkpoint file does not exist at {checkpoint_path.name}."

        with self.model_lock:
            try:
                print(f"Loading YOLO26n checkpoint: {checkpoint_path} on {self.device}...")
                loaded = YOLO(str(checkpoint_path))
                
                # Validate classes count
                if len(loaded.names) < 6:
                    print(f"Warning: Model classes count ({len(loaded.names)}) is less than minimum required (6).")

                # Store newly active model
                self.active_model = loaded
                self.active_model_id = model_id
                self.active_model_path = checkpoint_path
                # Warmup inference to pre-allocate PyTorch CPU memory and prevent first-frame lag
                try:
                    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
                    self.active_model.predict(dummy_frame, imgsz=384, verbose=False)
                except Exception:
                    pass
                print(f"Successfully loaded {meta['name']} ({checkpoint_path.name})")
                return True, f"Successfully loaded {meta['name']}."
            except Exception as e:
                return False, f"Error loading checkpoint: {str(e)}"

    def get_active_model(self) -> Optional[YOLO]:
        """Return the currently loaded YOLO model instance."""
        return self.active_model

    def get_status(self) -> Dict:
        """Return health and status dictionary."""
        return {
            "device": self.device,
            "active_model_id": self.active_model_id,
            "active_model_name": APPROVED_MODELS[self.active_model_id]["name"] if self.active_model_id else None,
            "model_loaded": self.active_model is not None,
            "classes": APPROVED_CLASSES
        }

model_manager = ModelManager()

