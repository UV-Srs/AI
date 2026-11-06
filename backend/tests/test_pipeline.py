"""
Unit Test: Integrated Inference Pipeline
Verifies simultaneous execution of:
1. YOLO26n PPE Detection
2. OpenCV YuNet Face Detection
3. Google MediaPipe 478 3D Mesh & Head Pose
4. dlib 128-D Descriptors
"""

import sys
import numpy as np
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.inference.pipeline import run_integrated_pipeline

def test_pipeline_on_synthetic_frame():
    # Construct a synthetic 640x480 test image with a face oval
    img = np.full((480, 640, 3), 120, dtype=np.uint8)
    cv2.ellipse(img, (320, 240), (80, 110), 0, 0, 360, (200, 180, 160), -1)
    
    _, enc = cv2.imencode(".jpg", img)
    img_bytes = enc.tobytes()

    res = run_integrated_pipeline(img_bytes, ppe_conf_threshold=0.15)
    assert res.get("success") is True, f"Pipeline failed: {res.get('error')}"
    assert "inference_time_ms" in res
    assert "fps" in res
    print(f"[PASS] Pipeline executed successfully (time: {res.get('inference_time_ms')} ms, fps: {res.get('fps')})")

if __name__ == "__main__":
    print("--- Running test_pipeline.py ---")
    test_pipeline_on_synthetic_frame()
    print("--- ALL PIPELINE TESTS PASSED ---")

