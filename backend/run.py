"""
Backend server runner for Live PPE Detection System.
Launches FastAPI with Uvicorn on http://127.0.0.1:8001.
"""

import sys
import os
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("  UNIFIED PPE & 3D FACE RECOGNITION BACKEND (FastAPI + YOLO26n + YuNet + MediaPipe + dlib)")
    print("  Host: http://127.0.0.1:8001")
    print("  API Docs: http://127.0.0.1:8001/docs")
    print("  Live Stream: ws://127.0.0.1:8001/ws/detection")
    print("=" * 70)
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20
    )

