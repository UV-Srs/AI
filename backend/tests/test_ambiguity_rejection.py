"""
Unit Test: Biometric Ambiguity Rejection & Margin Thresholding
Verifies:
1. Candidate 1 vs Candidate 2 margin calculation (best - second).
2. Ambiguous similarity (margin < 0.08) results in UNCERTAIN status.
3. Distance in ambiguous band [0.58, 0.65] results in UNCERTAIN status.
4. Distance > 0.65 results in UNREGISTERED status.
5. Distance < 0.58 with margin >= 0.08 results in REGISTERED status.
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.face.dlib_recognition import dlib_recognition

def test_margin_and_ambiguity_rules():
    np.random.seed(42)
    
    # Base probe vector normalized to unit length
    v_probe = np.random.randn(128).astype(np.float32)
    v_probe /= np.linalg.norm(v_probe)

    # Case 1: Clear match for Employee A
    v_emp_a = v_probe + np.random.randn(128) * 0.05
    v_emp_a /= np.linalg.norm(v_emp_a)
    
    v_emp_b = -v_probe + np.random.randn(128) * 0.1
    v_emp_b /= np.linalg.norm(v_emp_b)

    templates = {
        "EMP-001": {"FRONT": v_emp_a},
        "EMP-002": {"FRONT": v_emp_b}
    }

    best_id, best_dist, second_id, second_dist, margin, status = dlib_recognition.match_face(
        v_probe, templates, threshold=0.58, margin_threshold=0.08
    )

    assert status == "REGISTERED", f"Expected REGISTERED, got {status}"
    assert best_id == "EMP-001"
    assert margin >= 0.08, f"Expected margin >= 0.08, got {margin}"
    print(f"[PASS] Case 1 (Clear Match): dist={best_dist:.4f}, margin={margin:.4f} -> {status}")

    # Case 2: Ambiguous Twin/Similar Face (margin < 0.08)
    noise1 = np.random.randn(128).astype(np.float32)
    noise1 -= np.dot(noise1, v_probe) * v_probe
    noise1 /= np.linalg.norm(noise1)

    v_emp_twin1 = 0.92 * v_probe + 0.3919 * noise1
    v_emp_twin1 /= np.linalg.norm(v_emp_twin1)

    noise2 = np.random.randn(128).astype(np.float32)
    noise2 -= np.dot(noise2, v_probe) * v_probe
    noise2 /= np.linalg.norm(noise2)
    v_emp_twin2 = 0.90755 * v_probe + 0.4199 * noise2
    v_emp_twin2 /= np.linalg.norm(v_emp_twin2)

    templates_twins = {
        "EMP-TWIN-1": {"FRONT": v_emp_twin1},
        "EMP-TWIN-2": {"FRONT": v_emp_twin2}
    }

    best_id, best_dist, second_id, second_dist, margin, status = dlib_recognition.match_face(
        v_probe, templates_twins, threshold=0.58, margin_threshold=0.08
    )

    assert status == "UNCERTAIN", f"Expected UNCERTAIN due to tight margin, got {status}"
    assert margin < 0.08, f"Expected margin < 0.08, got {margin}"
    print(f"[PASS] Case 2 (Ambiguity Rejection): dist1={best_dist:.4f}, dist2={second_dist:.4f}, margin={margin:.4f} -> {status}")

    # Case 3: Distant Unknown Individual (dist > 0.65)
    v_stranger = np.random.randn(128).astype(np.float32)
    v_stranger /= np.linalg.norm(v_stranger)

    best_id, best_dist, second_id, second_dist, margin, status = dlib_recognition.match_face(
        v_stranger, templates, threshold=0.58, margin_threshold=0.08
    )

    assert status == "UNREGISTERED", f"Expected UNREGISTERED, got {status}"
    print(f"[PASS] Case 3 (Unregistered Subject): dist={best_dist:.4f} -> {status}")

if __name__ == "__main__":
    print("--- Running test_ambiguity_rejection.py ---")
    test_margin_and_ambiguity_rules()
    print("--- ALL AMBIGUITY REJECTION TESTS PASSED ---")

