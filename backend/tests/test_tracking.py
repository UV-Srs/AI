"""
Unit Test: Multi-Person Tracking & Consecutive Frame Streak
Verifies:
1. Stable tracking IDs (TRK-001, TRK-002, ...)
2. IoU correlation between consecutive frames.
3. 3-frame streak validation before identity confirmation.
4. Track disappearance and cleanup after timeout.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.face.face_tracking import MultiPersonTracker

def test_tracker_stability_and_streaks():
    tracker = MultiPersonTracker()

    # Frame 1: Two persons enter frame
    f1_boxes = [[100, 100, 200, 220], [400, 100, 500, 220]]
    matches_1 = tracker.update(f1_boxes)

    assert len(matches_1) == 2
    tid1, _, track1 = matches_1[0]
    tid2, _, track2 = matches_1[1]

    assert tid1 != tid2
    assert "TRK-" in tid1 and "TRK-" in tid2
    print(f"[PASS] Frame 1: Assigned unique track IDs {tid1} and {tid2}")

    # Simulate recognition for Person 1: Frame 1
    track1.update_identity("EMP-001", 0.40, "REGISTERED", required_streak=3)
    assert track1.confirmed_status == "UNCERTAIN"  # Streak is only 1/3!
    assert track1.consecutive_count == 1

    # Frame 2: Person 1 moves slightly
    f2_boxes = [[105, 102, 205, 222], [402, 98, 502, 218]]
    matches_2 = tracker.update(f2_boxes)
    f2_dict = {tid: t for tid, b, t in matches_2}

    assert tid1 in f2_dict, f"Track {tid1} lost between frames!"
    assert tid2 in f2_dict, f"Track {tid2} lost between frames!"
    
    # Simulate recognition for Person 1: Frame 2
    f2_dict[tid1].update_identity("EMP-001", 0.41, "REGISTERED", required_streak=3)
    assert f2_dict[tid1].confirmed_status == "UNCERTAIN"  # Streak is 2/3!
    assert f2_dict[tid1].consecutive_count == 2

    # Frame 3: Person 1 commits 3rd streak
    f3_boxes = [[110, 100, 210, 220]]
    matches_3 = tracker.update(f3_boxes)
    f3_dict = {tid: t for tid, b, t in matches_3}
    
    f3_dict[tid1].update_identity("EMP-001", 0.39, "REGISTERED", required_streak=3)
    assert f3_dict[tid1].confirmed_status == "REGISTERED"  # Streak is 3/3!
    assert f3_dict[tid1].confirmed_employee_id == "EMP-001"
    print(f"[PASS] Streak verified: Confirmed REGISTERED identity only after 3 consecutive frames.")

    # Person 2 disappeared in Frame 3. In 6 more empty frames, track2 should expire
    for _ in range(7):
        tracker.update([], max_disappeared=5)

    assert tid2 not in tracker.tracks, "Expired track was not pruned after max_disappeared timeout!"
    print(f"[PASS] Disappeared track {tid2} successfully pruned after timeout.")

if __name__ == "__main__":
    print("--- Running test_tracking.py ---")
    test_tracker_stability_and_streaks()
    print("--- ALL TRACKING TESTS PASSED ---")

