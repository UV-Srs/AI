import threading
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .face_alignment import compute_iou

class TrackedFace:
    def __init__(self, track_id: str, bbox: List[int]):
        self.track_id = track_id
        self.bbox = bbox
        self.centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        self.disappeared_frames = 0
        self.history: List[Tuple[Optional[str], float, str]] = []
        self.confirmed_employee_id: Optional[str] = None
        self.confirmed_status: str = "UNCERTAIN"
        self.confirmed_distance: float = 999.0
        self.consecutive_count: int = 0
        self.last_candidate_id: Optional[str] = None

        # Multi-person temporal caches to eliminate CPU lag
        self.cached_dlib_descriptor: Optional[List[float]] = None
        self.last_dlib_time: float = 0.0
        self.cached_mesh_data: Optional[Dict[str, Any]] = None
        self.last_mesh_time: float = 0.0
        self.cached_match_data: Optional[Dict[str, Any]] = None
        self.last_match_time: float = 0.0

    def update_position(self, bbox: List[int]):
        self.bbox = bbox
        self.centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        self.disappeared_frames = 0

    def update_identity(self, emp_id: Optional[str], distance: float, status: str, required_streak: int = 3):
        self.history.append((emp_id, distance, status))
        if len(self.history) > 20:
            self.history.pop(0)

        # Check for consecutive matching candidate
        if status == "REGISTERED" and emp_id is not None:
            if emp_id == self.last_candidate_id:
                self.consecutive_count += 1
            else:
                self.last_candidate_id = emp_id
                self.consecutive_count = 1

            if self.consecutive_count >= required_streak:
                self.confirmed_employee_id = emp_id
                self.confirmed_status = "REGISTERED"
                self.confirmed_distance = distance
            else:
                # Still validating over consecutive frames
                self.confirmed_status = "UNCERTAIN"
                self.confirmed_distance = distance
        else:
            self.consecutive_count = 0
            self.last_candidate_id = None
            self.confirmed_status = status
            self.confirmed_distance = distance
            if status == "UNREGISTERED":
                self.confirmed_employee_id = None

class MultiPersonTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks: Dict[str, TrackedFace] = {}
        self._lock = threading.Lock()

    def update(
        self,
        detected_bboxes: List[List[int]],
        iou_threshold: float = 0.30,
        max_disappeared: int = 15
    ) -> List[Tuple[str, List[int], TrackedFace]]:
        """
        Associates current frame detections with existing tracks using IoU & centroid proximity.
        Returns list of (track_id, bbox, tracked_face_instance).
        """
        with self._lock:
            if len(detected_bboxes) == 0:
                # Mark all existing tracks as disappeared
                to_delete = []
                for tid, track in self.tracks.items():
                    track.disappeared_frames += 1
                    if track.disappeared_frames > max_disappeared:
                        to_delete.append(tid)
                for tid in to_delete:
                    del self.tracks[tid]
                return []

            if len(self.tracks) == 0:
                # Register all as new tracks
                results = []
                for bbox in detected_bboxes:
                    tid = f"TRK-{self.next_id:03d}"
                    self.next_id += 1
                    track = TrackedFace(tid, bbox)
                    self.tracks[tid] = track
                    results.append((tid, bbox, track))
                return results

            # Compute IoU matrix between existing tracks and new bboxes
            track_ids = list(self.tracks.keys())
            existing_bboxes = [self.tracks[tid].bbox for tid in track_ids]

            iou_matrix = np.zeros((len(existing_bboxes), len(detected_bboxes)), dtype=np.float32)
            for i, e_box in enumerate(existing_bboxes):
                for j, d_box in enumerate(detected_bboxes):
                    iou_matrix[i, j] = compute_iou(e_box, d_box)

            # Greedy Hungarian/linear assignment matching
            assigned_tracks = set()
            assigned_detections = set()

            matches = []
            while True:
                max_val = np.max(iou_matrix)
                if max_val < iou_threshold:
                    break
                idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                i, j = idx[0], idx[1]
                matches.append((track_ids[i], detected_bboxes[j]))
                assigned_tracks.add(track_ids[i])
                assigned_detections.add(j)
                iou_matrix[i, :] = -1.0
                iou_matrix[:, j] = -1.0

            results = []
            for tid, bbox in matches:
                track = self.tracks[tid]
                track.update_position(bbox)
                results.append((tid, bbox, track))

            # New detections that weren't matched -> create new tracks
            for j, bbox in enumerate(detected_bboxes):
                if j not in assigned_detections:
                    tid = f"TRK-{self.next_id:03d}"
                    self.next_id += 1
                    track = TrackedFace(tid, bbox)
                    self.tracks[tid] = track
                    results.append((tid, bbox, track))

            # Existing tracks not matched -> mark disappeared
            to_delete = []
            for tid, track in self.tracks.items():
                if tid not in assigned_tracks:
                    track.disappeared_frames += 1
                    if track.disappeared_frames > max_disappeared:
                        to_delete.append(tid)

            for tid in to_delete:
                del self.tracks[tid]

            return results

face_tracker = MultiPersonTracker()

