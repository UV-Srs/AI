import numpy as np
from typing import List, Tuple, Dict, Any, Optional

def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxB_area = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def associate_yunet_and_mediapipe(
    yunet_faces: List[Dict[str, Any]],
    mediapipe_meshes: List[Dict[str, Any]],
    iou_threshold: float = 0.30
) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """
    Correlates YuNet face bounding boxes with MediaPipe 3D face mesh detections.
    Ensures that both models refer to the exact same anatomical subject.
    """
    paired_results = []
    used_mesh_indices = set()

    for face in yunet_faces:
        y_box = face["bbox"]
        best_iou = 0.0
        best_mesh_idx = None

        for m_idx, mesh in enumerate(mediapipe_meshes):
            if m_idx in used_mesh_indices:
                continue
            m_box = mesh["bbox"]
            iou = compute_iou(y_box, m_box)
            if iou > best_iou:
                best_iou = iou
                best_mesh_idx = m_idx

        if best_mesh_idx is not None and best_iou >= iou_threshold:
            used_mesh_indices.add(best_mesh_idx)
            paired_results.append((face, mediapipe_meshes[best_mesh_idx]))
        else:
            paired_results.append((face, None))

    return paired_results

def expand_face_bbox_proportionally(
    bbox: List[int],
    frame_width: int,
    frame_height: int,
    pad_ratio: float = 0.12
) -> List[int]:
    """Expands face bounding box slightly for optimal dlib landmark alignment."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    px = int(w * pad_ratio)
    py = int(h * pad_ratio)

    return [
        max(0, x1 - px),
        max(0, y1 - py),
        min(frame_width, x2 + px),
        min(frame_height, y2 + py)
    ]

