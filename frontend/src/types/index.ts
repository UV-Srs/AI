export interface DetectionItem {
  box: [number, number, number, number]; // [x1, y1, x2, y2]
  class_id: number;
  class_name: string;
  confidence: number;
}

export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
  px: number;
  py: number;
}

export interface MeshContours {
  face_oval: [number, number][];
  left_eye: [number, number][];
  right_eye: [number, number][];
  lips: [number, number][];
  left_eyebrow: [number, number][];
  right_eyebrow: [number, number][];
}

export interface HeadPose {
  yaw: number;
  pitch: number;
  roll: number;
  angle: 'FRONT' | 'LEFT' | 'RIGHT' | 'TILT_UP' | 'TILT_DOWN';
}

export interface FaceMesh {
  bbox: [number, number, number, number];
  landmarks_count: number;
  landmarks: LandmarkPoint[];
  contours: MeshContours;
  pose: HeadPose;
}

export interface FaceData {
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  score: number;
  employee_id: string | null;
  employee_name: string;
  department: string;
  designation: string;
  match_score: number;
  second_match_score?: number;
  match_margin?: number;
  identity_state: 'REGISTERED' | 'UNREGISTERED' | 'UNCERTAIN' | 'LIVENESS_REVIEW';
  mesh?: FaceMesh | null;
  pose?: HeadPose | null;
  quality: {
    acceptable: boolean;
    message: string;
    metrics: Record<string, number>;
  };
  liveness: {
    state: string;
    score: number;
  };
}

export interface AssociatedPerson {
  person_id?: string;
  person_label?: string;
  person_box: [number, number, number, number];
  face: FaceData | null;
  associated_ppe: DetectionItem[];
  checklist: Record<string, 'DETECTED' | 'MISSING' | 'NOT_REQUIRED'>;
  missing_ppe: string[];
  required_ppe: string[];
  ppe_kit_worn?: boolean;
  ppe_kit_status?: string;
  has_helmet?: boolean;
  has_vest?: boolean;
  overall_result: 'COMPLIANT' | 'NON_COMPLIANT' | 'NEEDS_REVIEW';
}

export interface IntegratedDetectionResponse {
  success: boolean;
  persons: AssociatedPerson[];
  unassociated_ppe: DetectionItem[];
  total_persons: number;
  compliant_count: number;
  violation_count: number;
  needs_review_count: number;
  inference_time_ms: number;
  fps: number;
  telemetry?: {
    fps: number;
    total_time_ms: number;
    yunet_time_ms: number;
    mediapipe_time_ms: number;
    dlib_time_ms: number;
    yolo_time_ms: number;
  };
  camera_id: string;
  class_thresholds?: Record<string, number>;
  peak_scores?: Record<string, number>;
  frame_width: number;
  frame_height: number;
  error?: string;
}

export interface PairCompareResponse {
  distance: number;
  best_threshold: number;
  uncertain_threshold: number;
  status: 'REGISTERED' | 'UNCERTAIN' | 'UNREGISTERED';
  verdict: string;
  explanation: string;
  image_a_quality: {
    metrics: Record<string, number>;
    message: string;
  };
  image_b_quality: {
    metrics: Record<string, number>;
    message: string;
  };
}

export type DetectionResponse = IntegratedDetectionResponse;

export interface DetectionEvent {
  id: string;
  timestamp: string;
  className: string;
  confidence: number;
  color?: string;
  employeeName?: string;
  complianceStatus?: string;
}

export interface Employee {
  id: string;
  name: string;
  department: string;
  designation: string;
  required_ppe: string[];
  registration_date?: string;
  is_active: boolean;
  consent_status: 'CONSENTED' | 'WITHDRAWN' | 'PENDING';
  consent_date?: string;
  photo_path?: string;
  has_biometric_template: boolean;
  enrollment_status?: 'PENDING' | 'ACTIVE' | 'REVOKED';
  completed_angles?: string[];
}

export interface BiometricDetails {
  employee_id: string;
  full_name?: string;
  consent_status: string;
  enrollment_status: string;
  completed_angles: string[];
  is_complete: boolean;
  captures: {
    id?: string;
    capture_angle: string;
    embedding_dimension: number;
    embedding_model: string;
    embedding_model_version: string;
    facial_geometry: Record<string, any>;
    face_pose: Record<string, any>;
    face_quality: Record<string, any>;
    created_at: string;
  }[];
}

export interface InspectionEvent {
  id: string;
  timestamp: string;
  camera_id: string;
  employee_id?: string;
  employee_name?: string;
  identity_state: string;
  match_score: number;
  ppe_summary: Record<string, string>;
  missing_ppe: string[];
  overall_result: 'COMPLIANT' | 'NON_COMPLIANT' | 'NEEDS_REVIEW';
  model_id: string;
}

export interface ComplianceStats {
  total_scans: number;
  compliant_scans: number;
  violations: number;
  compliance_rate: number;
  active_employees: number;
  enrolled_faces: number;
}

export interface SystemSettings {
  face_match_threshold: number;
  event_cooldown_seconds: number;
  retention_days: number;
  default_required_ppe: string[];
  class_thresholds?: Record<string, number>;
}

export interface ModelInfo {
  id: string;
  name: string;
  filename: string;
  path: string;
  size_mb: number;
  classes_count: number;
  classes: string[];
  is_active: boolean;
  exists: boolean;
}

export interface HealthStatus {
  status: string;
  app_name: string;
  device: string;
  active_model: string | null;
  model_loaded: boolean;
  classes: string[];
  face_engine_ready: boolean;
}

export interface CameraDevice {
  deviceId: string;
  label: string;
}

export interface CameraChannel {
  id: string;
  name: string;
  source_type: 'webcam' | 'video' | 'photo';
  device_id?: string;
  media_filename?: string;
  media_url?: string;
  is_active: boolean;
  created_at?: string;
}

export interface LearningStatus {
  status: 'idle' | 'training' | 'completed' | 'failed';
  progress: number;
  epoch: number;
  total_epochs: number;
  queue_length?: number;
  message: string;
  total_learned_samples: number;
  updated_at: string;
}

export interface LearningIngestResponse {
  success: boolean;
  sample_id: string;
  samples_added: number;
  media_type: 'image' | 'video';
  face_enrolled: boolean;
  classes_learned: string[];
  total_learned_samples: number;
  message: string;
}

export interface TrainingMetrics {
  has_metrics: boolean;
  total_epochs_recorded: number;
  latest: {
    epoch: number;
    'train/box_loss'?: number;
    'train/cls_loss'?: number;
    'train/dfl_loss'?: number;
    'metrics/precision(B)'?: number;
    'metrics/recall(B)'?: number;
    'metrics/mAP50(B)'?: number;
    'metrics/mAP50-95(B)'?: number;
    'lr/pg0'?: number;
    time?: number;
    [key: string]: any;
  };
  history: any[];
}

export interface TrainingDatasetSummary {
  total_train_images: number;
  total_val_images: number;
  total_learned_samples: number;
  total_bounding_boxes: number;
  class_distribution: Record<string, number>;
  tensor_input_specs: {
    resolution: string;
    channels: number;
    channel_order: string;
    data_type: string;
    pixel_range: string;
    batch_size: number;
    architecture: string;
    device: string;
  };
}

export interface TrainingSampleItem {
  filename: string;
  is_learned: boolean;
  image_url: string;
  annotations_count: number;
  annotations: {
    class_id: number;
    class_name: string;
    x_center: number;
    y_center: number;
    width: number;
    height: number;
    x1_norm: number;
    y1_norm: number;
    x2_norm: number;
    y2_norm: number;
  }[];
  file_size_kb: number;
}
