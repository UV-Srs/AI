import {
  HealthStatus, ModelInfo, Employee, InspectionEvent,
  ComplianceStats, SystemSettings, BiometricDetails,
  CameraChannel, LearningStatus, LearningIngestResponse,
  TrainingMetrics, TrainingDatasetSummary, TrainingSampleItem
} from '../types';

// Auto-detect dev mode (port 5173 / 5174) to connect directly to FastAPI backend on port 8001
// Normalize localhost to 127.0.0.1 to prevent Windows IPv6 [::1] connection issues
const isLocalDev = typeof window !== 'undefined' &&
  (window.location.port === '5173' || window.location.port === '5174');

const targetHost = typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? '127.0.0.1'
  : (typeof window !== 'undefined' ? window.location.hostname : '127.0.0.1');

export const BACKEND_URL = isLocalDev
  ? `${window.location.protocol}//${targetHost}:8001`
  : '';

export const API_BASE = `${BACKEND_URL}/api`;

/**
 * Resilient fetch with automatic fallback:
 * 1. Attempts direct backend URL (e.g. http://127.0.0.1:8001/api/...)
 * 2. If Failed to fetch / network error occurs, falls back to Vite proxy (/api/...)
 */
export async function resilientFetch(url: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch (err: any) {
    if (err?.name === 'TypeError' || (err?.message && err.message.toLowerCase().includes('fetch'))) {
      if (BACKEND_URL && url.startsWith(BACKEND_URL)) {
        const fallbackUrl = url.substring(BACKEND_URL.length);
        try {
          return await fetch(fallbackUrl, init);
        } catch {
          // Keep original error context
        }
      }
    }
    throw err;
  }
}

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`);
  return res.json();
}

export async function selectModel(modelId: string): Promise<{ success: boolean; message: string; active_model: string }> {
  const res = await fetch(`${API_BASE}/models/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId })
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to select model: ${res.statusText}`);
  }
  return res.json();
}

// ==========================================
// Employee Management APIs
// ==========================================

export async function fetchEmployees(activeOnly: boolean = false): Promise<Employee[]> {
  const res = await fetch(`${API_BASE}/employees?active_only=${activeOnly}`);
  if (!res.ok) throw new Error(`Failed to fetch employees: ${res.statusText}`);
  return res.json();
}

export async function saveEmployee(employee: Partial<Employee>): Promise<{ success: boolean; employee: Employee }> {
  const res = await fetch(`${API_BASE}/employees`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(employee)
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to save employee: ${res.statusText}`);
  }
  return res.json();
}

export async function deleteEmployee(empId: string): Promise<{ success: boolean; message: string }> {
  const res = await fetch(`${API_BASE}/employees/${encodeURIComponent(empId)}`, {
    method: 'DELETE'
  });
  if (!res.ok) throw new Error(`Failed to delete employee: ${res.statusText}`);
  return res.json();
}

export async function toggleEmployeeStatus(empId: string, isActive: boolean): Promise<void> {
  const formData = new FormData();
  formData.append('is_active', isActive ? 'true' : 'false');
  const res = await fetch(`${API_BASE}/employees/${encodeURIComponent(empId)}/status`, {
    method: 'POST',
    body: formData
  });
  if (!res.ok) throw new Error(`Failed to toggle employee status`);
}

export async function withdrawConsent(empId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/employees/${encodeURIComponent(empId)}/consent`, {
    method: 'POST'
  });
  if (!res.ok) throw new Error(`Failed to withdraw consent`);
}

export async function enrollFace(empId: string, imageBase64: string): Promise<{ success: boolean; message: string; quality_metrics: any }> {
  return enrollAngleCapture(empId, 'FRONT', imageBase64);
}

export async function enrollAngleCapture(empId: string, angle: string, imageBase64: string): Promise<{
  success: boolean;
  employee_id: string;
  capture_angle: string;
  completed_angles: string[];
  is_complete: boolean;
  enrollment_status: string;
  message: string;
  pose: any;
  quality_metrics: any;
}> {
  const res = await fetch(`${API_BASE}/employees/enroll-capture`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ employee_id: empId, capture_angle: angle, image_base64: imageBase64 })
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Capture failed: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchBiometrics(empId: string): Promise<BiometricDetails> {
  const res = await fetch(`${API_BASE}/employees/${encodeURIComponent(empId)}/biometrics`);
  if (!res.ok) throw new Error(`Failed to fetch biometric details: ${res.statusText}`);
  return res.json();
}

// ==========================================
// Inspection Events & Stats APIs
// ==========================================

export async function fetchEvents(params?: { limit?: number; employee_id?: string; overall_result?: string; identity_state?: string }): Promise<InspectionEvent[]> {
  const query = new URLSearchParams();
  if (params?.limit) query.append('limit', params.limit.toString());
  if (params?.employee_id) query.append('employee_id', params.employee_id);
  if (params?.overall_result) query.append('overall_result', params.overall_result);
  if (params?.identity_state) query.append('identity_state', params.identity_state);

  const res = await fetch(`${API_BASE}/events?${query.toString()}`);
  if (!res.ok) throw new Error(`Failed to fetch events: ${res.statusText}`);
  return res.json();
}

export function getExportEventsCsvUrl(): string {
  return `${API_BASE}/events/export?limit=1000`;
}

export async function fetchStats(): Promise<ComplianceStats> {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error(`Failed to fetch stats: ${res.statusText}`);
  return res.json();
}

// ==========================================
// Settings APIs
// ==========================================

export async function fetchSettings(): Promise<SystemSettings> {
  const res = await fetch(`${API_BASE}/settings`);
  if (!res.ok) throw new Error(`Failed to fetch settings: ${res.statusText}`);
  return res.json();
}

export async function updateSettings(settings: Partial<SystemSettings>): Promise<void> {
  const res = await fetch(`${API_BASE}/settings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings)
  });
  if (!res.ok) throw new Error(`Failed to update settings`);
}

export function getWebSocketUrl(): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  if (isLocalDev) {
    return `${protocol}//${targetHost}:8001/ws/detection`;
  }
  const host = window.location.host;
  return `${protocol}//${host}/ws/detection`;
}

// ==========================================
// Multi-Camera APIs
// ==========================================

export async function fetchCameras(): Promise<CameraChannel[]> {
  try {
    const res = await fetch(`${API_BASE}/cameras`);
    if (!res.ok) throw new Error(`Failed to fetch cameras: ${res.statusText}`);
    return res.json();
  } catch (err: any) {
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      console.warn('Backend server offline, using local camera defaults.');
    }
    throw err;
  }
}

export async function saveCamera(data: {
  id: string;
  name: string;
  source_type: string;
  device_id?: string;
  media_filename?: string;
  is_active?: boolean;
}): Promise<CameraChannel> {
  const formData = new FormData();
  formData.append('id', data.id);
  formData.append('name', data.name);
  formData.append('source_type', data.source_type);
  if (data.device_id) formData.append('device_id', data.device_id);
  if (data.media_filename) formData.append('media_filename', data.media_filename);
  formData.append('is_active', data.is_active ? 'true' : 'false');

  const res = await resilientFetch(`${API_BASE}/cameras`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(`Failed to save camera`);
  return res.json();
}

export async function deleteCamera(camId: string): Promise<void> {
  const res = await resilientFetch(`${API_BASE}/cameras/${encodeURIComponent(camId)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Failed to delete camera`);
}

export async function uploadCameraMedia(
  camId: string,
  file: File
): Promise<{ success: boolean; filename: string; source_type: string; media_url: string }> {
  const formData = new FormData();
  formData.append('file', file);
  try {
    const res = await resilientFetch(`${API_BASE}/cameras/${encodeURIComponent(camId)}/media`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || `Upload failed (${res.status}): ${res.statusText}`);
    }
    return res.json();
  } catch (err: any) {
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      throw new Error(
        'Backend server (http://127.0.0.1:8001) is offline. Please start the backend service (run start_ppe_detection.ps1 or python backend/run.py).'
      );
    }
    throw err;
  }
}

// ==========================================
// Continuous Self-Learning APIs
// ==========================================

export async function ingestLearningMedia(params: {
  file?: File;
  filename?: string;
  camera_id: string;
  media_type: string;
  employee_id?: string;
  employee_name?: string;
  assigned_classes?: string[];
}): Promise<LearningIngestResponse> {
  const formData = new FormData();
  if (params.file) formData.append('file', params.file);
  if (params.filename) formData.append('filename', params.filename);
  formData.append('camera_id', params.camera_id);
  formData.append('media_type', params.media_type);
  if (params.employee_id) formData.append('employee_id', params.employee_id);
  if (params.employee_name) formData.append('employee_name', params.employee_name);
  if (params.assigned_classes) {
    formData.append('assigned_classes', JSON.stringify(params.assigned_classes));
  }

  const res = await resilientFetch(`${API_BASE}/learning/ingest`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to ingest media for learning');
  }
  return res.json();
}

export async function triggerModelTraining(epochs: number = 2): Promise<{
  success: boolean;
  queued?: boolean;
  queue_length?: number;
  message: string;
  status: LearningStatus;
}> {
  const res = await resilientFetch(`${API_BASE}/learning/train?epochs=${epochs}`, {
    method: 'POST',
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || 'Failed to start model training');
  }
  return res.json();
}

export async function fetchLearningStatus(): Promise<LearningStatus> {
  const res = await resilientFetch(`${API_BASE}/learning/status`);
  if (!res.ok) throw new Error('Failed to fetch learning status');
  return res.json();
}

export interface MediaInspectionResult {
  success: boolean;
  preview_image?: string;
  detections: Array<{
    class_name: string;
    confidence: number;
    confidence_pct: number;
    box: [number, number, number, number];
  }>;
  detected_classes: string[];
  faces: Array<{
    bbox: [number, number, number, number];
    confidence: number;
    confidence_pct: number;
    employee_id?: string;
    employee_name?: string;
    quality: string;
    is_recognized: boolean;
  }>;
  persons_count: number;
  recognized_employee?: string;
  summary: string;
  inference_time_ms: number;
  error?: string;
}

export async function inspectMediaForLearning(params: {
  file?: File | Blob;
  filename?: string;
  media_type?: string;
}): Promise<MediaInspectionResult> {
  const formData = new FormData();
  if (params.file) {
    if (params.file instanceof File) {
      formData.append('file', params.file);
    } else {
      formData.append('file', params.file, 'capture.jpg');
    }
  }
  if (params.filename) formData.append('filename', params.filename);
  formData.append('media_type', params.media_type || 'image');

  const res = await resilientFetch(`${API_BASE}/learning/inspect-media`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || 'Failed to inspect media detections');
  }
  return res.json();
}

// ==========================================
// Model Training & Inspection APIs
// ==========================================

export async function fetchTrainingMetrics(): Promise<TrainingMetrics> {
  const res = await fetch(`${API_BASE}/training/metrics`);
  if (!res.ok) throw new Error('Failed to fetch training metrics');
  return res.json();
}

export async function fetchTrainingDatasetSummary(): Promise<TrainingDatasetSummary> {
  const res = await fetch(`${API_BASE}/training/dataset-summary`);
  if (!res.ok) throw new Error('Failed to fetch training dataset summary');
  return res.json();
}

export async function fetchTrainingSamples(limit: number = 40, sampleType: 'all' | 'learned' | 'base' = 'all'): Promise<TrainingSampleItem[]> {
  const res = await fetch(`${API_BASE}/training/samples?limit=${limit}&sample_type=${sampleType}`);
  if (!res.ok) throw new Error('Failed to fetch training samples');
  return res.json();
}
