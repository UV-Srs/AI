import React, { useRef, useEffect, useState, useCallback } from 'react';
import {
  Camera, CameraOff, Play, Square, Columns,
  Maximize2, Eye, Scan, Plus, Trash2, Upload,
  Sparkles, Brain, CheckCircle2, AlertCircle, RefreshCw,
  Layers, Film, Image as ImageIcon, X, Check, Users, UserCheck
} from 'lucide-react';
import { CameraDevice, CameraChannel, IntegratedDetectionResponse, LearningStatus } from '../types';
import { PPE_CLASSES } from '../services/classes';
import {
  uploadCameraMedia,
  ingestLearningMedia,
  triggerModelTraining,
  fetchLearningStatus,
  inspectMediaForLearning,
  MediaInspectionResult,
  BACKEND_URL
} from '../services/api';

interface WebcamViewProps {
  onFrameCaptured: (base64Frame: string) => void;
  detectionResult: IntegratedDetectionResponse | null;
  isDetecting: boolean;
  onToggleDetecting: () => void;
  showBoxes: boolean;
  showLabels: boolean;
  targetFps: number;
  onCameraActiveChange?: (active: boolean) => void;
  cameras: CameraChannel[];
  activeCameraId: string;
  onSelectCamera: (camId: string) => void;
  onCreateCamera: (name: string, sourceType: 'webcam' | 'video' | 'photo') => void;
  onDeleteCamera: (camId: string) => void;
  onUpdateCameraSource: (camId: string, sourceType: 'webcam' | 'video' | 'photo', filename?: string) => void;
}

export const WebcamView: React.FC<WebcamViewProps> = ({
  onFrameCaptured,
  detectionResult,
  isDetecting,
  onToggleDetecting,
  showBoxes,
  showLabels,
  targetFps,
  onCameraActiveChange,
  cameras,
  activeCameraId,
  onSelectCamera,
  onCreateCamera,
  onDeleteCamera,
  onUpdateCameraSource,
}) => {
  // Video and Media references
  const videoRef = useRef<HTMLVideoElement | null>(null); // Live webcam stream
  const uploadedVideoRef = useRef<HTMLVideoElement | null>(null); // Uploaded video
  const uploadedImgRef = useRef<HTMLImageElement | null>(null); // Uploaded photo
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Layout & Device states
  const [showMesh, setShowMesh] = useState<boolean>(false);
  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>('');
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // New Camera creation inline state
  const [isAddingCamera, setIsAddingCamera] = useState<boolean>(false);
  const [newCamName, setNewCamName] = useState<string>('');
  const [newCamType, setNewCamType] = useState<'webcam' | 'video' | 'photo'>('webcam');

  // Media upload state
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadedMediaUrl, setUploadedMediaUrl] = useState<string | null>(null);

  // Self-Learning Modal state
  const [showLearningModal, setShowLearningModal] = useState<boolean>(false);
  const [learnedEmpName, setLearnedEmpName] = useState<string>('');
  const [learnedClasses, setLearnedClasses] = useState<string[]>(['helmet', 'safety_vest']);
  const [learningEpochs, setLearningEpochs] = useState<number>(2);
  const [isLearning, setIsLearning] = useState<boolean>(false);
  const [learningProgress, setLearningProgress] = useState<number>(0);
  const [learningMsg, setLearningMsg] = useState<string>('');
  const [learningSuccess, setLearningSuccess] = useState<boolean | null>(null);
  const [inspectionResult, setInspectionResult] = useState<MediaInspectionResult | null>(null);
  const [isInspecting, setIsInspecting] = useState<boolean>(false);

  const streamRef = useRef<MediaStream | null>(null);
  const lastCaptureTimeRef = useRef<number>(0);

  const activeCamera = cameras.find((c) => c.id === activeCameraId) || cameras[0] || {
    id: 'CAM-01',
    name: 'Camera 1',
    source_type: 'webcam',
    is_active: true,
  };

  // 1. Enumerate available video devices
  const refreshDevices = useCallback(async () => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = allDevices
        .filter((d) => d.kind === 'videoinput')
        .map((d, idx) => ({
          deviceId: d.deviceId,
          label: d.label || `Camera ${idx + 1}`,
        }));
      setDevices(videoInputs);
      if (videoInputs.length > 0 && !selectedDevice) {
        setSelectedDevice(videoInputs[0].deviceId);
      }
    } catch (err: any) {
      console.warn('Error reading camera devices:', err);
    }
  }, [selectedDevice]);

  useEffect(() => {
    refreshDevices();
  }, [refreshDevices]);

  // 2. Start Live Webcam
  const startCamera = async () => {
    setCameraError(null);
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }

      const constraints: MediaStreamConstraints = {
        video: selectedDevice
          ? { deviceId: { exact: selectedDevice }, width: { ideal: 1280, max: 1920 }, height: { ideal: 720, max: 1080 } }
          : { width: { ideal: 1280, max: 1920 }, height: { ideal: 720, max: 1080 } },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current.play();
          }
        };
      }

      setIsCameraActive(true);
      onCameraActiveChange?.(true);
    } catch (err: any) {
      setCameraError(`Camera access denied: ${err.message || 'Check camera permissions.'}`);
      setIsCameraActive(false);
      onCameraActiveChange?.(false);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
    onCameraActiveChange?.(false);

    const canvas = overlayCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // Update media url when activeCamera changes
  useEffect(() => {
    if (activeCamera.media_filename) {
      setUploadedMediaUrl(`${BACKEND_URL}/api/media/${activeCamera.media_filename}`);
    } else {
      setUploadedMediaUrl(null);
    }
  }, [activeCamera.media_filename]);

  // Handle media file upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadError(null);
    try {
      const resp = await uploadCameraMedia(activeCamera.id, file);
      if (resp.success) {
        const fullUrl = resp.media_url.startsWith('http') ? resp.media_url : `${BACKEND_URL}${resp.media_url}`;
        setUploadedMediaUrl(fullUrl);
        onUpdateCameraSource(activeCamera.id, resp.source_type as any, resp.filename);
        // Automatically start detection if currently idle
        if (!isDetecting) {
          onToggleDetecting();
        }
      }
    } catch (err: any) {
      const errMsg = err?.message || 'Check server connection';
      const friendlyMsg = errMsg.toLowerCase().includes('failed to fetch')
        ? 'Unable to connect to backend server at http://127.0.0.1:8001. Please make sure python backend/run.py is running.'
        : `Media upload failed: ${errMsg}`;
      setUploadError(friendlyMsg);
    } finally {
      setIsUploading(false);
      e.target.value = '';
    }
  };

  // Extract current frame as Blob from active media source (webcam, uploaded video, or photo)
  const getActiveSourceBlob = async (): Promise<Blob | null> => {
    let sourceEl: CanvasImageSource | null = null;
    let sWidth = 0;
    let sHeight = 0;

    if (activeCamera.source_type === 'webcam') {
      if (videoRef.current && videoRef.current.readyState >= 2) {
        sourceEl = videoRef.current;
        sWidth = videoRef.current.videoWidth;
        sHeight = videoRef.current.videoHeight;
      }
    } else if (activeCamera.source_type === 'video') {
      if (uploadedVideoRef.current && uploadedVideoRef.current.readyState >= 2) {
        sourceEl = uploadedVideoRef.current;
        sWidth = uploadedVideoRef.current.videoWidth;
        sHeight = uploadedVideoRef.current.videoHeight;
      }
    } else if (activeCamera.source_type === 'photo') {
      if (uploadedImgRef.current && uploadedImgRef.current.complete) {
        sourceEl = uploadedImgRef.current;
        sWidth = uploadedImgRef.current.naturalWidth;
        sHeight = uploadedImgRef.current.naturalHeight;
      }
    }

    if (!sourceEl || sWidth === 0 || sHeight === 0) return null;

    const canvas = document.createElement('canvas');
    canvas.width = Math.min(sWidth, 640);
    const scale = canvas.width / sWidth;
    canvas.height = Math.round(sHeight * scale);
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.drawImage(sourceEl, 0, 0, canvas.width, canvas.height);

    return new Promise<Blob | null>((resolve) => {
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.85);
    });
  };

  // Inspect media detections (faces + PPE gear) using YOLO26n and SFace biometrics
  const runMediaInspection = useCallback(async () => {
    setIsInspecting(true);
    setInspectionResult(null);
    try {
      let res: MediaInspectionResult;
      if (activeCamera.media_filename) {
        res = await inspectMediaForLearning({
          filename: activeCamera.media_filename,
          media_type: activeCamera.source_type === 'video' ? 'video' : 'image',
        });
      } else {
        const blob = await getActiveSourceBlob();
        if (!blob) {
          setIsInspecting(false);
          return;
        }
        res = await inspectMediaForLearning({
          file: blob,
          media_type: 'image',
        });
      }

      setInspectionResult(res);
      if (res.detected_classes && res.detected_classes.length > 0) {
        setLearnedClasses(Array.from(new Set([...res.detected_classes])));
      }
      if (res.recognized_employee) {
        setLearnedEmpName(res.recognized_employee);
      }
    } catch (err: any) {
      console.error('Inspection failed:', err);
    } finally {
      setIsInspecting(false);
    }
  }, [activeCamera.media_filename, activeCamera.source_type]);

  // Run inspection when modal opens
  useEffect(() => {
    if (showLearningModal) {
      runMediaInspection();
    }
  }, [showLearningModal, activeCamera.media_filename, activeCamera.id, runMediaInspection]);

  // 3. Adaptive Frame Dispatch
  const captureAndSend = useCallback(() => {
    if (!isDetecting) return;

    let sourceEl: CanvasImageSource | null = null;
    let sWidth = 0;
    let sHeight = 0;

    if (activeCamera.source_type === 'webcam') {
      if (!isCameraActive || !videoRef.current || videoRef.current.readyState < 2) return;
      sourceEl = videoRef.current;
      sWidth = videoRef.current.videoWidth;
      sHeight = videoRef.current.videoHeight;
    } else if (activeCamera.source_type === 'video') {
      if (!uploadedVideoRef.current || uploadedVideoRef.current.readyState < 2) return;
      sourceEl = uploadedVideoRef.current;
      sWidth = uploadedVideoRef.current.videoWidth;
      sHeight = uploadedVideoRef.current.videoHeight;
    } else if (activeCamera.source_type === 'photo') {
      if (!uploadedImgRef.current || !uploadedImgRef.current.complete) return;
      sourceEl = uploadedImgRef.current;
      sWidth = uploadedImgRef.current.naturalWidth;
      sHeight = uploadedImgRef.current.naturalHeight;
    }

    if (!sourceEl || sWidth === 0 || sHeight === 0) return;

    const targetWidth = 640;
    const scale = targetWidth / sWidth;
    const targetHeight = Math.round(sHeight * scale);

    let canvas = captureCanvasRef.current;
    if (!canvas) {
      canvas = document.createElement('canvas');
      captureCanvasRef.current = canvas;
    }
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(sourceEl, 0, 0, targetWidth, targetHeight);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.70);
    onFrameCaptured(dataUrl);
  }, [activeCamera.source_type, isCameraActive, isDetecting, onFrameCaptured]);

  // Frame capture loop
  useEffect(() => {
    if (!isDetecting) return;

    const minIntervalMs = Math.max(60, Math.round(1000 / targetFps));
    const now = performance.now();
    const elapsed = now - lastCaptureTimeRef.current;
    const delay = Math.max(0, minIntervalMs - elapsed);

    const timer = setTimeout(() => {
      lastCaptureTimeRef.current = performance.now();
      captureAndSend();
    }, delay);

    return () => clearTimeout(timer);
  }, [detectionResult, isDetecting, targetFps, captureAndSend]);

  // 4. Draw Clean Overlays onto Overlay Canvas
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;

    let displayWidth = 0;
    let displayHeight = 0;

    if (activeCamera.source_type === 'webcam' && videoRef.current) {
      displayWidth = videoRef.current.clientWidth;
      displayHeight = videoRef.current.clientHeight;
    } else if (activeCamera.source_type === 'video' && uploadedVideoRef.current) {
      displayWidth = uploadedVideoRef.current.clientWidth;
      displayHeight = uploadedVideoRef.current.clientHeight;
    } else if (activeCamera.source_type === 'photo' && uploadedImgRef.current) {
      displayWidth = uploadedImgRef.current.clientWidth;
      displayHeight = uploadedImgRef.current.clientHeight;
    }

    if (displayWidth === 0 || displayHeight === 0) return;

    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!isDetecting || !detectionResult || !showBoxes) {
      return;
    }

    const origW = detectionResult.frame_width || 640;
    const origH = detectionResult.frame_height || 480;
    const scaleX = displayWidth / origW;
    const scaleY = displayHeight / origH;

    const drawRoundRect = (x: number, y: number, w: number, h: number, r: number) => {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    };

    // Draw detected persons
    if (detectionResult.persons && detectionResult.persons.length > 0) {
      detectionResult.persons.forEach((person, idx) => {
        const [x1, y1, x2, y2] = person.person_box;
        const rx1 = x1 * scaleX;
        const ry1 = y1 * scaleY;
        const rw = (x2 - x1) * scaleX;
        const rh = (y2 - y1) * scaleY;

        const isCompliant = person.overall_result === 'COMPLIANT';
        const isViolation = person.overall_result === 'NON_COMPLIANT';
        const statusColor = isCompliant ? '#10b981' : isViolation ? '#ef4444' : '#f59e0b';
        const personTitle = person.face ? person.face.employee_name : (person.person_label || `Person ${idx + 1}`);

        // 1. Person Boundary Box
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = statusColor;
        ctx.fillStyle = `${statusColor}15`;
        drawRoundRect(rx1, ry1, rw, rh, 8);
        ctx.fill();
        ctx.stroke();

        // 2. Status Label
        if (showLabels) {
          ctx.font = 'bold 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
          const headerText = `${personTitle} — ${isCompliant ? 'COMPLIANT ✓' : 'VIOLATION ⚠'}`;
          const m = ctx.measureText(headerText);
          const tagW = m.width + 16;
          const tagH = 22;
          const tagY = Math.max(0, ry1 - tagH - 2);

          ctx.fillStyle = statusColor;
          drawRoundRect(rx1, tagY, tagW, tagH, 4);
          ctx.fill();

          ctx.fillStyle = '#ffffff';
          ctx.fillText(headerText, rx1 + 8, tagY + 15);
        }

        // 3. Face Bounding Box
        if (person.face) {
          const [fx1, fy1, fx2, fy2] = person.face.bbox;
          const rfx1 = fx1 * scaleX;
          const rfy1 = fy1 * scaleY;
          const rfw = (fx2 - fx1) * scaleX;
          const rfh = (fy2 - fy1) * scaleY;
          const isReg = person.face.identity_state === 'REGISTERED';
          const faceColor = isReg ? '#10b981' : '#3b82f6';

          ctx.lineWidth = 1.5;
          ctx.strokeStyle = faceColor;
          drawRoundRect(rfx1, rfy1, rfw, rfh, 4);
          ctx.stroke();

          if (showMesh && person.face.mesh?.landmarks) {
            ctx.fillStyle = 'rgba(6, 182, 212, 0.5)';
            person.face.mesh.landmarks.forEach((lm) => {
              ctx.beginPath();
              ctx.arc(lm.px * scaleX, lm.py * scaleY, 1, 0, 2 * Math.PI);
              ctx.fill();
            });
          }
        }

        // 4. Associated PPE Boxes
        if (person.associated_ppe && person.associated_ppe.length > 0) {
          person.associated_ppe.forEach((ppe) => {
            const [px1, py1, px2, py2] = ppe.box;
            const bx1 = px1 * scaleX;
            const by1 = py1 * scaleY;
            const bw = (px2 - px1) * scaleX;
            const bh = (py2 - py1) * scaleY;
            const cfg = PPE_CLASSES[ppe.class_name] || { color: '#0284c7', label: ppe.class_name };

            ctx.lineWidth = 2;
            ctx.strokeStyle = cfg.color;
            ctx.fillStyle = `${cfg.color}20`;
            drawRoundRect(bx1, by1, bw, bh, 4);
            ctx.fill();
            ctx.stroke();

            if (showLabels) {
              const confPct = Math.round(ppe.confidence * 100);
              const label = `${cfg.label.split('/')[0].trim()} ${confPct}%`;
              ctx.font = 'bold 10px sans-serif';
              const lm = ctx.measureText(label);
              const lw = lm.width + 8;
              const lh = 16;
              const ly = by1 - lh >= 0 ? by1 - lh : by1;

              ctx.fillStyle = cfg.color;
              drawRoundRect(bx1, ly, lw, lh, 3);
              ctx.fill();

              ctx.fillStyle = '#ffffff';
              ctx.fillText(label, bx1 + 4, ly + 12);
            }
          });
        }
      });
    }
  }, [detectionResult, isDetecting, showBoxes, showLabels, showMesh, activeCamera.source_type]);

  // Background fine-tuning status polling (continues active even when modal is closed)
  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const s = await fetchLearningStatus();
        if (!active) return;
        if (s.status === 'training') {
          setIsLearning(true);
          setLearningProgress(s.progress || 20);
          setLearningMsg(s.message || 'Fine-tuning in background...');
        } else if (isLearning && s.status === 'completed') {
          setIsLearning(false);
          setLearningProgress(100);
          setLearningSuccess(true);
          setLearningMsg(s.message || 'Model fine-tuning complete!');
        } else if (isLearning && s.status === 'failed') {
          setIsLearning(false);
          setLearningSuccess(false);
          setLearningMsg(s.message || 'Training failed');
        }
      } catch {
        // ignore transient network glitch
      }
    };

    poll();
    const interval = setInterval(poll, isLearning ? 1200 : 3500);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [isLearning]);

  // Execute Learning & Fine-Tuning (runs in background and queues if another pass is running)
  const handleStartLearning = async () => {
    setIsLearning(true);
    setLearningProgress(15);
    setLearningMsg('Ingesting training media and extracting annotations...');
    setLearningSuccess(null);

    try {
      // 1. Capture current frame as blob if live webcam or no filename
      let fileBlob: Blob | undefined;
      if (activeCamera.source_type === 'webcam' || !activeCamera.media_filename) {
        const b = await getActiveSourceBlob();
        if (b) fileBlob = b;
      }

      // 2. Ingest into dataset
      const ingestResp = await ingestLearningMedia({
        file: fileBlob ? (new File([fileBlob], 'learned_frame.jpg', { type: 'image/jpeg' })) : undefined,
        filename: activeCamera.media_filename || undefined,
        camera_id: activeCamera.id,
        media_type: activeCamera.source_type === 'video' ? 'video' : 'image',
        employee_name: learnedEmpName.trim() || undefined,
        assigned_classes: learnedClasses,
      });

      setLearningProgress(35);
      const addedText = ingestResp.samples_added > 0 ? `${ingestResp.samples_added} new sample(s)` : 'media';
      setLearningMsg(`Ingested ${addedText}. Dispatching fine-tuning job...`);

      // 3. Trigger fine-tuning (runs in background, queues automatically if another training is active)
      const trainResp = await triggerModelTraining(learningEpochs);

      if (trainResp.queued) {
        setLearningMsg(trainResp.message || 'Training already active. Your new photos have been queued and will train automatically next!');
      } else {
        setLearningMsg(trainResp.message || 'Fine-tuning running in background...');
      }

    } catch (err: any) {
      setIsLearning(false);
      setLearningSuccess(false);
      setLearningMsg(`Learning error: ${err.message}`);
    }
  };

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-xs overflow-hidden flex flex-col select-none">
      {/* 1. Multi-Camera Channel Switcher Bar */}
      <div className="px-4 py-2.5 bg-gray-50 border-b border-gray-200 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 overflow-x-auto py-0.5 max-w-full">
          {cameras.map((cam) => {
            const isSelected = cam.id === activeCamera.id;
            return (
              <div
                key={cam.id}
                className={`shrink-0 flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold cursor-pointer transition-all border ${
                  isSelected
                    ? 'bg-blue-600 text-white border-blue-600 shadow-2xs'
                    : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-100'
                }`}
                onClick={() => onSelectCamera(cam.id)}
              >
                {cam.source_type === 'webcam' ? (
                  <Camera className="w-3.5 h-3.5" />
                ) : cam.source_type === 'video' ? (
                  <Film className="w-3.5 h-3.5" />
                ) : (
                  <ImageIcon className="w-3.5 h-3.5" />
                )}
                <span>{cam.name}</span>
                {cameras.length > 1 && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteCamera(cam.id);
                    }}
                    className={`ml-1 p-0.5 rounded hover:bg-black/20 ${isSelected ? 'text-white/80' : 'text-gray-400'}`}
                    title="Delete camera channel"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>
            );
          })}

          {/* Add Camera Button */}
          {!isAddingCamera ? (
            <button
              type="button"
              onClick={() => setIsAddingCamera(true)}
              className="shrink-0 flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-semibold text-blue-700 hover:bg-blue-50 border border-dashed border-blue-300 transition-colors"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Add Camera</span>
            </button>
          ) : (
            <form
              onSubmit={(e) => {
                e.preventDefault();
                const name = newCamName.trim() || `Camera ${cameras.length + 1}`;
                onCreateCamera(name, newCamType);
                setNewCamName('');
                setIsAddingCamera(false);
              }}
              className="shrink-0 flex items-center gap-1.5 bg-white p-1 rounded-lg border border-blue-400 shadow-sm"
            >
              <input
                type="text"
                placeholder={`Camera ${cameras.length + 1}`}
                value={newCamName}
                onChange={(e) => setNewCamName(e.target.value)}
                className="text-xs px-2 py-1 border border-gray-300 rounded focus:outline-blue-500 w-36"
                autoFocus
              />
              <select
                value={newCamType}
                onChange={(e) => setNewCamType(e.target.value as any)}
                className="text-xs px-1.5 py-1 border border-gray-300 rounded bg-gray-50 font-medium"
              >
                <option value="webcam">Live Webcam</option>
                <option value="video">Upload Video</option>
                <option value="photo">Upload Photo</option>
              </select>
              <button
                type="submit"
                className="px-2.5 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-xs font-semibold transition-colors"
              >
                Create
              </button>
              <button
                type="button"
                onClick={() => setIsAddingCamera(false)}
                className="p-1 text-gray-400 hover:text-gray-600"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </form>
          )}
        </div>

        {/* Source Toggle for Active Camera */}
        <div className="flex items-center gap-1 bg-gray-200/70 p-0.5 rounded-lg text-xs">
          <button
            type="button"
            onClick={() => onUpdateCameraSource(activeCamera.id, 'webcam')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-md font-medium transition-colors ${
              activeCamera.source_type === 'webcam' ? 'bg-white text-blue-700 font-semibold shadow-2xs' : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <Camera className="w-3 h-3" />
            <span>Webcam</span>
          </button>
          <button
            type="button"
            onClick={() => onUpdateCameraSource(activeCamera.id, 'video')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-md font-medium transition-colors ${
              activeCamera.source_type === 'video' ? 'bg-white text-blue-700 font-semibold shadow-2xs' : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <Film className="w-3 h-3" />
            <span>Video</span>
          </button>
          <button
            type="button"
            onClick={() => onUpdateCameraSource(activeCamera.id, 'photo')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-md font-medium transition-colors ${
              activeCamera.source_type === 'photo' ? 'bg-white text-blue-700 font-semibold shadow-2xs' : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <ImageIcon className="w-3 h-3" />
            <span>Photo</span>
          </button>
        </div>
      </div>

      {/* Upload Error / Warning Banner */}
      {uploadError && (
        <div className="mx-4 my-2 p-2.5 bg-red-50 border border-red-200 text-red-800 rounded-lg text-xs flex items-center justify-between shadow-2xs">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-red-600 shrink-0" />
            <span className="font-medium">{uploadError}</span>
          </div>
          <button
            type="button"
            onClick={() => setUploadError(null)}
            className="text-red-500 hover:text-red-700 p-1 rounded"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* 2. Main Viewport Canvas Area */}
      <div className="bg-slate-950 p-2 relative flex flex-col justify-center items-center">
        <div className="relative w-full aspect-video min-h-[440px] bg-slate-900 rounded-lg overflow-hidden border border-slate-800 flex items-center justify-center">
          
          {/* Header pill overlay */}
          <div className="absolute top-2 left-2 z-10 flex items-center gap-2 pointer-events-none">
            <div className="bg-black/75 backdrop-blur-xs border border-white/10 px-2.5 py-1 rounded-md flex items-center gap-1.5 shadow-sm">
              <Scan className="w-3.5 h-3.5 text-blue-400" />
              <span className="text-[11px] font-bold text-white tracking-wide">
                {activeCamera.name.toUpperCase()} ({activeCamera.source_type.toUpperCase()})
              </span>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse ml-0.5" />
            </div>
          </div>

          {/* Performance Pill Overlay */}
          {isDetecting && detectionResult && (
            <div className="absolute top-2 right-2 z-10 flex items-center gap-2 pointer-events-none">
              <div className="bg-black/75 backdrop-blur-xs border border-white/10 px-2.5 py-1 rounded-md flex items-center gap-1.5 shadow-sm">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="text-[11px] font-semibold text-white">
                  {detectionResult.fps > 0 ? `${detectionResult.fps} FPS` : 'Detecting...'}
                </span>
                <span className="text-[10px] text-slate-400">({detectionResult.inference_time_ms}ms)</span>
              </div>
            </div>
          )}

          {/* Floating Background Training Status Indicator (Visible when modal is closed) */}
          {isLearning && !showLearningModal && (
            <div
              onClick={() => setShowLearningModal(true)}
              className="absolute top-12 right-2 z-20 cursor-pointer bg-indigo-900/90 hover:bg-indigo-950 text-white px-3 py-1.5 rounded-full shadow-lg border border-indigo-400/40 backdrop-blur-md flex items-center gap-2 text-xs font-medium transition-all group"
              title="Click to open training monitor"
            >
              <RefreshCw className="w-3.5 h-3.5 animate-spin text-indigo-300" />
              <span>Fine-Tuning ML ({learningProgress}%)</span>
              <span className="text-[10px] bg-indigo-700 group-hover:bg-indigo-600 px-1.5 py-0.5 rounded font-bold text-indigo-100">
                View
              </span>
            </div>
          )}

          {/* SOURCE A: Live Webcam Stream */}
          {activeCamera.source_type === 'webcam' && (
            <div className="relative w-full h-full flex items-center justify-center">
              <video
                ref={videoRef}
                playsInline
                muted
                className={`w-full h-full object-contain ${isCameraActive ? 'block' : 'hidden'}`}
              />
              {!isCameraActive && (
                <div className="flex flex-col items-center justify-center p-6 text-center text-slate-400">
                  <Camera className="w-10 h-10 mb-2 opacity-40 text-blue-400" />
                  <span className="text-sm font-semibold text-slate-200">Live Webcam Offline</span>
                  <span className="text-xs text-slate-400 mt-1">Click "Turn Camera ON" below to start physical stream</span>
                </div>
              )}
            </div>
          )}

          {/* SOURCE B: Uploaded Video Player */}
          {activeCamera.source_type === 'video' && (
            <div className="relative w-full h-full flex items-center justify-center">
              {uploadedMediaUrl ? (
                <video
                  ref={uploadedVideoRef}
                  src={uploadedMediaUrl}
                  crossOrigin="anonymous"
                  playsInline
                  loop
                  autoPlay
                  controls
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="flex flex-col items-center justify-center p-6 text-center text-slate-400">
                  <Film className="w-10 h-10 mb-2 opacity-40 text-blue-400" />
                  <span className="text-sm font-semibold text-slate-200">No Video Uploaded</span>
                  <span className="text-xs text-slate-400 mt-1">Upload an inspection MP4/WebM video below</span>
                </div>
              )}
            </div>
          )}

          {/* SOURCE C: Uploaded Photo Viewer */}
          {activeCamera.source_type === 'photo' && (
            <div className="relative w-full h-full flex items-center justify-center">
              {uploadedMediaUrl ? (
                <img
                  ref={uploadedImgRef}
                  src={uploadedMediaUrl}
                  crossOrigin="anonymous"
                  alt="Uploaded Inspection"
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="flex flex-col items-center justify-center p-6 text-center text-slate-400">
                  <ImageIcon className="w-10 h-10 mb-2 opacity-40 text-blue-400" />
                  <span className="text-sm font-semibold text-slate-200">No Photo Uploaded</span>
                  <span className="text-xs text-slate-400 mt-1">Upload an inspection JPG/PNG photo below</span>
                </div>
              )}
            </div>
          )}

          {/* Bounding Box Canvas Overlay */}
          <canvas
            ref={overlayCanvasRef}
            className={`absolute inset-0 pointer-events-none w-full h-full object-contain ${
              isDetecting ? 'block' : 'hidden'
            }`}
          />
        </div>

        {/* Camera error notification */}
        {cameraError && (
          <div className="w-full mt-2 p-2 bg-red-950/80 border border-red-800 text-red-200 text-xs rounded flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-red-400" />
            <span>{cameraError}</span>
          </div>
        )}
      </div>

      {/* 3. Action Control Bar */}
      <div className="p-3 bg-white border-t border-gray-200 flex flex-wrap items-center justify-between gap-3">
        {/* Left: Device & Source Specific Controls */}
        <div className="flex items-center gap-2.5">
          {activeCamera.source_type === 'webcam' ? (
            <>
              {isCameraActive ? (
                <button
                  type="button"
                  onClick={stopCamera}
                  className="px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-gray-100 hover:bg-gray-200 text-gray-800 flex items-center gap-1.5 transition-colors border border-gray-300"
                >
                  <CameraOff className="w-3.5 h-3.5 text-red-600" />
                  <span>Turn Camera OFF</span>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={startCamera}
                  className="px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-blue-600 hover:bg-blue-700 text-white flex items-center gap-1.5 transition-colors shadow-2xs"
                >
                  <Camera className="w-3.5 h-3.5" />
                  <span>Turn Camera ON</span>
                </button>
              )}

              {devices.length > 0 && (
                <select
                  value={selectedDevice}
                  onChange={(e) => {
                    setSelectedDevice(e.target.value);
                    if (isCameraActive) {
                      stopCamera();
                      setTimeout(startCamera, 150);
                    }
                  }}
                  className="bg-gray-50 border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs font-medium text-gray-700 focus:outline-blue-500 max-w-[180px]"
                >
                  {devices.map((d) => (
                    <option key={d.deviceId} value={d.deviceId}>
                      {d.label}
                    </option>
                  ))}
                </select>
              )}
            </>
          ) : (
            <label className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-50 hover:bg-gray-100 border border-gray-300 rounded-lg text-xs font-semibold text-gray-700 cursor-pointer transition-colors">
              <Upload className="w-3.5 h-3.5 text-blue-600" />
              <span>{isUploading ? 'Uploading...' : activeCamera.source_type === 'video' ? 'Upload Video' : 'Upload Photo'}</span>
              <input
                type="file"
                accept={activeCamera.source_type === 'video' ? 'video/*' : 'image/*'}
                onChange={handleFileUpload}
                disabled={isUploading}
                className="hidden"
              />
            </label>
          )}

          {/* Toggle Mesh */}
          <button
            type="button"
            onClick={() => setShowMesh((prev) => !prev)}
            className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1 transition-colors border ${
              showMesh ? 'bg-cyan-50 text-cyan-700 border-cyan-300' : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50'
            }`}
            title="Toggle 3D face mesh dots"
          >
            <Layers className="w-3 h-3" />
            <span>{showMesh ? 'Mesh ON' : 'Mesh'}</span>
          </button>
        </div>

        {/* Right: Primary Action Buttons */}
        <div className="flex items-center gap-2">
          {/* Start/Stop AI Detection Button */}
          <button
            type="button"
            onClick={onToggleDetecting}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-colors shadow-2xs ${
              isDetecting
                ? 'bg-amber-600 hover:bg-amber-700 text-white'
                : 'bg-emerald-600 hover:bg-emerald-700 text-white'
            }`}
          >
            {isDetecting ? (
              <>
                <Square className="w-3.5 h-3.5 fill-current" />
                <span>Pause Detection</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Start Detection</span>
              </>
            )}
          </button>

          {/* Self-Learning & Retrain Button */}
          <button
            type="button"
            onClick={() => setShowLearningModal(true)}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-colors shadow-2xs text-white ${
              isLearning
                ? 'bg-purple-600 hover:bg-purple-700 animate-pulse'
                : 'bg-indigo-600 hover:bg-indigo-700'
            }`}
            title="Ingest current media to teach the ML new faces and hard hats"
          >
            {isLearning ? (
              <RefreshCw className="w-3.5 h-3.5 animate-spin text-purple-200" />
            ) : (
              <Brain className="w-3.5 h-3.5 text-indigo-200" />
            )}
            <span>{isLearning ? `Fine-Tuning (${learningProgress}%)` : 'Train ML from Media'}</span>
          </button>
        </div>
      </div>

      {/* 4. Self-Learning & Continuous Training Modal */}
      {showLearningModal && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-2xs flex items-center justify-center p-3 sm:p-4">
          <div className="bg-white rounded-xl shadow-2xl border border-gray-200 max-w-2xl w-full max-h-[90vh] flex flex-col overflow-hidden select-none animate-in fade-in zoom-in-95 duration-150">
            {/* Modal Header */}
            <div className="px-5 py-3.5 bg-gray-50 border-b border-gray-200 flex items-center justify-between shrink-0">
              <div className="flex items-center gap-2">
                <div className="p-1.5 bg-indigo-100 text-indigo-700 rounded-lg">
                  <Brain className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-gray-900">Train ML from Media</h3>
                  <p className="text-[11px] text-gray-500">Continuous biometric and PPE model training</p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setShowLearningModal(false)}
                className="p-1 text-gray-400 hover:text-gray-600 rounded transition-colors"
                title="Close modal (training continues in background)"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Modal Body (Scrollable) */}
            <div className="p-5 space-y-4 overflow-y-auto flex-1">
              {/* Media Detection & Inspection Section */}
              <div className="bg-slate-50 border border-slate-200 rounded-xl p-3.5 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-indigo-600" />
                    <span className="text-xs font-bold text-gray-900">What the ML Detected:</span>
                  </div>
                  <button
                    type="button"
                    onClick={runMediaInspection}
                    disabled={isInspecting}
                    className="px-2.5 py-1 text-[11px] font-semibold text-indigo-700 bg-white border border-indigo-200 rounded-md hover:bg-indigo-50 flex items-center gap-1.5 shadow-2xs transition-colors"
                    title="Re-run ML detection on current media"
                  >
                    <RefreshCw className={`w-3 h-3 ${isInspecting ? 'animate-spin text-indigo-600' : ''}`} />
                    <span>{isInspecting ? 'Analyzing...' : 'Re-scan Media'}</span>
                  </button>
                </div>

                {isInspecting ? (
                  <div className="py-8 flex flex-col items-center justify-center gap-2 text-gray-500">
                    <RefreshCw className="w-6 h-6 animate-spin text-indigo-600" />
                    <span className="text-xs font-medium">Running YOLO26n & SFace biometrics to detect people and PPE...</span>
                  </div>
                ) : inspectionResult ? (
                  <div className="space-y-3">
                    {/* Visual Annotated Frame Preview */}
                    {inspectionResult.preview_image && (
                      <div className="relative rounded-lg overflow-hidden border border-gray-300 bg-black aspect-video flex items-center justify-center group shadow-2xs">
                        <img
                          src={inspectionResult.preview_image}
                          alt="ML Detections Annotated Preview"
                          className="w-full h-full object-contain"
                        />
                        <div className="absolute top-2 left-2 bg-black/80 backdrop-blur-2xs text-white text-[10px] font-mono px-2 py-0.5 rounded shadow">
                          {inspectionResult.summary}
                        </div>
                        <div className="absolute bottom-2 right-2 bg-indigo-950/80 backdrop-blur-2xs text-indigo-200 text-[10px] font-mono px-2 py-0.5 rounded">
                          ⚡ {inspectionResult.inference_time_ms} ms
                        </div>
                      </div>
                    )}

                    {/* Breakdown Badges & Cards */}
                    <div className="space-y-1.5">
                      <div className="text-[11px] font-semibold text-gray-700 flex items-center justify-between">
                        <span>Detected Entities:</span>
                        <span className="text-[10px] text-gray-500">
                          {inspectionResult.detections.length + inspectionResult.faces.length} items detected
                        </span>
                      </div>

                      <div className="flex flex-wrap gap-1.5">
                        {/* Person/Face badge */}
                        {inspectionResult.faces.length > 0 ? (
                          inspectionResult.faces.map((f, i) => (
                            <span
                              key={i}
                              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold bg-blue-100 text-blue-900 border border-blue-200"
                            >
                              <UserCheck className="w-3.5 h-3.5 text-blue-600" />
                              <span>{f.employee_name || 'Worker'}</span>
                              <span className="text-[10px] bg-blue-200/90 text-blue-950 px-1 py-0.5 rounded font-mono font-bold">
                                {f.confidence_pct}%
                              </span>
                              <span className="text-[9px] text-blue-700">({f.quality})</span>
                            </span>
                          ))
                        ) : inspectionResult.persons_count > 0 ? (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold bg-blue-100 text-blue-900 border border-blue-200">
                            <Users className="w-3.5 h-3.5 text-blue-600" />
                            <span>{inspectionResult.persons_count} Worker(s) Detected</span>
                          </span>
                        ) : null}

                        {/* PPE Detection Badges */}
                        {inspectionResult.detections.length > 0 ? (
                          inspectionResult.detections.map((d, i) => {
                            const confColor =
                              d.class_name === 'helmet'
                                ? 'bg-emerald-100 text-emerald-900 border-emerald-300'
                                : d.class_name === 'safety_vest'
                                ? 'bg-orange-100 text-orange-900 border-orange-300'
                                : d.class_name === 'gloves'
                                ? 'bg-amber-100 text-amber-900 border-amber-300'
                                : d.class_name === 'safety_shoes'
                                ? 'bg-indigo-100 text-indigo-900 border-indigo-300'
                                : 'bg-purple-100 text-purple-900 border-purple-300';

                            return (
                              <span
                                key={i}
                                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold border ${confColor}`}
                              >
                                <CheckCircle2 className="w-3 h-3 shrink-0" />
                                <span>{d.class_name.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase())}</span>
                                <span className="font-mono text-[10px] bg-white/80 px-1 rounded font-bold">
                                  {d.confidence_pct}%
                                </span>
                              </span>
                            );
                          })
                        ) : (
                          <span className="text-xs text-amber-800 bg-amber-50 border border-amber-200 px-2.5 py-1 rounded-md flex items-center gap-1">
                            <AlertCircle className="w-3.5 h-3.5 text-amber-600 shrink-0" />
                            No PPE detected in this shot. Check gear below to teach the ML.
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="p-3 text-center text-xs text-gray-500 bg-white rounded-lg border border-dashed border-gray-300">
                    Click <strong className="text-indigo-600 cursor-pointer" onClick={runMediaInspection}>Re-scan Media</strong> to inspect what the ML detects on this media.
                  </div>
                )}
              </div>

              {/* Subject Face Name */}
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold text-gray-700">
                    Assign Person / Employee (Optional)
                  </label>
                  {inspectionResult?.recognized_employee && (
                    <span className="text-[10px] text-emerald-700 font-medium bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                      Recognized: {inspectionResult.recognized_employee}
                    </span>
                  )}
                </div>
                <input
                  type="text"
                  placeholder="e.g. Sarah Connor (Safety Officer)"
                  value={learnedEmpName}
                  onChange={(e) => setLearnedEmpName(e.target.value)}
                  disabled={isLearning}
                  className="w-full text-xs p-2 border border-gray-300 rounded-lg focus:outline-indigo-500 bg-white"
                />
                <span className="text-[10px] text-gray-400">
                  If entered, facial embeddings will be saved to your employee registry.
                </span>
              </div>

              {/* Target Gear Checkboxes */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold text-gray-700 block">
                    PPE Gear to Learn from this Media:
                  </label>
                  {inspectionResult?.detected_classes && inspectionResult.detected_classes.length > 0 && (
                    <span className="text-[10px] text-indigo-600 font-medium">
                      Auto-selected from detections
                    </span>
                  )}
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                  {[
                    { id: 'helmet', label: 'Hard Hat / Helmet' },
                    { id: 'safety_vest', label: 'Safety Vest' },
                    { id: 'gloves', label: 'Safety Gloves' },
                    { id: 'safety_shoes', label: 'Safety Boots' },
                    { id: 'goggles', label: 'Safety Goggles' },
                    { id: 'mask', label: 'Face Mask' },
                  ].map((g) => {
                    const checked = learnedClasses.includes(g.id);
                    const isDetected = inspectionResult?.detected_classes?.includes(g.id);
                    return (
                      <label
                        key={g.id}
                        className={`flex items-center justify-between p-2 rounded-lg border cursor-pointer text-xs transition-colors ${
                          checked
                            ? 'bg-indigo-50 border-indigo-300 text-indigo-900 font-medium'
                            : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={checked}
                            disabled={isLearning}
                            onChange={(e) => {
                              if (e.target.checked) setLearnedClasses((prev) => [...prev, g.id]);
                              else setLearnedClasses((prev) => prev.filter((k) => k !== g.id));
                            }}
                            className="accent-indigo-600 rounded"
                          />
                          <span>{g.label}</span>
                        </div>
                        {isDetected && (
                          <span className="text-[9px] bg-emerald-100 text-emerald-800 px-1 py-0.5 rounded font-bold">
                            AI ✓
                          </span>
                        )}
                      </label>
                    );
                  })}
                </div>
              </div>

              {/* Training Epochs */}
              <div className="space-y-1 pt-1">
                <div className="flex justify-between text-xs">
                  <span className="font-semibold text-gray-700">Fine-Tuning Epochs</span>
                  <span className="font-mono font-bold text-indigo-600">{learningEpochs} Epochs</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="5"
                  step="1"
                  value={learningEpochs}
                  onChange={(e) => setLearningEpochs(parseInt(e.target.value, 10))}
                  disabled={isLearning}
                  className="w-full accent-indigo-600 h-1.5 bg-gray-200 rounded-lg cursor-pointer"
                />
                <span className="text-[10px] text-gray-400 block">
                  Recommended: 2 epochs (fast CPU fine-tuning without blocking live detection).
                </span>
              </div>

              {/* Progress & Feedback */}
              {isLearning && (
                <div className="space-y-2 pt-2 border-t border-gray-100">
                  <div className="flex justify-between text-xs">
                    <span className="font-medium text-gray-700 flex items-center gap-1.5">
                      <RefreshCw className="w-3.5 h-3.5 animate-spin text-indigo-600" />
                      <span>{learningMsg || 'Training in progress...'}</span>
                    </span>
                    <span className="font-mono font-bold text-indigo-600">{learningProgress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                    <div
                      className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${learningProgress}%` }}
                    />
                  </div>
                </div>
              )}

              {learningSuccess !== null && !isLearning && (
                <div
                  className={`p-3 rounded-lg text-xs flex items-center gap-2 ${
                    learningSuccess ? 'bg-emerald-50 text-emerald-800 border border-emerald-200' : 'bg-red-50 text-red-800 border border-red-200'
                  }`}
                >
                  {learningSuccess ? (
                    <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-600" />
                  ) : (
                    <AlertCircle className="w-4 h-4 shrink-0 text-red-600" />
                  )}
                  <span>{learningMsg}</span>
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="px-5 py-3 bg-gray-50 border-t border-gray-200 flex items-center justify-between gap-2 shrink-0">
              <span className="text-[11px] text-gray-500 font-medium">
                {isLearning ? '⚡ Fine-tuning active in background' : 'Hot-reloads weights on completion'}
              </span>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowLearningModal(false)}
                  className="px-3.5 py-1.5 text-xs font-semibold text-gray-700 hover:text-gray-900 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors shadow-2xs"
                  title="Close popup (training continues running safely in background)"
                >
                  Close
                </button>
                <button
                  type="button"
                  onClick={handleStartLearning}
                  className={`px-4 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-colors shadow-xs text-white ${
                    isLearning
                      ? 'bg-purple-600 hover:bg-purple-700'
                      : 'bg-indigo-600 hover:bg-indigo-700'
                  }`}
                  title={isLearning ? 'Queue another training pass with newly uploaded media' : 'Start fine-tuning ML on this media'}
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>{isLearning ? `+ Queue Another (${learningEpochs} Epochs)` : 'Start Learning & Retrain'}</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
