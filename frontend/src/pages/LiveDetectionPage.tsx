import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Header, TabType } from '../components/Header';
import { WebcamView } from '../components/WebcamView';
import { RightPanel } from '../components/RightPanel';
import { BottomPanel } from '../components/BottomPanel';
import { DashboardView } from '../views/DashboardView';
import { EmployeesView } from '../views/EmployeesView';
import { EventsView } from '../views/EventsView';
import { SettingsView } from '../views/SettingsView';
import { RecognitionTestingView } from '../views/RecognitionTestingView';
import { TrainingView } from '../views/TrainingView';
import { fetchHealth, fetchModels, selectModel, getWebSocketUrl, fetchSettings, updateSettings, fetchCameras, saveCamera, deleteCamera } from '../services/api';
import { HealthStatus, ModelInfo, IntegratedDetectionResponse, DetectionEvent, CameraChannel } from '../types';

const DEFAULT_CLASS_THRESHOLDS: Record<string, number> = {
  helmet: 0.15,
  safety_vest: 0.15,
  gloves: 0.15,
  safety_shoes: 0.15,
  goggles: 0.15,
  mask: 0.15,
  ear_protection: 0.15,
  person: 0.25,
};

export const LiveDetectionPage: React.FC = () => {
  // Navigation tab state
  const [activeTab, setActiveTab] = useState<TabType>('live');

  // Backend & model states
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('combined_sh17');
  const [isModelSwitching, setIsModelSwitching] = useState<boolean>(false);
  const [modelSwitchMessage, setModelSwitchMessage] = useState<string | null>(null);

  // Multi-Camera Channels with localStorage persistence
  const [cameras, setCameras] = useState<CameraChannel[]>(() => {
    try {
      const stored = localStorage.getItem('ppe_camera_channels');
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch {}
    return [
      { id: 'CAM-01', name: 'Camera 1 (Main Gate)', source_type: 'webcam', is_active: true }
    ];
  });
  const [activeCameraId, setActiveCameraId] = useState<string>(() => {
    try {
      const stored = localStorage.getItem('ppe_active_camera_id');
      if (stored) return stored;
    } catch {}
    return 'CAM-01';
  });

  // App & Detection states
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [isDetecting, setIsDetecting] = useState<boolean>(false);
  const [detectionResult, setDetectionResult] = useState<IntegratedDetectionResponse | null>(null);
  const [events, setEvents] = useState<DetectionEvent[]>([]);

  // Detection configuration (calibrated for combined SH17 8-class sensitivity)
  const [confThreshold, setConfThreshold] = useState<number>(0.15);
  const [classThresholds, setClassThresholds] = useState<Record<string, number>>(DEFAULT_CLASS_THRESHOLDS);
  const [targetFps, setTargetFps] = useState<number>(8);
  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const [showLabels, setShowLabels] = useState<boolean>(true);

  // WebSocket state & reference
  const wsConnectedRef = useRef<boolean>(false);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const isAwaitingResponseRef = useRef<boolean>(false);
  const saveTimeoutRef = useRef<number | null>(null);
  const lastLoggedTimesRef = useRef<Record<string, number>>({});

  // 1. Initial data fetch (Health, Models, System Settings & Cameras)
  const loadSystemInfo = useCallback(async () => {
    try {
      const [healthData, modelsData, settingsData, camerasData] = await Promise.all([
        fetchHealth().catch(() => null),
        fetchModels().catch(() => null),
        fetchSettings().catch(() => null),
        fetchCameras().catch(() => null),
      ]);

      if (healthData) {
        setHealth(healthData);
        if (healthData.active_model) {
          setSelectedModelId(healthData.active_model);
        }
      }
      if (modelsData && Array.isArray(modelsData)) {
        setModels(modelsData);
      }
      if (settingsData && settingsData.class_thresholds) {
        const merged = { ...DEFAULT_CLASS_THRESHOLDS, ...settingsData.class_thresholds };
        if (merged.person && merged.person > 0.30) {
          merged.person = 0.20;
        }
        setClassThresholds(merged);
      }
      if (camerasData && Array.isArray(camerasData) && camerasData.length > 0) {
        setCameras((prev) => {
          const remoteIds = new Set(camerasData.map((c) => c.id));
          const localOnly = prev.filter((c) => !remoteIds.has(c.id));
          const merged = [...camerasData, ...localOnly];
          try { localStorage.setItem('ppe_camera_channels', JSON.stringify(merged)); } catch {}
          return merged;
        });
        setActiveCameraId((curr) => {
          const exists = camerasData.some((c) => c.id === curr);
          const active = exists ? curr : camerasData[0].id;
          try { localStorage.setItem('ppe_active_camera_id', active); } catch {}
          return active;
        });
      }
    } catch (err) {
      console.warn('System info load error:', err);
    }
  }, []);

  const handleCreateCamera = useCallback(async (name: string, sourceType: 'webcam' | 'video' | 'photo') => {
    const newId = `CAM-${String(Date.now()).slice(-4)}`;
    const newCam: CameraChannel = {
      id: newId,
      name,
      source_type: sourceType,
      device_id: '',
      media_filename: '',
      is_active: true,
      created_at: new Date().toISOString()
    };
    // 1. Optimistic UI update: added camera displays immediately on the UI
    setCameras((prev) => {
      const next = [...prev.filter((c) => c.id !== newId), newCam];
      try { localStorage.setItem('ppe_camera_channels', JSON.stringify(next)); } catch {}
      return next;
    });
    setActiveCameraId(newId);
    try { localStorage.setItem('ppe_active_camera_id', newId); } catch {}

    // 2. Persist to backend and synchronize
    try {
      const saved = await saveCamera(newCam);
      if (saved && saved.id) {
        setCameras((prev) => {
          const updated = prev.map((c) => (c.id === newId ? saved : c));
          try { localStorage.setItem('ppe_camera_channels', JSON.stringify(updated)); } catch {}
          return updated;
        });
      }
    } catch (err) {
      console.warn('Backend failed to persist camera:', err);
    }
  }, []);

  const handleDeleteCamera = useCallback(async (camId: string) => {
    setCameras((prev) => {
      const remaining = prev.filter((c) => c.id !== camId);
      try { localStorage.setItem('ppe_camera_channels', JSON.stringify(remaining)); } catch {}
      if (remaining.length > 0) {
        setActiveCameraId((curr) => {
          const nextActive = curr === camId ? remaining[0].id : curr;
          try { localStorage.setItem('ppe_active_camera_id', nextActive); } catch {}
          return nextActive;
        });
      }
      return remaining;
    });
    try {
      await deleteCamera(camId);
    } catch (err) {
      console.warn('Failed to delete camera:', err);
    }
  }, []);

  const handleUpdateCameraSource = useCallback(async (camId: string, sourceType: 'webcam' | 'video' | 'photo', filename?: string) => {
    setCameras((prev) => {
      const cam = prev.find((c) => c.id === camId);
      if (!cam) return prev;
      const updated = {
        ...cam,
        source_type: sourceType,
        media_filename: filename !== undefined ? filename : cam.media_filename,
      };
      saveCamera({
        id: updated.id,
        name: updated.name,
        source_type: updated.source_type,
        device_id: updated.device_id,
        media_filename: updated.media_filename,
        is_active: updated.is_active,
      }).catch((e) => console.warn('Failed to persist camera update:', e));
      return prev.map((c) => (c.id === camId ? updated : c));
    });
  }, []);

  const handleClassThresholdChange = useCallback((className: string, val: number) => {
    setClassThresholds((prev) => {
      const updated = { ...prev, [className]: val };
      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = window.setTimeout(() => {
        updateSettings({ class_thresholds: updated }).catch((err) =>
          console.warn('Failed to persist thresholds:', err)
        );
      }, 1000);
      return updated;
    });
  }, []);

  const handleResetClassThresholds = useCallback(() => {
    setClassThresholds(DEFAULT_CLASS_THRESHOLDS);
    updateSettings({ class_thresholds: DEFAULT_CLASS_THRESHOLDS }).catch((err) =>
      console.warn('Failed to reset thresholds:', err)
    );
  }, []);

  useEffect(() => {
    loadSystemInfo();
    const interval = setInterval(loadSystemInfo, 10000);
    return () => clearInterval(interval);
  }, [loadSystemInfo]);

  // 2. WebSocket connection management
  const connectWebSocket = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.CONNECTING || wsRef.current.readyState === WebSocket.OPEN)) {
      return;
    }

    try {
      const wsUrl = getWebSocketUrl();
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setWsConnected(true);
        isAwaitingResponseRef.current = false;
      };

      ws.onmessage = (event) => {
        isAwaitingResponseRef.current = false;
        try {
          const data = JSON.parse(event.data);
          if (data && data.success) {
            const resp = data as IntegratedDetectionResponse;
            setDetectionResult(resp);

            // Log detection events for identified persons (Activity log captured 3s per shot)
            if (resp.persons && resp.persons.length > 0) {
              const nowMs = Date.now();
              const newEvents: DetectionEvent[] = [];

              resp.persons.forEach((p, i) => {
                const empName = p.face ? p.face.employee_name : (p.person_label || `Person ${i + 1}`);
                const personKey = p.person_id || empName;
                const lastLogged = lastLoggedTimesRef.current[personKey] || 0;

                // Rate limit: 3 seconds per shot
                if (nowMs - lastLogged >= 3000) {
                  lastLoggedTimesRef.current[personKey] = nowMs;
                  const timeStr = new Date(nowMs).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                  const isKitWorn = p.ppe_kit_worn ?? (p.overall_result === 'COMPLIANT');

                  newEvents.push({
                    id: `${nowMs}-${i}-${empName}`,
                    timestamp: timeStr,
                    className: isKitWorn ? 'PPE Kit: WORN ✓' : `PPE Kit: NOT WORN (Missing ${p.missing_ppe.join(', ')})`,
                    confidence: p.face ? p.face.match_score : 1.0,
                    employeeName: empName,
                    complianceStatus: p.overall_result,
                    color: isKitWorn ? '#10b981' : '#ef4444',
                  });
                }
              });

              if (newEvents.length > 0) {
                setEvents((prev) => [...newEvents, ...prev].slice(0, 50));
              }
            }
          }
        } catch (err) {
          console.warn('Failed to parse WebSocket detection frame:', err);
        }
      };

      ws.onerror = (err) => {
        console.warn('WebSocket error:', err);
      };

      ws.onclose = () => {
        setWsConnected(false);
        wsRef.current = null;
        isAwaitingResponseRef.current = false;

        if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connectWebSocket();
        }, 3000);
      };

      wsRef.current = ws;
    } catch (err) {
      console.warn('Failed to initialize WebSocket:', err);
      setWsConnected(false);
    }
  }, []);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  // 3. Dispatch frame to WebSocket
  const handleFrameCaptured = useCallback(
    (base64Frame: string) => {
      if (!isDetecting) return;
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        return;
      }
      if (isAwaitingResponseRef.current) {
        return;
      }

      try {
        isAwaitingResponseRef.current = true;
        const payload = JSON.stringify({
          image: base64Frame,
          conf_threshold: confThreshold,
          class_thresholds: classThresholds,
          camera_id: activeCameraId,
          imgsz: 416,
        });
        wsRef.current.send(payload);
      } catch (err) {
        console.warn('Error sending frame to WebSocket:', err);
        isAwaitingResponseRef.current = false;
      }
    },
    [isDetecting, confThreshold, classThresholds, activeCameraId]
  );

  // 4. Model selection
  const handleSelectModel = async (modelId: string) => {
    if (modelId === selectedModelId || isModelSwitching) return;

    setIsModelSwitching(true);
    setModelSwitchMessage(null);

    try {
      const resp = await selectModel(modelId);
      if (resp.success) {
        setSelectedModelId(modelId);
        setModelSwitchMessage(`Active model updated to ${resp.active_model}`);
        await loadSystemInfo();
      } else {
        setModelSwitchMessage(`Error: ${resp.message}`);
      }
    } catch (err: any) {
      setModelSwitchMessage(`Error: ${err.message || 'Model switch failed'}`);
    } finally {
      setIsModelSwitching(false);
      setTimeout(() => setModelSwitchMessage(null), 3500);
    }
  };

  const activeModel = models.find((m) => m.id === selectedModelId) || null;

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 flex flex-col font-sans">
      {/* Top Application Header with Navigation Tabs */}
      <Header
        health={health}
        activeModel={activeModel}
        isCameraActive={isCameraActive}
        isDetecting={isDetecting}
        wsConnected={wsConnected}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 w-full flex-1 flex flex-col gap-6">
        {activeTab === 'dashboard' && (
          <DashboardView onNavigate={(tab) => setActiveTab(tab)} />
        )}

        {activeTab === 'live' && (
          <>
            <div className="flex flex-col lg:flex-row gap-6 items-start">
              {/* Webcam Viewport */}
              <div className="flex-1 w-full min-w-0">
                <WebcamView
                  onFrameCaptured={handleFrameCaptured}
                  detectionResult={detectionResult}
                  isDetecting={isDetecting}
                  onToggleDetecting={() => setIsDetecting((prev) => !prev)}
                  showBoxes={showBoxes}
                  showLabels={showLabels}
                  targetFps={targetFps}
                  onCameraActiveChange={setIsCameraActive}
                  cameras={cameras}
                  activeCameraId={activeCameraId}
                  onSelectCamera={setActiveCameraId}
                  onCreateCamera={handleCreateCamera}
                  onDeleteCamera={handleDeleteCamera}
                  onUpdateCameraSource={handleUpdateCameraSource}
                />
              </div>

              {/* Right Control & Diagnostics Panel */}
              <RightPanel
                models={models}
                selectedModelId={selectedModelId}
                onSelectModel={handleSelectModel}
                isModelSwitching={isModelSwitching}
                modelSwitchMessage={modelSwitchMessage}
                confThreshold={confThreshold}
                onConfThresholdChange={setConfThreshold}
                classThresholds={classThresholds}
                onClassThresholdChange={handleClassThresholdChange}
                onResetClassThresholds={handleResetClassThresholds}
                targetFps={targetFps}
                onTargetFpsChange={setTargetFps}
                lastDetection={detectionResult}
                showBoxes={showBoxes}
                onToggleShowBoxes={() => setShowBoxes((prev) => !prev)}
                showLabels={showLabels}
                onToggleShowLabels={() => setShowLabels((prev) => !prev)}
              />
            </div>

            {/* Lower section: Events Stream & System Details */}
            <BottomPanel
              events={events}
              onClearEvents={() => setEvents([])}
              activeModelName={activeModel?.name || 'YOLO26n PPE Detector'}
            />
          </>
        )}

        {activeTab === 'training' && <TrainingView />}

        {activeTab === 'employees' && <EmployeesView />}

        {activeTab === 'testing' && <RecognitionTestingView />}

        {activeTab === 'events' && <EventsView />}

        {activeTab === 'settings' && <SettingsView />}
      </main>
    </div>
  );
};
