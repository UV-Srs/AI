import React, { useState, useEffect } from 'react';
import {
  Settings, Save, Lock, Sliders, CheckCircle2,
  AlertCircle, RefreshCw, Database
} from 'lucide-react';
import { SystemSettings } from '../types';
import { fetchSettings, updateSettings } from '../services/api';

const PPE_OPTIONS = [
  { id: 'helmet', label: 'Helmet / Hard Hat' },
  { id: 'safety_vest', label: 'Safety Vest' },
  { id: 'goggles', label: 'Safety Goggles / Glasses' },
  { id: 'gloves', label: 'Gloves' },
  { id: 'safety_shoes', label: 'Safety Boots / Shoes' },
  { id: 'mask', label: 'Mask / Respirator' },
  { id: 'ear_protection', label: 'Ear Protection / Muffs' },
];

export const SettingsView: React.FC = () => {
  const [settings, setSettings] = useState<SystemSettings>({
    face_match_threshold: 0.363,
    event_cooldown_seconds: 20,
    retention_days: 90,
    default_required_ppe: ['helmet', 'safety_vest'],
  });

  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const loadSettings = async () => {
    try {
      setLoading(true);
      const data = await fetchSettings();
      if (data) {
        setSettings(data);
      }
    } catch (err: any) {
      console.error('Failed to load settings:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSettings();
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setStatusMessage(null);

    try {
      await updateSettings(settings);
      setStatusMessage({ type: 'success', text: 'System settings saved and applied successfully.' });
    } catch (err: any) {
      setStatusMessage({ type: 'error', text: err.message || 'Failed to update system settings.' });
    } finally {
      setSaving(false);
      setTimeout(() => setStatusMessage(null), 4000);
    }
  };

  return (
    <div className="flex flex-col gap-6 w-full select-none font-sans">
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-white border border-gray-200 p-6 rounded-xl shadow-sm">
        <div>
          <h2 className="text-xl font-bold text-gray-900 tracking-tight flex items-center gap-2.5">
            <Settings className="w-5 h-5 text-blue-600" />
            <span>System Settings</span>
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            Configure face recognition matching threshold, event logging cooldown, and privacy settings.
          </p>
        </div>

        <button
          type="button"
          onClick={loadSettings}
          disabled={loading}
          className="p-2 bg-white hover:bg-gray-50 text-gray-600 rounded-lg border border-gray-300 transition-colors shadow-sm"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <form onSubmit={handleSave} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Calibration & Thresholds */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm space-y-6">
            <div className="flex items-center gap-2 border-b border-gray-100 pb-3">
              <Sliders className="w-4 h-4 text-blue-600" />
              <h3 className="text-sm font-bold text-gray-900">
                Detection & Verification Thresholds
              </h3>
            </div>

            {/* Threshold 1: Face Match Cosine Threshold */}
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs">
                <div>
                  <span className="font-semibold text-gray-800 text-sm">
                    Face Recognition Match Threshold
                  </span>
                  <p className="text-xs text-gray-500 mt-0.5">
                    Recommended: <span className="font-mono text-blue-600 font-semibold">0.363</span> (SFace 128-d cosine distance threshold; false match rate &lt; 0.1%).
                  </p>
                </div>
                <span className="text-base font-bold font-mono text-blue-600">
                  {settings.face_match_threshold.toFixed(3)}
                </span>
              </div>
              <input
                type="range"
                min="0.200"
                max="0.650"
                step="0.005"
                value={settings.face_match_threshold}
                onChange={(e) =>
                  setSettings({ ...settings, face_match_threshold: parseFloat(e.target.value) })
                }
                className="w-full accent-blue-600 h-2 bg-gray-200 rounded-lg cursor-pointer"
              />
              <div className="flex justify-between text-xs text-gray-400 font-mono">
                <span>0.200 (Lenient)</span>
                <span>0.363 (Recommended)</span>
                <span>0.650 (Strict)</span>
              </div>
            </div>

            {/* Threshold 2: Event Logging Cooldown */}
            <div className="space-y-2 pt-4 border-t border-gray-100">
              <div className="flex justify-between items-center text-xs">
                <div>
                  <span className="font-semibold text-gray-800 text-sm">
                    Event Deduplication Cooldown
                  </span>
                  <p className="text-xs text-gray-500 mt-0.5">
                    Interval between audit records per person to prevent logging identical events every frame.
                  </p>
                </div>
                <span className="text-base font-bold font-mono text-blue-600">
                  {settings.event_cooldown_seconds}s
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="60"
                step="1"
                value={settings.event_cooldown_seconds}
                onChange={(e) =>
                  setSettings({ ...settings, event_cooldown_seconds: parseInt(e.target.value, 10) })
                }
                className="w-full accent-blue-600 h-2 bg-gray-200 rounded-lg cursor-pointer"
              />
              <div className="flex justify-between text-xs text-gray-400 font-mono">
                <span>1 Second</span>
                <span>3 Seconds (1 Shot / 3s)</span>
                <span>60 Seconds</span>
              </div>
            </div>

            {/* Threshold 3: Data Retention */}
            <div className="space-y-2 pt-4 border-t border-gray-100">
              <div className="flex justify-between items-center text-xs">
                <div>
                  <span className="font-semibold text-gray-800 text-sm">
                    Audit Log Retention Window
                  </span>
                  <p className="text-xs text-gray-500 mt-0.5">
                    Number of days inspection logs are retained before automatic cleanup.
                  </p>
                </div>
                <span className="text-base font-bold font-mono text-blue-600">
                  {settings.retention_days} Days
                </span>
              </div>
              <input
                type="range"
                min="14"
                max="365"
                step="7"
                value={settings.retention_days}
                onChange={(e) =>
                  setSettings({ ...settings, retention_days: parseInt(e.target.value, 10) })
                }
                className="w-full accent-blue-600 h-2 bg-gray-200 rounded-lg cursor-pointer"
              />
            </div>

            {/* Default Required PPE */}
            <div className="space-y-2 pt-4 border-t border-gray-100">
              <span className="font-semibold text-gray-800 text-sm block">
                Default Required PPE Items (Used for unassigned or unregistered subjects):
              </span>
              <div className="grid grid-cols-2 gap-2.5 text-xs">
                {PPE_OPTIONS.map((opt) => {
                  const checked = settings.default_required_ppe.includes(opt.id);
                  return (
                    <label
                      key={opt.id}
                      className={`flex items-center gap-2 p-2.5 rounded-lg border cursor-pointer transition-colors ${
                        checked
                          ? 'bg-blue-50 border-blue-300 text-blue-900 font-medium'
                          : 'bg-white border-gray-200 text-gray-700 hover:bg-gray-50'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSettings({
                              ...settings,
                              default_required_ppe: [...settings.default_required_ppe, opt.id],
                            });
                          } else {
                            setSettings({
                              ...settings,
                              default_required_ppe: settings.default_required_ppe.filter(
                                (id) => id !== opt.id
                              ),
                            });
                          }
                        }}
                        className="accent-blue-600 rounded"
                      />
                      <span>{opt.label}</span>
                    </label>
                  );
                })}
              </div>
            </div>

            {/* Per-Class Confidence Thresholds */}
            <div className="space-y-3 pt-4 border-t border-gray-100">
              <div>
                <span className="font-semibold text-gray-800 text-sm block">
                  Per-Class Detection Sensitivity Thresholds
                </span>
                <p className="text-xs text-gray-500 mt-0.5">
                  Set baseline acceptance thresholds for each PPE class. Detections exceeding these thresholds are accepted as compliant items.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                {[
                  { id: 'helmet', label: 'Helmet / Hard Hat', def: 0.15 },
                  { id: 'safety_vest', label: 'Safety Vest', def: 0.15 },
                  { id: 'gloves', label: 'Gloves', def: 0.15 },
                  { id: 'safety_shoes', label: 'Safety Shoes', def: 0.15 },
                  { id: 'goggles', label: 'Safety Goggles', def: 0.15 },
                  { id: 'mask', label: 'Face Mask', def: 0.15 },
                  { id: 'ear_protection', label: 'Ear Protection', def: 0.15 },
                  { id: 'person', label: 'Person Body Detector', def: 0.25 },
                ].map((item) => {
                  const val = settings.class_thresholds?.[item.id] ?? item.def;
                  return (
                    <div key={item.id} className="bg-gray-50 p-2.5 rounded-lg border border-gray-200 space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="font-medium text-gray-800">{item.label}</span>
                        <span className="font-mono font-bold text-blue-600">{Math.round(val * 100)}%</span>
                      </div>
                      <input
                        type="range"
                        min="0.01"
                        max="0.80"
                        step="0.01"
                        value={val}
                        onChange={(e) => {
                          const numVal = parseFloat(e.target.value);
                          setSettings({
                            ...settings,
                            class_thresholds: {
                              ...(settings.class_thresholds || {}),
                              [item.id]: numVal,
                            },
                          });
                        }}
                        className="w-full accent-blue-600 h-1.5 bg-gray-200 rounded-lg cursor-pointer"
                      />
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Action Bar & Feedback */}
            {statusMessage && (
              <div
                className={`p-3 rounded-lg text-sm flex items-center gap-2 ${
                  statusMessage.type === 'success'
                    ? 'bg-green-50 border border-green-200 text-green-700'
                    : 'bg-red-50 border border-red-200 text-red-700'
                }`}
              >
                {statusMessage.type === 'success' ? (
                  <CheckCircle2 className="w-4 h-4 shrink-0 text-green-600" />
                ) : (
                  <AlertCircle className="w-4 h-4 shrink-0 text-red-600" />
                )}
                <span>{statusMessage.text}</span>
              </div>
            )}

            <div className="flex justify-end pt-2">
              <button
                type="submit"
                disabled={saving}
                className="flex items-center gap-2 px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors shadow-sm disabled:opacity-50"
              >
                <Save className="w-4 h-4" />
                <span>{saving ? 'Saving...' : 'Save Settings'}</span>
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Privacy, Governance & Architecture Notice */}
        <div className="space-y-6">
          <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm space-y-4 text-xs">
            <div className="flex items-center gap-2 border-b border-gray-100 pb-3">
              <Lock className="w-4 h-4 text-blue-600" />
              <h3 className="text-sm font-bold text-gray-900">
                Privacy & Data Governance
              </h3>
            </div>

            <p className="text-gray-600 leading-relaxed text-xs">
              This system processes all biometric and compliance information locally on this machine.
            </p>

            <div className="space-y-2.5 text-xs text-gray-600">
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                <span className="font-semibold text-gray-900 block mb-1">
                  1. Local On-Device Processing
                </span>
                No face photos, video streams, or biometric data are transmitted over the internet or sent to third-party cloud services.
              </div>

              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                <span className="font-semibold text-gray-900 block mb-1">
                  2. Irreversible Numerical Vectors
                </span>
                Raw face images are not saved to disk. Only 128-dimensional floating-point vectors are stored in PostgreSQL database.
              </div>

              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                <span className="font-semibold text-gray-900 block mb-1">
                  3. Unconditional Consent Withdrawal
                </span>
                Employees may withdraw biometric consent at any time. When withdrawn, the face embedding vector is permanently erased.
              </div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm space-y-3 text-xs">
            <div className="flex items-center gap-2 border-b border-gray-100 pb-2">
              <Database className="w-4 h-4 text-blue-600" />
              <h3 className="text-xs font-bold text-gray-900">
                PostgreSQL Database Info
              </h3>
            </div>
            <div className="space-y-1.5 text-xs text-gray-600">
              <div className="flex justify-between">
                <span>Database:</span>
                <span className="text-gray-900 font-mono">PostgreSQL (PPE_Detect)</span>
              </div>
              <div className="flex justify-between">
                <span>Face Engine:</span>
                <span className="text-blue-700 font-medium">OpenCV SFace (Apache 2.0)</span>
              </div>
              <div className="flex justify-between">
                <span>Detector:</span>
                <span className="text-blue-700 font-medium">OpenCV YuNet (Apache 2.0)</span>
              </div>
            </div>
          </div>
        </div>
      </form>
    </div>
  );
};
