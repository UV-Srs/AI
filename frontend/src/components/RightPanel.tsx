import React, { useState } from 'react';
import {
  Sliders, Cpu, Activity, Zap, CheckCircle2,
  UserCheck, Check, X, RotateCcw, Shield, Settings as SettingsIcon
} from 'lucide-react';
import { ModelInfo, IntegratedDetectionResponse } from '../types';

interface PPEClassMeta {
  key: string;
  label: string;
  defaultVal: number;
}

const PPE_CLASS_CONFIG: PPEClassMeta[] = [
  { key: 'helmet', label: 'Hard Hat / Helmet', defaultVal: 0.15 },
  { key: 'safety_vest', label: 'Safety Vest', defaultVal: 0.15 },
  { key: 'gloves', label: 'Safety Gloves', defaultVal: 0.15 },
  { key: 'safety_shoes', label: 'Safety Shoes / Boots', defaultVal: 0.15 },
  { key: 'goggles', label: 'Safety Goggles', defaultVal: 0.15 },
  { key: 'mask', label: 'Face Mask', defaultVal: 0.15 },
  { key: 'ear_protection', label: 'Ear Protection', defaultVal: 0.15 },
  { key: 'person', label: 'Person Body Detector', defaultVal: 0.25 },
];

interface RightPanelProps {
  models: ModelInfo[];
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
  isModelSwitching: boolean;
  modelSwitchMessage: string | null;
  confThreshold: number;
  onConfThresholdChange: (val: number) => void;
  classThresholds: Record<string, number>;
  onClassThresholdChange: (className: string, val: number) => void;
  onResetClassThresholds: () => void;
  targetFps: number;
  onTargetFpsChange: (fps: number) => void;
  lastDetection: IntegratedDetectionResponse | null;
  showBoxes: boolean;
  onToggleShowBoxes: () => void;
  showLabels: boolean;
  onToggleShowLabels: () => void;
}

export const RightPanel: React.FC<RightPanelProps> = ({
  models,
  selectedModelId,
  onSelectModel,
  isModelSwitching,
  modelSwitchMessage,
  confThreshold,
  onConfThresholdChange,
  classThresholds,
  onClassThresholdChange,
  onResetClassThresholds,
  targetFps,
  onTargetFpsChange,
  lastDetection,
  showBoxes,
  onToggleShowBoxes,
  showLabels,
  onToggleShowLabels,
}) => {
  const [activeTab, setActiveTab] = useState<'sensitivity' | 'settings'>('sensitivity');

  return (
    <aside className="w-full lg:w-96 flex flex-col gap-3 select-none">
      {/* 1. Live Detection Status HUD */}
      <div className="bg-white border border-gray-200 rounded-xl p-3.5 shadow-xs">
        <div className="flex items-center justify-between mb-2 pb-2 border-b border-gray-100">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <h2 className="text-xs font-bold uppercase tracking-wider text-gray-700">Live Status</h2>
          </div>
          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="text-blue-600 font-bold">{lastDetection?.fps || 0.0} FPS</span>
            <span className="text-gray-300">|</span>
            <span className="text-gray-600 font-medium">{lastDetection?.inference_time_ms || 0} ms</span>
          </div>
        </div>

        {/* Compact Counters */}
        <div className="grid grid-cols-3 gap-2 text-center text-xs">
          <div className="bg-gray-50 py-1.5 px-2 rounded-lg border border-gray-100">
            <span className="text-gray-400 block text-[10px] font-medium">People</span>
            <span className="text-base font-bold font-mono text-gray-900">
              {lastDetection?.total_persons || 0}
            </span>
          </div>
          <div className="bg-emerald-50 py-1.5 px-2 rounded-lg border border-emerald-200">
            <span className="text-emerald-700 block text-[10px] font-medium">Compliant</span>
            <span className="text-base font-bold font-mono text-emerald-700">
              {lastDetection?.compliant_count || 0}
            </span>
          </div>
          <div className="bg-red-50 py-1.5 px-2 rounded-lg border border-red-200">
            <span className="text-red-700 block text-[10px] font-medium">Violations</span>
            <span className="text-base font-bold font-mono text-red-700">
              {lastDetection?.violation_count || 0}
            </span>
          </div>
        </div>
      </div>

      {/* 2. Detected Personnel & PPE Checklist */}
      <div className="bg-white border border-gray-200 rounded-xl p-3.5 shadow-xs">
        <div className="flex items-center justify-between mb-2 pb-1.5 border-b border-gray-100">
          <h3 className="text-xs font-bold text-gray-800 flex items-center gap-1.5">
            <UserCheck className="w-3.5 h-3.5 text-blue-600" />
            <span>Active Personnel</span>
          </h3>
          <span className="text-[10px] font-semibold px-2 py-0.5 bg-gray-100 text-gray-700 rounded-full">
            {lastDetection?.persons?.length || 0} In View
          </span>
        </div>

        <div className="max-h-44 overflow-y-auto space-y-2 pr-0.5">
          {!lastDetection?.persons || lastDetection.persons.length === 0 ? (
            <div className="py-4 text-center text-xs text-gray-400 italic">
              No persons in view. Step in front of the camera.
            </div>
          ) : (
            lastDetection.persons.map((person, idx) => {
              const isCompliant = person.overall_result === 'COMPLIANT';
              const isViolation = person.overall_result === 'NON_COMPLIANT';

              return (
                <div
                  key={idx}
                  className={`p-2.5 rounded-lg border text-xs ${
                    isCompliant
                      ? 'bg-emerald-50/50 border-emerald-200'
                      : isViolation
                      ? 'bg-red-50/50 border-red-200'
                      : 'bg-amber-50/50 border-amber-200'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold text-gray-900 truncate max-w-[180px]">
                      {person.face ? person.face.employee_name : (person.person_label || `Person ${idx + 1}`)}
                    </span>
                    <span
                      className={`text-[9px] font-bold px-1.5 py-0.5 rounded uppercase ${
                        isCompliant
                          ? 'bg-emerald-100 text-emerald-800'
                          : isViolation
                          ? 'bg-red-100 text-red-800'
                          : 'bg-amber-100 text-amber-800'
                      }`}
                    >
                      {person.overall_result}
                    </span>
                  </div>

                  {person.face && (
                    <div className="text-[10px] text-gray-500 font-mono flex items-center justify-between mb-1.5">
                      <span>ID: {person.face.employee_id || 'N/A'}</span>
                      <span className="text-blue-700 font-semibold">
                        Match: {Math.round(person.face.match_score * 100)}%
                      </span>
                    </div>
                  )}

                  {/* Checklist pills */}
                  <div className="flex flex-wrap gap-1 pt-1 border-t border-gray-200/60">
                    {Object.entries(person.checklist).map(([gear, state]) => {
                      const detected = state === 'DETECTED';
                      return (
                        <span
                          key={gear}
                          className={`text-[9px] px-1.5 py-0.5 rounded flex items-center gap-0.5 font-medium ${
                            detected
                              ? 'bg-emerald-100 text-emerald-800'
                              : 'bg-red-100 text-red-800'
                          }`}
                        >
                          {detected ? <Check className="w-2.5 h-2.5 text-emerald-700" /> : <X className="w-2.5 h-2.5 text-red-700" />}
                          <span>{gear.replace('_', ' ')}</span>
                        </span>
                      );
                    })}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* 3. Controls (Tabbed: Sensitivity vs Settings) */}
      <div className="bg-white border border-gray-200 rounded-xl shadow-xs overflow-hidden flex flex-col">
        {/* Tab Headers */}
        <div className="grid grid-cols-2 bg-gray-50 border-b border-gray-200 text-xs">
          <button
            type="button"
            onClick={() => setActiveTab('sensitivity')}
            className={`py-2 px-3 font-semibold flex items-center justify-center gap-1.5 transition-colors ${
              activeTab === 'sensitivity'
                ? 'bg-white text-blue-600 border-b-2 border-blue-600 shadow-2xs'
                : 'text-gray-500 hover:text-gray-800'
            }`}
          >
            <Shield className="w-3.5 h-3.5" />
            <span>Gear Sensitivity</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('settings')}
            className={`py-2 px-3 font-semibold flex items-center justify-center gap-1.5 transition-colors ${
              activeTab === 'settings'
                ? 'bg-white text-blue-600 border-b-2 border-blue-600 shadow-2xs'
                : 'text-gray-500 hover:text-gray-800'
            }`}
          >
            <SettingsIcon className="w-3.5 h-3.5" />
            <span>Camera & Model</span>
          </button>
        </div>

        {/* Tab 1: Class Sensitivity Thresholds */}
        {activeTab === 'sensitivity' && (
          <div className="p-3.5 space-y-2.5">
            <div className="flex items-center justify-between pb-1.5 border-b border-gray-100">
              <span className="text-[11px] text-gray-500">
                Adjust confidence for hard hats & gear:
              </span>
              <button
                type="button"
                onClick={onResetClassThresholds}
                title="Reset all thresholds to defaults"
                className="flex items-center gap-1 px-2 py-0.5 text-[10px] font-semibold text-gray-600 hover:text-blue-700 hover:bg-blue-50 border border-gray-200 rounded transition-colors"
              >
                <RotateCcw className="w-2.5 h-2.5" />
                <span>Reset</span>
              </button>
            </div>

            <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
              {PPE_CLASS_CONFIG.map((item) => {
                const currentVal = classThresholds[item.key] ?? item.defaultVal;
                const peakScore = lastDetection?.peak_scores?.[item.key] ?? null;
                const isDetected = peakScore !== null && peakScore >= currentVal;

                return (
                  <div key={item.key} className="bg-gray-50/80 p-2 rounded-lg border border-gray-150 space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-gray-800 text-[11px]">{item.label}</span>
                      <span className="font-mono font-bold text-blue-600 text-[11px] px-1.5 py-0.2 bg-blue-50 rounded border border-blue-200">
                        {Math.round(currentVal * 100)}%
                      </span>
                    </div>

                    <input
                      type="range"
                      min="0.01"
                      max="0.80"
                      step="0.01"
                      value={currentVal}
                      onChange={(e) => onClassThresholdChange(item.key, parseFloat(e.target.value))}
                      className="w-full accent-blue-600 h-1.5 bg-gray-200 rounded-lg cursor-pointer"
                    />

                    <div className="flex items-center justify-between text-[10px]">
                      <span className="text-gray-400 text-[9px]">Threshold</span>
                      {peakScore !== null && peakScore > 0 ? (
                        isDetected ? (
                          <span className="px-1.5 py-0.2 rounded bg-emerald-100 text-emerald-800 font-mono font-bold flex items-center gap-0.5">
                            <Check className="w-2.5 h-2.5 text-emerald-700" />
                            <span>Seen: {Math.round(peakScore * 100)}% (DETECTED)</span>
                          </span>
                        ) : (
                          <span className="px-1.5 py-0.2 rounded bg-amber-50 text-amber-800 font-mono font-medium border border-amber-200" title="Slide lower to detect">
                            <span>Seen: {Math.round(peakScore * 100)}% &lt; {Math.round(currentVal * 100)}%</span>
                          </span>
                        )
                      ) : (
                        <span className="text-gray-400 font-mono text-[9px]">Not in view</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Tab 2: Camera & Model Settings */}
        {activeTab === 'settings' && (
          <div className="p-3.5 space-y-3.5">
            {/* Model Selection Dropdown */}
            <div className="space-y-1">
              <label className="text-xs font-semibold text-gray-700 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-blue-600" />
                <span>Active Model</span>
              </label>
              <select
                value={selectedModelId}
                onChange={(e) => onSelectModel(e.target.value)}
                disabled={isModelSwitching}
                className="w-full bg-gray-50 border border-gray-300 rounded-lg p-2 text-xs font-medium text-gray-800 focus:outline-blue-500"
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} ({m.size_mb} MB)
                  </option>
                ))}
              </select>
              {modelSwitchMessage && (
                <div className="p-1.5 text-[11px] rounded bg-blue-50 border border-blue-200 text-blue-800 font-medium text-center">
                  {modelSwitchMessage}
                </div>
              )}
            </div>

            {/* Target FPS Slider */}
            <div className="space-y-1 pt-2 border-t border-gray-100">
              <div className="flex justify-between text-xs">
                <span className="text-gray-600 font-medium">Capture Rate</span>
                <span className="text-blue-600 font-bold font-mono">{targetFps} FPS</span>
              </div>
              <input
                type="range"
                min="1"
                max="15"
                step="1"
                value={targetFps}
                onChange={(e) => onTargetFpsChange(parseInt(e.target.value, 10))}
                className="w-full accent-blue-600 h-1.5 bg-gray-200 rounded-lg cursor-pointer"
              />
            </div>

            {/* Global Fallback Threshold */}
            <div className="space-y-1 pt-2 border-t border-gray-100">
              <div className="flex justify-between text-xs">
                <span className="text-gray-600 font-medium">Global Fallback Sensitivity</span>
                <span className="text-blue-600 font-bold font-mono">{Math.round(confThreshold * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="0.80"
                step="0.01"
                value={confThreshold}
                onChange={(e) => onConfThresholdChange(parseFloat(e.target.value))}
                className="w-full accent-blue-600 h-1.5 bg-gray-200 rounded-lg cursor-pointer"
              />
            </div>

            {/* Display Checkboxes */}
            <div className="grid grid-cols-2 gap-2 pt-2 border-t border-gray-100 text-xs">
              <label className="flex items-center gap-2 text-gray-700 cursor-pointer font-medium">
                <input
                  type="checkbox"
                  checked={showBoxes}
                  onChange={onToggleShowBoxes}
                  className="accent-blue-600 rounded"
                />
                <span>Show Boxes</span>
              </label>
              <label className="flex items-center gap-2 text-gray-700 cursor-pointer font-medium">
                <input
                  type="checkbox"
                  checked={showLabels}
                  onChange={onToggleShowLabels}
                  className="accent-blue-600 rounded"
                />
                <span>Show Labels</span>
              </label>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};
