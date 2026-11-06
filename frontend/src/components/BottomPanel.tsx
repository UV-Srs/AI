import React from 'react';
import { ListFilter, Info, Trash2 } from 'lucide-react';
import { DetectionEvent } from '../types';
import { PPE_CLASSES } from '../services/classes';

interface BottomPanelProps {
  events: DetectionEvent[];
  onClearEvents: () => void;
  activeModelName: string;
}

export const BottomPanel: React.FC<BottomPanelProps> = ({
  events,
  onClearEvents,
  activeModelName,
}) => {
  return (
    <footer className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm select-none">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Latest Detection Events */}
        <div className="lg:col-span-2 space-y-3">
          <div className="flex items-center justify-between border-b border-gray-100 pb-2.5">
            <div className="flex items-center gap-2">
              <ListFilter className="w-4 h-4 text-blue-600" />
              <h3 className="text-sm font-bold text-gray-900">
                Live Activity Log
              </h3>
              <span className="text-[10px] font-semibold px-2 py-0.5 bg-blue-50 text-blue-700 border border-blue-200 rounded-md">
                Captured 3s / shot
              </span>
              <span className="text-xs font-semibold px-2 py-0.5 bg-gray-100 text-gray-600 rounded-md">
                {events.length} shots
              </span>
            </div>
            {events.length > 0 && (
              <button
                type="button"
                onClick={onClearEvents}
                className="text-xs text-gray-600 hover:text-red-600 flex items-center gap-1.5 transition-colors px-2.5 py-1 rounded-md bg-white border border-gray-200 shadow-xs"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Clear Stream</span>
              </button>
            )}
          </div>

          <div className="h-28 overflow-y-auto pr-1 space-y-2">
            {events.length === 0 ? (
              <div className="h-full flex items-center justify-center text-xs text-gray-400 italic">
                No events in current session. Start detection to view real-time feed updates.
              </div>
            ) : (
              events.map((evt) => {
                const cfg = PPE_CLASSES[evt.className];
                const isViolation = evt.complianceStatus === 'NON_COMPLIANT';
                return (
                  <div
                    key={evt.id}
                    className="flex items-center justify-between px-3 py-2 rounded-lg bg-gray-50 border border-gray-100 text-xs"
                  >
                    <div className="flex items-center gap-2.5">
                      <span className="text-xs text-gray-400 font-mono">{evt.timestamp}</span>
                      <span
                        className="w-2.5 h-2.5 rounded-full inline-block"
                        style={{ backgroundColor: cfg?.color || (isViolation ? '#dc2626' : '#16a34a') }}
                      />
                      <span className="font-semibold text-gray-800">
                        {evt.employeeName ? `${evt.employeeName} — ` : ''}
                        {cfg?.label.split('/')[0].trim() || evt.className}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      {evt.complianceStatus && (
                        <span
                          className={`text-[10px] px-2 py-0.5 rounded-md font-bold uppercase ${
                            evt.complianceStatus === 'COMPLIANT'
                              ? 'bg-green-100 text-green-800'
                              : 'bg-red-100 text-red-800'
                          }`}
                        >
                          {evt.complianceStatus}
                        </span>
                      )}
                      <span className="text-xs font-semibold text-blue-700 font-mono">
                        {Math.round(evt.confidence * 100)}%
                      </span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Right: Technical Specification */}
        <div className="bg-gray-50 border border-gray-200/80 rounded-lg p-3.5 space-y-2 text-xs">
          <div className="flex items-center gap-2 text-gray-900 font-bold border-b border-gray-200 pb-2">
            <Info className="w-4 h-4 text-blue-600" />
            <span>System Details</span>
          </div>

          <div className="space-y-1.5 text-xs text-gray-600">
            <div className="flex justify-between">
              <span className="text-gray-500">PPE Detection:</span>
              <span className="text-gray-900 font-medium">{activeModelName}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Face Engine:</span>
              <span className="text-blue-700 font-medium">OpenCV YuNet + SFace (Apache 2.0)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Inference Device:</span>
              <span className="text-gray-900 font-medium">Local CPU (PyTorch)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Database:</span>
              <span className="text-gray-900 font-medium">PostgreSQL (PPE_Detect)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Privacy:</span>
              <span className="text-green-700 font-semibold">100% On-Device / Zero Cloud</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};
