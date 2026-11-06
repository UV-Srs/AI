import React, { useState, useEffect } from 'react';
import {
  Shield, Cpu, Video, VideoOff, Wifi, WifiOff, Clock,
  LayoutDashboard, Eye, Users, FileText, Settings, Scale, Brain
} from 'lucide-react';
import { HealthStatus, ModelInfo } from '../types';

export type TabType = 'dashboard' | 'live' | 'training' | 'employees' | 'testing' | 'events' | 'settings';

interface HeaderProps {
  health: HealthStatus | null;
  activeModel: ModelInfo | null;
  isCameraActive: boolean;
  isDetecting: boolean;
  wsConnected: boolean;
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  activeModel,
  isCameraActive,
  isDetecting,
  wsConnected,
  activeTab,
  onTabChange,
}) => {
  const [time, setTime] = useState<string>('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTime(now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  const navItems: { id: TabType; label: string; icon: React.ReactNode }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard className="w-4 h-4" /> },
    { id: 'live', label: 'Live Monitoring', icon: <Eye className="w-4 h-4" /> },
    { id: 'training', label: 'Model Training & Data', icon: <Brain className="w-4 h-4" /> },
    { id: 'employees', label: 'Employees', icon: <Users className="w-4 h-4" /> },
    { id: 'testing', label: 'Recognition Testing', icon: <Scale className="w-4 h-4" /> },
    { id: 'events', label: 'Audit Logs', icon: <FileText className="w-4 h-4" /> },
    { id: 'settings', label: 'Settings', icon: <Settings className="w-4 h-4" /> },
  ];

  return (
    <header className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-30 select-none">
      {/* Top Application Bar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex flex-wrap items-center justify-between gap-4">
        {/* Left: Branding */}
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-blue-600 text-white rounded-xl shadow-sm">
            <Shield className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base sm:text-lg font-bold text-gray-900 tracking-tight">
                PPE & Face Recognition System
              </h1>
              <span className="text-xs font-semibold px-2 py-0.5 bg-blue-50 text-blue-700 border border-blue-200 rounded-full">
                YOLO26n + SFace
              </span>
            </div>
            <p className="text-xs text-gray-500 font-normal">
              Industrial Safety Compliance & Personnel Verification
            </p>
          </div>
        </div>

        {/* Right: Status Indicators */}
        <div className="flex items-center gap-2.5 text-xs">
          {/* Active Model */}
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-gray-50 border border-gray-200 rounded-lg text-gray-700">
            <Cpu className="w-3.5 h-3.5 text-gray-500" />
            <span className="text-gray-500 font-medium">Model:</span>
            <span className="font-semibold text-blue-700 truncate max-w-[130px]">
              {activeModel ? activeModel.name : health?.active_model || 'YOLO26n'}
            </span>
            <span className="text-[11px] px-1.5 py-0.5 bg-gray-200 text-gray-700 rounded font-mono font-medium">
              {health?.device || 'CPU'}
            </span>
          </div>

          {/* Camera Status */}
          <div
            className={`flex items-center gap-1.5 px-3 py-1.5 border rounded-lg font-medium transition-colors ${
              isCameraActive
                ? isDetecting
                  ? 'bg-green-50 border-green-200 text-green-700'
                  : 'bg-blue-50 border-blue-200 text-blue-700'
                : 'bg-gray-50 border-gray-200 text-gray-500'
            }`}
          >
            {isCameraActive ? (
              <Video className="w-3.5 h-3.5 animate-pulse text-green-600" />
            ) : (
              <VideoOff className="w-3.5 h-3.5 text-gray-400" />
            )}
            <span>
              {isCameraActive ? (isDetecting ? 'Detecting' : 'Camera On') : 'Camera Off'}
            </span>
          </div>

          {/* Connection Status */}
          <div
            className={`flex items-center gap-1.5 px-3 py-1.5 border rounded-lg font-medium ${
              wsConnected
                ? 'bg-green-50 border-green-200 text-green-700'
                : 'bg-red-50 border-red-200 text-red-700'
            }`}
          >
            {wsConnected ? (
              <>
                <Wifi className="w-3.5 h-3.5 text-green-600" />
                <span>Online</span>
              </>
            ) : (
              <>
                <WifiOff className="w-3.5 h-3.5 text-red-600" />
                <span>Offline</span>
              </>
            )}
          </div>

          {/* Clock */}
          <div className="hidden lg:flex items-center gap-1.5 px-3 py-1.5 bg-gray-50 border border-gray-200 text-gray-600 rounded-lg font-mono">
            <Clock className="w-3.5 h-3.5 text-gray-400" />
            <span>{time}</span>
          </div>
        </div>
      </div>

      {/* Navigation Tab Bar */}
      <div className="border-t border-gray-200 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center gap-1 overflow-x-auto">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onTabChange(item.id)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors border-b-2 whitespace-nowrap ${
                  isActive
                    ? 'text-blue-600 border-blue-600 bg-blue-50/50'
                    : 'text-gray-600 border-transparent hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                {item.icon}
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
