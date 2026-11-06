import React, { useState, useEffect } from 'react';
import {
  ShieldCheck, AlertTriangle, Users, Activity, ArrowRight,
  Download, Eye, RefreshCw, FileText
} from 'lucide-react';
import { ComplianceStats, InspectionEvent } from '../types';
import { fetchStats, fetchEvents, getExportEventsCsvUrl } from '../services/api';
import { TabType } from '../components/Header';

interface DashboardViewProps {
  onNavigate: (tab: TabType) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({ onNavigate }) => {
  const [stats, setStats] = useState<ComplianceStats | null>(null);
  const [recentEvents, setRecentEvents] = useState<InspectionEvent[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const [s, evts] = await Promise.all([
        fetchStats().catch(() => null),
        fetchEvents({ limit: 8 }).catch(() => []),
      ]);
      if (s) setStats(s);
      if (evts) setRecentEvents(evts);
    } catch (err) {
      console.error('Failed to load dashboard metrics:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();
    const interval = setInterval(loadDashboardData, 12000);
    return () => clearInterval(interval);
  }, []);

  const complianceRate = stats ? Math.round(stats.compliance_rate) : 0;

  return (
    <div className="flex flex-col gap-6 w-full select-none font-sans">
      {/* Top Banner */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-white border border-gray-200 p-6 rounded-xl shadow-sm">
        <div>
          <h2 className="text-xl font-bold text-gray-900 tracking-tight flex items-center gap-2.5">
            <Activity className="w-5 h-5 text-blue-600" />
            <span>Compliance Dashboard</span>
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            Real-time overview of workplace safety compliance and biometric verification.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={loadDashboardData}
            disabled={loading}
            className="flex items-center gap-2 px-3.5 py-2 bg-white hover:bg-gray-50 text-gray-700 rounded-lg text-sm font-medium border border-gray-300 transition-colors shadow-sm disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>

          <a
            href={getExportEventsCsvUrl()}
            download="ppe_compliance_audit.csv"
            className="flex items-center gap-2 px-4 py-2 bg-white hover:bg-gray-50 text-gray-700 rounded-lg text-sm font-medium border border-gray-300 transition-colors shadow-sm"
          >
            <Download className="w-4 h-4 text-gray-500" />
            <span>Export CSV</span>
          </a>

          <button
            type="button"
            onClick={() => onNavigate('live')}
            className="flex items-center gap-2 px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium shadow-sm transition-colors"
          >
            <Eye className="w-4 h-4" />
            <span>Open Live Camera</span>
          </button>
        </div>
      </div>

      {/* KPI Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* KPI 1: Overall Compliance Rate */}
        <div className="bg-white border border-gray-200 p-5 rounded-xl shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between text-gray-500 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Overall Compliance</span>
            <div className="p-2 bg-green-50 text-green-700 rounded-lg border border-green-200">
              <ShieldCheck className="w-4 h-4" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-extrabold font-mono text-green-600">
              {stats ? `${complianceRate}%` : '--'}
            </span>
            <span className="text-xs text-gray-500">compliant rate</span>
          </div>
          <div className="w-full bg-gray-100 h-2 rounded-full overflow-hidden mt-3">
            <div
              className="bg-green-600 h-full rounded-full transition-all duration-500"
              style={{ width: `${complianceRate}%` }}
            />
          </div>
        </div>

        {/* KPI 2: Total Scans */}
        <div className="bg-white border border-gray-200 p-5 rounded-xl shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between text-gray-500 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Total Inspections</span>
            <div className="p-2 bg-blue-50 text-blue-700 rounded-lg border border-blue-200">
              <Activity className="w-4 h-4" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-extrabold font-mono text-gray-900">
              {stats?.total_scans ?? 0}
            </span>
            <span className="text-xs text-gray-500">frames processed</span>
          </div>
          <p className="text-xs text-gray-500 mt-2 font-mono">
            {stats?.compliant_scans ?? 0} full compliance passes
          </p>
        </div>

        {/* KPI 3: Violations Logged */}
        <div className="bg-white border border-gray-200 p-5 rounded-xl shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between text-gray-500 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Safety Violations</span>
            <div className="p-2 bg-red-50 text-red-700 rounded-lg border border-red-200">
              <AlertTriangle className="w-4 h-4" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-extrabold font-mono text-red-600">
              {stats?.violations ?? 0}
            </span>
            <span className="text-xs text-gray-500">incidents</span>
          </div>
          <p className="text-xs text-red-600 mt-2 font-medium">
            Missing required PPE items
          </p>
        </div>

        {/* KPI 4: Active Enrolled Employees */}
        <div className="bg-white border border-gray-200 p-5 rounded-xl shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between text-gray-500 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Enrolled Personnel</span>
            <div className="p-2 bg-blue-50 text-blue-700 rounded-lg border border-blue-200">
              <Users className="w-4 h-4" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-extrabold font-mono text-blue-600">
              {stats?.enrolled_faces ?? 0}
            </span>
            <span className="text-xs text-gray-500">
              of {stats?.active_employees ?? 0} active
            </span>
          </div>
          <p className="text-xs text-gray-500 mt-2 font-mono">
            Face biometric templates
          </p>
        </div>
      </div>

      {/* Recent Inspection Events Table */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex items-center justify-between border-b border-gray-100 pb-3">
          <div>
            <h3 className="text-base font-bold text-gray-900 flex items-center gap-2">
              <FileText className="w-4 h-4 text-blue-600" />
              <span>Recent Inspection Events</span>
            </h3>
            <p className="text-xs text-gray-500 mt-0.5">
              Latest automatic safety inspection and personnel identification logs
            </p>
          </div>
          <button
            type="button"
            onClick={() => onNavigate('events')}
            className="flex items-center gap-1.5 text-sm font-semibold text-blue-600 hover:text-blue-700 transition-colors"
          >
            <span>View Full Audit Log</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm border-collapse">
            <thead>
              <tr className="border-b border-gray-200 bg-gray-50 text-gray-600 uppercase text-xs font-semibold">
                <th className="py-3 px-3">Event ID</th>
                <th className="py-3 px-3">Timestamp</th>
                <th className="py-3 px-3">Employee</th>
                <th className="py-3 px-3">Identity Status</th>
                <th className="py-3 px-3">Result</th>
                <th className="py-3 px-3">Missing Items</th>
                <th className="py-3 px-3 text-right">Match</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {recentEvents.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-gray-400 italic">
                    No inspection events recorded yet. Start live detection to begin monitoring.
                  </td>
                </tr>
              ) : (
                recentEvents.map((evt) => {
                  const isCompliant = evt.overall_result === 'COMPLIANT';
                  return (
                    <tr key={evt.id} className="hover:bg-gray-50 transition-colors">
                      <td className="py-3 px-3 text-gray-500 font-mono text-xs">{evt.id}</td>
                      <td className="py-3 px-3 text-gray-700 font-mono text-xs">{evt.timestamp}</td>
                      <td className="py-3 px-3 font-semibold text-gray-900">
                        {evt.employee_name || 'Unregistered'}
                      </td>
                      <td className="py-3 px-3">
                        <span className="text-xs px-2 py-0.5 rounded-md bg-gray-100 text-gray-700 border border-gray-200 font-medium">
                          {evt.identity_state}
                        </span>
                      </td>
                      <td className="py-3 px-3">
                        <span
                          className={`text-xs px-2.5 py-0.5 rounded-full font-bold uppercase ${
                            isCompliant
                              ? 'bg-green-100 text-green-800 border border-green-200'
                              : 'bg-red-100 text-red-800 border border-red-200'
                          }`}
                        >
                          {evt.overall_result}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-red-600 text-xs font-medium">
                        {evt.missing_ppe && evt.missing_ppe.length > 0
                          ? evt.missing_ppe.join(', ')
                          : 'None'}
                      </td>
                      <td className="py-3 px-3 text-right text-blue-600 font-bold font-mono">
                        {Math.round(evt.match_score * 100)}%
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
