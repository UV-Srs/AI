import React, { useState, useEffect } from 'react';
import {
  FileText, Download, Search, RefreshCw, X
} from 'lucide-react';
import { InspectionEvent } from '../types';
import { fetchEvents, getExportEventsCsvUrl } from '../services/api';

export const EventsView: React.FC = () => {
  const [events, setEvents] = useState<InspectionEvent[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [resultFilter, setResultFilter] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedEvent, setSelectedEvent] = useState<InspectionEvent | null>(null);

  const loadEvents = async () => {
    try {
      setLoading(true);
      const params: any = { limit: 200 };
      if (resultFilter !== 'ALL') {
        params.overall_result = resultFilter;
      }
      const data = await fetchEvents(params);
      setEvents(data);
    } catch (err: any) {
      console.error('Failed to load events:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadEvents();
  }, [resultFilter]);

  const filtered = events.filter((e) => {
    const q = searchQuery.toLowerCase();
    const empName = (e.employee_name || '').toLowerCase();
    const empId = (e.employee_id || '').toLowerCase();
    const evtId = (e.id || '').toLowerCase();
    return empName.includes(q) || empId.includes(q) || evtId.includes(q);
  });

  return (
    <div className="flex flex-col gap-6 w-full select-none font-sans">
      {/* Top Header & Export Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-white border border-gray-200 p-6 rounded-xl shadow-sm">
        <div>
          <h2 className="text-xl font-bold text-gray-900 tracking-tight flex items-center gap-2.5">
            <FileText className="w-5 h-5 text-blue-600" />
            <span>Inspection Audit Logs</span>
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            Complete records of PPE compliance scans and facial verification events.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Result Filter Tabs */}
          <div className="flex items-center bg-gray-100 p-1 rounded-lg border border-gray-200 text-xs">
            {['ALL', 'COMPLIANT', 'NON_COMPLIANT', 'NEEDS_REVIEW'].map((tab) => (
              <button
                key={tab}
                type="button"
                onClick={() => setResultFilter(tab)}
                className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                  resultFilter === tab
                    ? 'bg-blue-600 text-white shadow-xs'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                {tab === 'ALL' ? 'All' : tab === 'NON_COMPLIANT' ? 'Violations' : tab === 'COMPLIANT' ? 'Compliant' : 'Review'}
              </button>
            ))}
          </div>

          {/* Search Field */}
          <div className="relative">
            <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search by name or ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 pr-4 py-2 bg-white border border-gray-300 rounded-lg text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 w-56"
            />
          </div>

          <button
            type="button"
            onClick={loadEvents}
            disabled={loading}
            className="p-2 bg-white hover:bg-gray-50 text-gray-600 rounded-lg border border-gray-300 transition-colors shadow-sm"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>

          <a
            href={getExportEventsCsvUrl()}
            download="ppe_inspection_audit_log.csv"
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors shadow-sm"
          >
            <Download className="w-4 h-4" />
            <span>Export CSV</span>
          </a>
        </div>
      </div>

      {/* Events Table */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm space-y-4">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm border-collapse">
            <thead>
              <tr className="border-b border-gray-200 bg-gray-50 text-gray-600 uppercase text-xs font-semibold">
                <th className="py-3 px-3">Event ID</th>
                <th className="py-3 px-3">Timestamp</th>
                <th className="py-3 px-3">Camera</th>
                <th className="py-3 px-3">Employee</th>
                <th className="py-3 px-3">Identity Status</th>
                <th className="py-3 px-3">Compliance Result</th>
                <th className="py-3 px-3">Missing Items</th>
                <th className="py-3 px-3 text-right">Match</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-8 text-center text-gray-400 italic">
                    {loading ? 'Loading events...' : 'No inspection records found.'}
                  </td>
                </tr>
              ) : (
                filtered.map((evt) => {
                  const isCompliant = evt.overall_result === 'COMPLIANT';
                  return (
                    <tr
                      key={evt.id}
                      onClick={() => setSelectedEvent(evt)}
                      className="hover:bg-gray-50 transition-colors cursor-pointer"
                    >
                      <td className="py-3 px-3 text-gray-500 font-mono text-xs">{evt.id}</td>
                      <td className="py-3 px-3 text-gray-700 font-mono text-xs">{evt.timestamp}</td>
                      <td className="py-3 px-3 text-gray-500 text-xs">{evt.camera_id}</td>
                      <td className="py-3 px-3 font-semibold text-gray-900">
                        {evt.employee_name || 'Unregistered'}
                        {evt.employee_id ? ` (${evt.employee_id})` : ''}
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

      {/* Event Detail Modal */}
      {selectedEvent && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white border border-gray-200 rounded-xl max-w-md w-full p-6 shadow-2xl animate-fade-in text-gray-900 space-y-4">
            <div className="flex items-center justify-between border-b border-gray-100 pb-3">
              <div>
                <h3 className="text-base font-bold text-gray-900">
                  Inspection Record
                </h3>
                <p className="text-xs text-gray-500 font-mono">{selectedEvent.id}</p>
              </div>
              <button
                type="button"
                onClick={() => setSelectedEvent(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3 text-sm">
              <div className="flex justify-between py-1.5 border-b border-gray-100">
                <span className="text-gray-500">Timestamp:</span>
                <span className="font-mono text-gray-900">{selectedEvent.timestamp}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-gray-100">
                <span className="text-gray-500">Camera:</span>
                <span className="font-mono text-gray-900">{selectedEvent.camera_id}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-gray-100">
                <span className="text-gray-500">Identified Employee:</span>
                <span className="font-semibold text-gray-900">
                  {selectedEvent.employee_name || 'Unregistered Person'}
                </span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-gray-100">
                <span className="text-gray-500">Identity State:</span>
                <span className="text-gray-800 font-medium">{selectedEvent.identity_state}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-gray-100">
                <span className="text-gray-500">Biometric Match:</span>
                <span className="font-mono text-blue-600 font-bold">
                  {Math.round(selectedEvent.match_score * 100)}%
                </span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-gray-100">
                <span className="text-gray-500">Overall Result:</span>
                <span
                  className={`font-bold uppercase ${
                    selectedEvent.overall_result === 'COMPLIANT'
                      ? 'text-green-600'
                      : 'text-red-600'
                  }`}
                >
                  {selectedEvent.overall_result}
                </span>
              </div>

              <div className="pt-2">
                <div className="text-gray-700 font-medium text-xs mb-2">PPE Item Checklist:</div>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(selectedEvent.ppe_summary || {}).map(([gear, st]) => {
                    const isDetected = st === 'DETECTED';
                    return (
                      <div
                        key={gear}
                        className={`p-2 rounded-lg border flex items-center justify-between text-xs ${
                          isDetected
                            ? 'bg-green-50 border-green-200 text-green-800'
                            : 'bg-red-50 border-red-200 text-red-800'
                        }`}
                      >
                        <span className="capitalize">{gear.replace('_', ' ')}</span>
                        <span className="font-semibold">{st}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="flex justify-end pt-3 border-t border-gray-100">
              <button
                type="button"
                onClick={() => setSelectedEvent(null)}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors shadow-sm"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
