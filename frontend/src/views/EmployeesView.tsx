import React, { useState, useEffect, useRef } from 'react';
import {
  Users, UserPlus, Search, ShieldCheck, ShieldAlert, Trash2,
  Camera, Check, X, AlertCircle, RefreshCw, EyeOff, CheckCircle2,
  Scan, Info, ChevronRight
} from 'lucide-react';
import { Employee, BiometricDetails } from '../types';
import {
  fetchEmployees, saveEmployee, deleteEmployee,
  toggleEmployeeStatus, withdrawConsent, enrollAngleCapture, fetchBiometrics
} from '../services/api';

const PPE_OPTIONS = [
  { id: 'helmet', label: 'Helmet / Hard Hat' },
  { id: 'safety_vest', label: 'Safety Vest' },
  { id: 'goggles', label: 'Safety Goggles / Glasses' },
  { id: 'gloves', label: 'Gloves' },
  { id: 'safety_shoes', label: 'Safety Boots / Shoes' },
  { id: 'mask', label: 'Mask / Respirator' },
  { id: 'ear_protection', label: 'Ear Protection / Muffs' },
];

const CAPTURE_ANGLES: Array<{ id: 'FRONT' | 'LEFT' | 'RIGHT'; label: string; instruction: string }> = [
  { id: 'FRONT', label: '1. Frontal Pose', instruction: 'Look directly into the camera with a neutral expression' },
  { id: 'LEFT', label: '2. Left Angle', instruction: 'Turn your head slightly to your left (~15° angle)' },
  { id: 'RIGHT', label: '3. Right Angle', instruction: 'Turn your head slightly to your right (~15° angle)' },
];

export const EmployeesView: React.FC = () => {
  const [employees, setEmployees] = useState<Employee[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [showAddModal, setShowAddModal] = useState<boolean>(false);
  const [showEnrollModal, setShowEnrollModal] = useState<boolean>(false);
  const [showDetailsModal, setShowDetailsModal] = useState<boolean>(false);
  const [selectedEmp, setSelectedEmp] = useState<Employee | null>(null);
  const [biometricDetails, setBiometricDetails] = useState<BiometricDetails | null>(null);
  const [detailsLoading, setDetailsLoading] = useState<boolean>(false);

  // Form states for new employee
  const [newId, setNewId] = useState<string>('');
  const [newName, setNewName] = useState<string>('');
  const [newDepartment, setNewDepartment] = useState<string>('Operations');
  const [newDesignation, setNewDesignation] = useState<string>('Site Engineer');
  const [newRequiredPpe, setNewRequiredPpe] = useState<string[]>(['helmet', 'safety_vest']);
  const [consentAgreed, setConsentAgreed] = useState<boolean>(true);
  const [formError, setFormError] = useState<string | null>(null);
  const [formSubmitting, setFormSubmitting] = useState<boolean>(false);

  // 3-Angle Enrollment Wizard states
  const [currentAngle, setCurrentAngle] = useState<'FRONT' | 'LEFT' | 'RIGHT'>('FRONT');
  const [completedAngles, setCompletedAngles] = useState<string[]>([]);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [streamActive, setStreamActive] = useState<boolean>(false);
  const [enrollmentStatus, setEnrollmentStatus] = useState<string | null>(null);
  const [enrollmentError, setEnrollmentError] = useState<string | null>(null);
  const [enrolling, setEnrolling] = useState<boolean>(false);
  const streamRef = useRef<MediaStream | null>(null);

  const loadData = async () => {
    try {
      setLoading(true);
      const data = await fetchEmployees();
      setEmployees(data);
    } catch (err: any) {
      console.error('Failed to load employees:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const filtered = employees.filter((e) => {
    const q = searchQuery.toLowerCase();
    return (
      e.name.toLowerCase().includes(q) ||
      e.id.toLowerCase().includes(q) ||
      e.department.toLowerCase().includes(q)
    );
  });

  const handleCreateEmployee = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);

    if (!newId.trim() || !newName.trim()) {
      setFormError('Employee ID and Full Name are required.');
      return;
    }

    try {
      setFormSubmitting(true);
      const res = await saveEmployee({
        id: newId.trim().toUpperCase(),
        name: newName.trim(),
        department: newDepartment.trim(),
        designation: newDesignation.trim(),
        required_ppe: newRequiredPpe,
        consent_status: consentAgreed ? 'CONSENTED' : 'PENDING',
        is_active: true,
      });

      setShowAddModal(false);
      setNewId('');
      setNewName('');
      await loadData();

      if (consentAgreed && res.employee) {
        startEnrollmentForEmp(res.employee);
      }
    } catch (err: any) {
      setFormError(err.message || 'Failed to save employee profile.');
    } finally {
      setFormSubmitting(false);
    }
  };

  const startEnrollmentForEmp = (emp: Employee) => {
    setSelectedEmp(emp);
    const existing = emp.completed_angles || [];
    setCompletedAngles(existing);
    // Select first missing angle
    if (!existing.includes('FRONT')) setCurrentAngle('FRONT');
    else if (!existing.includes('LEFT')) setCurrentAngle('LEFT');
    else if (!existing.includes('RIGHT')) setCurrentAngle('RIGHT');
    else setCurrentAngle('FRONT');
    setEnrollmentError(null);
    setEnrollmentStatus(null);
    setShowEnrollModal(true);
  };

  const handleOpenDetails = async (emp: Employee) => {
    setSelectedEmp(emp);
    setShowDetailsModal(true);
    setDetailsLoading(true);
    try {
      const details = await fetchBiometrics(emp.id);
      setBiometricDetails(details);
    } catch (err: any) {
      console.error('Error fetching biometrics:', err);
      setBiometricDetails(null);
    } finally {
      setDetailsLoading(false);
    }
  };

  const handleToggleStatus = async (emp: Employee) => {
    try {
      await toggleEmployeeStatus(emp.id, !emp.is_active);
      await loadData();
    } catch (err: any) {
      alert(`Error updating status: ${err.message}`);
    }
  };

  const handleWithdrawConsent = async (emp: Employee) => {
    if (
      !window.confirm(
        `Withdraw biometric consent for ${emp.name}? This will permanently purge all 3 face capture templates from PostgreSQL.`
      )
    ) {
      return;
    }
    try {
      await withdrawConsent(emp.id);
      await loadData();
    } catch (err: any) {
      alert(`Error withdrawing consent: ${err.message}`);
    }
  };

  const handleDeleteEmployee = async (emp: Employee) => {
    if (
      !window.confirm(
        `Are you sure you want to delete employee ${emp.name} (${emp.id})? All biometric records and profiles will be removed from PostgreSQL.`
      )
    ) {
      return;
    }
    try {
      await deleteEmployee(emp.id);
      await loadData();
    } catch (err: any) {
      alert(`Error deleting employee: ${err.message}`);
    }
  };

  // Camera enrollment
  const startEnrollmentCamera = async () => {
    setEnrollmentError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setStreamActive(true);
    } catch (err: any) {
      setEnrollmentError('Could not start webcam. Please grant camera permission in your browser.');
    }
  };

  const stopEnrollmentCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStreamActive(false);
  };

  const captureAngle = async () => {
    if (!videoRef.current || !selectedEmp) return;
    setEnrolling(true);
    setEnrollmentError(null);
    setEnrollmentStatus(null);

    try {
      const video = videoRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Canvas rendering error');

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const base64 = canvas.toDataURL('image/jpeg', 0.95);

      const res = await enrollAngleCapture(selectedEmp.id, currentAngle, base64);
      if (res.success) {
        setCompletedAngles(res.completed_angles);
        await loadData();

        if (res.is_complete) {
          setEnrollmentStatus('Success: All 3 angles (FRONT, LEFT, RIGHT) captured! Profile is now ACTIVE in PostgreSQL.');
          setTimeout(() => {
            stopEnrollmentCamera();
            setShowEnrollModal(false);
            setSelectedEmp(null);
          }, 2000);
        } else {
          setEnrollmentStatus(`Success: ${currentAngle} angle captured (Sharpness: ${Math.round(res.quality_metrics?.sharpness || 0)}). Please continue with the next angle.`);
          // Auto-advance to next missing angle
          if (currentAngle === 'FRONT' && !res.completed_angles.includes('LEFT')) {
            setCurrentAngle('LEFT');
          } else if (currentAngle === 'LEFT' && !res.completed_angles.includes('RIGHT')) {
            setCurrentAngle('RIGHT');
          } else if (!res.completed_angles.includes('FRONT')) {
            setCurrentAngle('FRONT');
          }
        }
      }
    } catch (err: any) {
      setEnrollmentError(err.message || 'Capture failed. Ensure correct head orientation and lighting.');
    } finally {
      setEnrolling(false);
    }
  };

  return (
    <div className="flex flex-col gap-6 w-full select-none font-sans">
      {/* Top Header & Search Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-white border border-gray-200 p-6 rounded-xl shadow-sm">
        <div>
          <h2 className="text-xl font-bold text-gray-900 tracking-tight flex items-center gap-2.5">
            <Users className="w-5 h-5 text-blue-600" />
            <span>Employee Directory & Biometrics</span>
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            PostgreSQL-backed biometric storage with 3-Angle enrollment (FRONT, LEFT, RIGHT) and role-based PPE compliance.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search by name, ID, department..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 pr-4 py-2 bg-white border border-gray-300 rounded-lg text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 w-64"
            />
          </div>

          <button
            type="button"
            onClick={loadData}
            disabled={loading}
            className="p-2 bg-white hover:bg-gray-50 text-gray-600 rounded-lg border border-gray-300 transition-colors shadow-sm"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>

          <button
            type="button"
            onClick={() => setShowAddModal(true)}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors shadow-sm"
          >
            <UserPlus className="w-4 h-4" />
            <span>Add Employee</span>
          </button>
        </div>
      </div>

      {/* Employees Table */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm space-y-4">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm border-collapse">
            <thead>
              <tr className="border-b border-gray-200 bg-gray-50 text-gray-600 uppercase text-xs font-semibold">
                <th className="py-3 px-3">Employee</th>
                <th className="py-3 px-3">Department & Role</th>
                <th className="py-3 px-3">Mandatory PPE</th>
                <th className="py-3 px-3">Biometric Profile</th>
                <th className="py-3 px-3">Angles Captured</th>
                <th className="py-3 px-3">Consent</th>
                <th className="py-3 px-3">Status</th>
                <th className="py-3 px-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-8 text-center text-gray-400 italic">
                    {searchQuery ? 'No employees matching search criteria.' : 'No employees registered yet. Click "Add Employee" above to get started.'}
                  </td>
                </tr>
              ) : (
                filtered.map((emp) => {
                  const angles = emp.completed_angles || [];
                  const isComplete = angles.length === 3;
                  const isConsented = emp.consent_status === 'CONSENTED';

                  return (
                    <tr key={emp.id} className="hover:bg-gray-50 transition-colors">
                      {/* Name & ID */}
                      <td className="py-3 px-3">
                        <div className="font-semibold text-gray-900">
                          {emp.name}
                        </div>
                        <div className="text-xs text-blue-600 font-mono">
                          ID: {emp.id}
                        </div>
                      </td>

                      {/* Department & Role */}
                      <td className="py-3 px-3">
                        <div className="text-gray-900 font-medium">{emp.department}</div>
                        <div className="text-xs text-gray-500">{emp.designation}</div>
                      </td>

                      {/* Mandatory PPE */}
                      <td className="py-3 px-3">
                        <div className="flex flex-wrap gap-1 max-w-xs">
                          {emp.required_ppe && emp.required_ppe.length > 0 ? (
                            emp.required_ppe.map((gear) => (
                              <span
                                key={gear}
                                className="text-xs px-2 py-0.5 rounded-md bg-gray-100 text-gray-700 font-medium"
                              >
                                {gear.replace('_', ' ')}
                              </span>
                            ))
                          ) : (
                            <span className="text-gray-400 text-xs">None</span>
                          )}
                        </div>
                      </td>

                      {/* Biometric Status */}
                      <td className="py-3 px-3">
                        <span
                          className={`text-xs px-2.5 py-1 rounded-full font-semibold inline-flex items-center gap-1.5 ${
                            isComplete
                              ? 'bg-blue-50 text-blue-700 border border-blue-200'
                              : 'bg-amber-50 text-amber-700 border border-amber-200'
                          }`}
                        >
                          {isComplete ? <CheckCircle2 className="w-3.5 h-3.5 text-blue-600" /> : <EyeOff className="w-3.5 h-3.5 text-amber-500" />}
                          <span>{isComplete ? 'ACTIVE (3/3)' : `PENDING (${angles.length}/3)`}</span>
                        </span>
                      </td>

                      {/* Angles Captured Badges */}
                      <td className="py-3 px-3">
                        <div className="flex items-center gap-1">
                          {['FRONT', 'LEFT', 'RIGHT'].map((a) => {
                            const done = angles.includes(a);
                            return (
                              <span
                                key={a}
                                className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-semibold ${
                                  done
                                    ? 'bg-green-100 text-green-800 border border-green-200'
                                    : 'bg-gray-100 text-gray-400 border border-gray-200'
                                }`}
                              >
                                {a}
                              </span>
                            );
                          })}
                        </div>
                      </td>

                      {/* Consent Status */}
                      <td className="py-3 px-3">
                        <span
                          className={`text-xs px-2.5 py-1 rounded-full font-semibold ${
                            isConsented
                              ? 'bg-green-50 text-green-700 border border-green-200'
                              : 'bg-red-50 text-red-700 border border-red-200'
                          }`}
                        >
                          {emp.consent_status}
                        </span>
                      </td>

                      {/* Active Status Toggle */}
                      <td className="py-3 px-3">
                        <button
                          type="button"
                          onClick={() => handleToggleStatus(emp)}
                          className={`text-xs px-2.5 py-1 rounded-full font-semibold transition-colors ${
                            emp.is_active
                              ? 'bg-green-100 text-green-800 hover:bg-green-200'
                              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                          }`}
                        >
                          {emp.is_active ? 'Active' : 'Inactive'}
                        </button>
                      </td>

                      {/* Action Buttons */}
                      <td className="py-3 px-3 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          {/* 3-Angle Enrollment Wizard */}
                          <button
                            type="button"
                            title="3-Angle Biometric Enrollment (FRONT, LEFT, RIGHT)"
                            onClick={() => startEnrollmentForEmp(emp)}
                            className="p-1.5 bg-blue-50 hover:bg-blue-100 text-blue-600 rounded-lg border border-blue-200 transition-colors"
                          >
                            <Camera className="w-4 h-4" />
                          </button>

                          {/* View Biometrics & Geometry Modal */}
                          <button
                            type="button"
                            title="Inspect Biometric Templates & Anthropometric Geometry"
                            onClick={() => handleOpenDetails(emp)}
                            className="p-1.5 bg-indigo-50 hover:bg-indigo-100 text-indigo-600 rounded-lg border border-indigo-200 transition-colors"
                          >
                            <Scan className="w-4 h-4" />
                          </button>

                          {/* Withdraw Consent */}
                          {angles.length > 0 && (
                            <button
                              type="button"
                              title="Withdraw Biometric Consent (Purges all 3 captures)"
                              onClick={() => handleWithdrawConsent(emp)}
                              className="p-1.5 bg-amber-50 hover:bg-amber-100 text-amber-700 rounded-lg border border-amber-200 transition-colors"
                            >
                              <ShieldAlert className="w-4 h-4" />
                            </button>
                          )}

                          {/* Delete */}
                          <button
                            type="button"
                            title="Delete Employee"
                            onClick={() => handleDeleteEmployee(emp)}
                            className="p-1.5 bg-red-50 hover:bg-red-100 text-red-600 rounded-lg border border-red-200 transition-colors"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Add Employee Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white border border-gray-200 rounded-xl max-w-lg w-full p-6 shadow-2xl animate-fade-in text-gray-900">
            <div className="flex items-center justify-between border-b border-gray-100 pb-3 mb-4">
              <h3 className="text-base font-bold text-gray-900 flex items-center gap-2">
                <UserPlus className="w-5 h-5 text-blue-600" />
                <span>Add New Employee</span>
              </h3>
              <button
                type="button"
                onClick={() => setShowAddModal(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {formError && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{formError}</span>
              </div>
            )}

            <form onSubmit={handleCreateEmployee} className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-gray-700 font-medium text-xs mb-1">
                    Employee ID *
                  </label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. EMP-101"
                    value={newId}
                    onChange={(e) => setNewId(e.target.value)}
                    className="w-full bg-white border border-gray-300 rounded-lg px-3 py-2 text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono text-xs uppercase"
                  />
                </div>
                <div>
                  <label className="block text-gray-700 font-medium text-xs mb-1">
                    Full Name *
                  </label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Alex Johnson"
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    className="w-full bg-white border border-gray-300 rounded-lg px-3 py-2 text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 text-xs"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-gray-700 font-medium text-xs mb-1">
                    Department
                  </label>
                  <input
                    type="text"
                    value={newDepartment}
                    onChange={(e) => setNewDepartment(e.target.value)}
                    className="w-full bg-white border border-gray-300 rounded-lg px-3 py-2 text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 text-xs"
                  />
                </div>
                <div>
                  <label className="block text-gray-700 font-medium text-xs mb-1">
                    Designation
                  </label>
                  <input
                    type="text"
                    value={newDesignation}
                    onChange={(e) => setNewDesignation(e.target.value)}
                    className="w-full bg-white border border-gray-300 rounded-lg px-3 py-2 text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 text-xs"
                  />
                </div>
              </div>

              <div>
                <label className="block text-gray-700 font-medium text-xs mb-2">
                  Mandatory PPE Requirements for this Role:
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {PPE_OPTIONS.map((opt) => {
                    const checked = newRequiredPpe.includes(opt.id);
                    return (
                      <label
                        key={opt.id}
                        className={`flex items-center gap-2 p-2 rounded-lg border cursor-pointer transition-colors text-xs ${
                          checked
                            ? 'bg-blue-50 border-blue-300 text-blue-900 font-medium'
                            : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setNewRequiredPpe([...newRequiredPpe, opt.id]);
                            } else {
                              setNewRequiredPpe(newRequiredPpe.filter((id) => id !== opt.id));
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

              <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg space-y-2">
                <label className="flex items-start gap-2.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={consentAgreed}
                    onChange={(e) => setConsentAgreed(e.target.checked)}
                    className="accent-blue-600 rounded mt-0.5"
                  />
                  <div className="text-xs text-gray-600 leading-relaxed">
                    <span className="font-semibold text-gray-900">
                      Biometric Consent Notice:
                    </span>{' '}
                    Employee consents to on-device 3-angle facial biometric analysis (FRONT, LEFT, RIGHT). 128-dimensional templates are securely stored in PostgreSQL and never sent to cloud services. Consent may be revoked anytime.
                  </div>
                </label>
              </div>

              <div className="flex items-center justify-end gap-3 pt-3 border-t border-gray-100">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="px-4 py-2 bg-white hover:bg-gray-50 text-gray-700 rounded-lg border border-gray-300 font-medium text-xs"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={formSubmitting}
                  className="px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg shadow-sm text-xs disabled:opacity-50 transition-colors"
                >
                  {formSubmitting ? 'Saving...' : 'Register Employee'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* 3-Angle Face Enrollment Wizard Modal */}
      {showEnrollModal && selectedEmp && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white border border-gray-200 rounded-xl max-w-xl w-full p-6 shadow-2xl animate-fade-in text-gray-900 space-y-4">
            <div className="flex items-center justify-between border-b border-gray-100 pb-3">
              <div>
                <h3 className="text-base font-bold text-gray-900 flex items-center gap-2">
                  <Camera className="w-5 h-5 text-blue-600" />
                  <span>3-Angle Biometric Enrollment Wizard</span>
                </h3>
                <p className="text-xs text-gray-500 mt-0.5">
                  Subject: <span className="text-blue-700 font-semibold">{selectedEmp.name}</span> ({selectedEmp.id})
                </p>
              </div>
              <button
                type="button"
                onClick={() => {
                  stopEnrollmentCamera();
                  setShowEnrollModal(false);
                  setSelectedEmp(null);
                }}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* 3-Angle Progress Tabs */}
            <div className="grid grid-cols-3 gap-2">
              {CAPTURE_ANGLES.map((step) => {
                const isDone = completedAngles.includes(step.id);
                const isCurrent = currentAngle === step.id;
                return (
                  <button
                    key={step.id}
                    type="button"
                    onClick={() => setCurrentAngle(step.id)}
                    className={`flex flex-col items-center p-2 rounded-lg border text-left transition-all ${
                      isCurrent
                        ? 'border-blue-600 bg-blue-50/80 shadow-xs ring-1 ring-blue-600'
                        : isDone
                        ? 'border-green-300 bg-green-50/50 text-gray-700'
                        : 'border-gray-200 bg-gray-50 text-gray-500'
                    }`}
                  >
                    <div className="flex items-center gap-1.5 text-xs font-bold">
                      {isDone ? (
                        <Check className="w-3.5 h-3.5 text-green-600 stroke-[3]" />
                      ) : (
                        <span className={`w-2 h-2 rounded-full ${isCurrent ? 'bg-blue-600' : 'bg-gray-300'}`} />
                      )}
                      <span className={isCurrent ? 'text-blue-900' : isDone ? 'text-green-900' : 'text-gray-600'}>
                        {step.label}
                      </span>
                    </div>
                    <span className="text-[10px] text-gray-500 mt-0.5">
                      {isDone ? 'Captured' : isCurrent ? 'Active Step' : 'Pending'}
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Active Instruction Notice */}
            <div className="p-2.5 bg-blue-50/80 border border-blue-200 rounded-lg flex items-center gap-2 text-xs text-blue-900">
              <Info className="w-4 h-4 text-blue-600 shrink-0" />
              <span>
                <strong>Target Pose ({currentAngle}):</strong>{' '}
                {CAPTURE_ANGLES.find((a) => a.id === currentAngle)?.instruction}
              </span>
            </div>

            {/* Camera Viewport */}
            <div className="relative bg-slate-900 rounded-lg overflow-hidden aspect-[4/3] flex items-center justify-center border border-gray-200">
              <video
                ref={videoRef}
                playsInline
                muted
                className={`w-full h-full object-cover ${streamActive ? 'block' : 'hidden'}`}
              />

              {!streamActive && (
                <div className="flex flex-col items-center justify-center p-6 text-center bg-white m-4 rounded-xl shadow-sm border border-gray-200">
                  <div className="p-3 bg-blue-50 text-blue-600 rounded-full mb-3">
                    <Camera className="w-6 h-6" />
                  </div>
                  <p className="text-xs text-gray-500 mb-4 max-w-xs">
                    Activate the webcam to begin capturing 3-angle biometric templates for PostgreSQL storage.
                  </p>
                  <button
                    type="button"
                    onClick={startEnrollmentCamera}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg text-xs transition-colors shadow-sm"
                  >
                    Start Enrollment Camera
                  </button>
                </div>
              )}

              {/* Angle Guide Outline */}
              {streamActive && (
                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                  <div className={`w-48 h-60 border-2 border-dashed rounded-[48px] transition-colors ${
                    currentAngle === 'FRONT'
                      ? 'border-blue-400'
                      : currentAngle === 'LEFT'
                      ? 'border-indigo-400 -rotate-6'
                      : 'border-indigo-400 rotate-6'
                  }`} />
                  <span className="mt-2 bg-black/60 backdrop-blur-xs text-white text-[11px] px-2.5 py-0.5 rounded-full font-mono">
                    Pose: {currentAngle}
                  </span>
                </div>
              )}
            </div>

            {/* Status Feedback */}
            {enrollmentError && (
              <div className="p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{enrollmentError}</span>
              </div>
            )}

            {enrollmentStatus && (
              <div className="p-3 bg-green-50 border border-green-200 text-green-700 rounded-lg text-xs flex items-center gap-2">
                <Check className="w-4 h-4 shrink-0" />
                <span>{enrollmentStatus}</span>
              </div>
            )}

            <div className="flex items-center justify-between pt-2">
              <span className="text-xs text-gray-500 font-mono">
                Storage: PostgreSQL 18 (BYTEA 128-d)
              </span>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => {
                    stopEnrollmentCamera();
                    setShowEnrollModal(false);
                    setSelectedEmp(null);
                  }}
                  className="px-4 py-2 bg-white hover:bg-gray-50 text-gray-700 rounded-lg text-xs font-medium border border-gray-300"
                >
                  Close
                </button>
                {streamActive && (
                  <button
                    type="button"
                    onClick={captureAngle}
                    disabled={enrolling}
                    className="px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-semibold shadow-sm disabled:opacity-50 transition-colors flex items-center gap-1.5"
                  >
                    {enrolling ? 'Extracting & Validating...' : `Capture ${currentAngle} Angle`}
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Biometric & Anthropometric Geometry Details Modal */}
      {showDetailsModal && selectedEmp && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white border border-gray-200 rounded-xl max-w-2xl w-full p-6 shadow-2xl animate-fade-in text-gray-900 space-y-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-gray-100 pb-3">
              <div>
                <h3 className="text-base font-bold text-gray-900 flex items-center gap-2">
                  <Scan className="w-5 h-5 text-indigo-600" />
                  <span>Biometric Profile & Facial Geometry (PostgreSQL)</span>
                </h3>
                <p className="text-xs text-gray-500 mt-0.5">
                  Employee: <span className="text-blue-700 font-semibold">{selectedEmp.name}</span> ({selectedEmp.id})
                </p>
              </div>
              <button
                type="button"
                onClick={() => {
                  setShowDetailsModal(false);
                  setSelectedEmp(null);
                  setBiometricDetails(null);
                }}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {detailsLoading ? (
              <div className="py-12 text-center text-gray-400 text-sm">
                <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2 text-indigo-600" />
                <span>Loading biometric records from PostgreSQL...</span>
              </div>
            ) : !biometricDetails || biometricDetails.captures.length === 0 ? (
              <div className="py-8 text-center text-gray-500 bg-gray-50 rounded-lg p-6 border border-gray-200">
                <EyeOff className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                <p className="font-semibold text-sm">No Biometric Captures Found</p>
                <p className="text-xs text-gray-500 mt-1">
                  This employee has not completed 3-angle biometric enrollment yet.
                </p>
              </div>
            ) : (
              <div className="space-y-4 text-xs">
                {/* Profile Overview */}
                <div className="grid grid-cols-3 gap-3 bg-gray-50 p-3 rounded-lg border border-gray-200">
                  <div>
                    <span className="text-gray-500 block">Enrollment Status:</span>
                    <span className="font-bold text-blue-700">{biometricDetails.enrollment_status}</span>
                  </div>
                  <div>
                    <span className="text-gray-500 block">Consent Status:</span>
                    <span className="font-bold text-green-700">{biometricDetails.consent_status}</span>
                  </div>
                  <div>
                    <span className="text-gray-500 block">Completed Angles:</span>
                    <span className="font-bold text-gray-900">{biometricDetails.completed_angles.join(', ')} (3 required)</span>
                  </div>
                </div>

                {/* Captures Table */}
                <div>
                  <h4 className="font-bold text-gray-800 mb-2">Stored Face Captures (employee_face_captures):</h4>
                  <div className="border border-gray-200 rounded-lg overflow-hidden">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-gray-100 text-gray-700 font-semibold border-b border-gray-200">
                        <tr>
                          <th className="p-2">Angle</th>
                          <th className="p-2">Embedding</th>
                          <th className="p-2">Pose (Yaw / Roll)</th>
                          <th className="p-2">Sharpness</th>
                          <th className="p-2">Recognition Model</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {biometricDetails.captures.map((c) => (
                          <tr key={c.capture_angle} className="hover:bg-gray-50">
                            <td className="p-2 font-bold text-blue-700 font-mono">{c.capture_angle}</td>
                            <td className="p-2 font-mono">{c.embedding_dimension}-d BYTEA (512B)</td>
                            <td className="p-2">
                              {c.face_pose?.estimated_yaw_degrees !== undefined ? `${c.face_pose.estimated_yaw_degrees}° yaw, ${c.face_pose?.roll_degrees || 0}° roll` : 'N/A'}
                            </td>
                            <td className="p-2">{c.face_quality?.sharpness || 'N/A'}</td>
                            <td className="p-2">{c.embedding_model} ({c.embedding_model_version})</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Calculated Anthropometric Geometry Ratios */}
                {biometricDetails.captures.length > 0 && biometricDetails.captures[0].facial_geometry && (
                  <div className="p-3 bg-indigo-50/50 border border-indigo-200 rounded-lg space-y-2">
                    <h4 className="font-bold text-indigo-950 flex items-center gap-1.5">
                      <Info className="w-4 h-4 text-indigo-600" />
                      <span>Calculated Facial Geometry Ratios (Anthropometric Features):</span>
                    </h4>
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                      <div className="bg-white p-2 rounded border border-indigo-100">
                        <span className="text-gray-500 block">Width/Height Ratio:</span>
                        <span className="font-semibold">{biometricDetails.captures[0].facial_geometry.face_width_to_height_ratio || 'N/A'}</span>
                      </div>
                      <div className="bg-white p-2 rounded border border-indigo-100">
                        <span className="text-gray-500 block">Inter-Ocular Dist:</span>
                        <span className="font-semibold">{biometricDetails.captures[0].facial_geometry.inter_ocular_distance_px ? `${biometricDetails.captures[0].facial_geometry.inter_ocular_distance_px} px` : 'N/A'}</span>
                      </div>
                      <div className="bg-white p-2 rounded border border-indigo-100">
                        <span className="text-gray-500 block">Eye / Face Width:</span>
                        <span className="font-semibold">{biometricDetails.captures[0].facial_geometry.inter_eye_to_face_width_ratio || 'N/A'}</span>
                      </div>
                      <div className="bg-white p-2 rounded border border-indigo-100">
                        <span className="text-gray-500 block">Mouth Width:</span>
                        <span className="font-semibold">{biometricDetails.captures[0].facial_geometry.mouth_width_px ? `${biometricDetails.captures[0].facial_geometry.mouth_width_px} px` : 'N/A'}</span>
                      </div>
                      <div className="bg-white p-2 rounded border border-indigo-100">
                        <span className="text-gray-500 block">Nose-to-Mouth:</span>
                        <span className="font-semibold">{biometricDetails.captures[0].facial_geometry.nose_to_mouth_distance_px ? `${biometricDetails.captures[0].facial_geometry.nose_to_mouth_distance_px} px` : 'N/A'}</span>
                      </div>
                      <div className="bg-white p-2 rounded border border-indigo-100">
                        <span className="text-gray-500 block">Jaw / Face Width:</span>
                        <span className="font-semibold">{biometricDetails.captures[0].facial_geometry.jaw_to_face_width_ratio || 'N/A'}</span>
                      </div>
                    </div>
                    <p className="text-[11px] text-gray-500 italic mt-2">
                      Notice: 2D Camera limitation — Z-depth is anthropometrically estimated and not from a 3D sensor. These geometric proportions assist alignment and do not uniquely identify an individual.
                    </p>
                  </div>
                )}
              </div>
            )}

            <div className="flex justify-end pt-2 border-t border-gray-100">
              <button
                type="button"
                onClick={() => {
                  setShowDetailsModal(false);
                  setSelectedEmp(null);
                  setBiometricDetails(null);
                }}
                className="px-4 py-2 bg-white hover:bg-gray-50 text-gray-700 rounded-lg text-xs font-medium border border-gray-300"
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
