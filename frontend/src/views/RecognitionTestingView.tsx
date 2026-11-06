import React, { useState } from 'react';
import { 
  Scale, 
  Upload, 
  CheckCircle2, 
  AlertTriangle, 
  Info, 
  Eye, 
  Layers, 
  ShieldCheck,
  Binary
} from 'lucide-react';
import { PairCompareResponse } from '../types';

export const RecognitionTestingView: React.FC = () => {
  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);
  const [previewA, setPreviewA] = useState<string | null>(null);
  const [previewB, setPreviewB] = useState<string | null>(null);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PairCompareResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleSelectA = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setFileA(file);
      setPreviewA(URL.createObjectURL(file));
      setResult(null);
      setErrorMsg(null);
    }
  };

  const handleSelectB = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setFileB(file);
      setPreviewB(URL.createObjectURL(file));
      setResult(null);
      setErrorMsg(null);
    }
  };

  const runComparison = async () => {
    if (!fileA || !fileB) {
      setErrorMsg('Please select both Probe Face A and Reference Face B.');
      return;
    }

    setLoading(true);
    setErrorMsg(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file_a', fileA);
      formData.append('file_b', fileB);

      const res = await fetch('/api/recognition/compare-pair', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Pair comparison failed');
      }

      setResult(data);
    } catch (err: any) {
      setErrorMsg(err.message || 'Error executing biometric comparison');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto py-2">
      {/* Top Banner */}
      <div className="p-6 rounded-2xl bg-white border border-gray-200 shadow-sm flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-3 rounded-xl bg-blue-50 text-blue-600 border border-blue-200">
            <Scale className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-900">Biometric Ambiguity & Similar-Person Testing Lab</h1>
            <p className="text-xs text-gray-500 mt-0.5">
              Evaluates 128-D dlib facial descriptors against calibrated Euclidean boundaries and rejects false positive matches.
            </p>
          </div>
        </div>

        <div className="hidden sm:flex items-center space-x-2 text-xs font-mono px-3 py-1.5 rounded-lg bg-gray-50 border border-gray-200">
          <span className="w-2 h-2 rounded-full bg-emerald-500" />
          <span className="text-gray-600">dlib ResNet-34 · CPU Execution</span>
        </div>
      </div>

      {/* Threshold Reference Card */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
        <div className="p-3.5 rounded-xl bg-emerald-50/80 border border-emerald-200">
          <div className="font-mono font-bold text-emerald-800">d &le; 0.58 &amp; Margin &ge; 0.08</div>
          <p className="text-emerald-700 mt-1">
            <strong>REGISTERED:</strong> Confident identity match with clear separation from 2nd closest candidate.
          </p>
        </div>
        <div className="p-3.5 rounded-xl bg-amber-50/80 border border-amber-200">
          <div className="font-mono font-bold text-amber-800">0.58 &le; d &le; 0.65 OR Margin &lt; 0.08</div>
          <p className="text-amber-700 mt-1">
            <strong>UNCERTAIN:</strong> Ambiguity rejection zone. Prevents false positive assignment between similar individuals.
          </p>
        </div>
        <div className="p-3.5 rounded-xl bg-rose-50/80 border border-rose-200">
          <div className="font-mono font-bold text-rose-800">d &gt; 0.65</div>
          <p className="text-rose-700 mt-1">
            <strong>UNREGISTERED:</strong> Distinct identity. Distance significantly exceeds biometric match envelope.
          </p>
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-sm flex items-center space-x-2">
          <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Side-by-Side Image Upload & Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Probe Image A */}
        <div className="p-5 rounded-2xl bg-white border border-gray-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-800">Probe Face A (Test / Live Frame)</h3>
            <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200">
              Subject A
            </span>
          </div>

          <div className="aspect-4/3 rounded-xl bg-gray-50 border-2 border-dashed border-gray-300 overflow-hidden flex flex-col items-center justify-center relative">
            {previewA ? (
              <img src={previewA} alt="Probe A" className="w-full h-full object-contain" />
            ) : (
              <label className="cursor-pointer flex flex-col items-center space-y-2 text-gray-400 hover:text-blue-600 transition-colors">
                <Upload className="w-8 h-8" />
                <span className="text-xs font-medium">Upload Probe Face A (JPG/PNG)</span>
                <input type="file" accept="image/*" onChange={handleSelectA} className="hidden" />
              </label>
            )}
          </div>

          {previewA && (
            <label className="block text-center cursor-pointer text-xs text-blue-600 hover:text-blue-700 font-medium">
              <span>Change Image A</span>
              <input type="file" accept="image/*" onChange={handleSelectA} className="hidden" />
            </label>
          )}
        </div>

        {/* Candidate Image B */}
        <div className="p-5 rounded-2xl bg-white border border-gray-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-800">Reference Face B (Comparison Candidate)</h3>
            <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
              Subject B
            </span>
          </div>

          <div className="aspect-4/3 rounded-xl bg-gray-50 border-2 border-dashed border-gray-300 overflow-hidden flex flex-col items-center justify-center relative">
            {previewB ? (
              <img src={previewB} alt="Candidate B" className="w-full h-full object-contain" />
            ) : (
              <label className="cursor-pointer flex flex-col items-center space-y-2 text-gray-400 hover:text-indigo-600 transition-colors">
                <Upload className="w-8 h-8" />
                <span className="text-xs font-medium">Upload Reference Face B (JPG/PNG)</span>
                <input type="file" accept="image/*" onChange={handleSelectB} className="hidden" />
              </label>
            )}
          </div>

          {previewB && (
            <label className="block text-center cursor-pointer text-xs text-indigo-600 hover:text-indigo-700 font-medium">
              <span>Change Image B</span>
              <input type="file" accept="image/*" onChange={handleSelectB} className="hidden" />
            </label>
          )}
        </div>
      </div>

      {/* Compare Button */}
      <div className="flex justify-center">
        <button
          onClick={runComparison}
          disabled={!fileA || !fileB || loading}
          className={`flex items-center space-x-2 px-8 py-3 rounded-xl font-medium text-sm transition-all ${
            fileA && fileB && !loading
              ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/25'
              : 'bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200'
          }`}
        >
          <Scale className="w-5 h-5" />
          <span>{loading ? 'Generating 128-D Descriptors...' : 'Compute Euclidean Distance & Evaluate'}</span>
        </button>
      </div>

      {/* Comparison Results Card */}
      {result && (
        <div className="p-6 rounded-2xl bg-white border border-gray-200 shadow-sm space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-gray-100">
            <div>
              <span className="text-xs font-mono font-semibold text-gray-500">TEST VERDICT</span>
              <h2 className="text-2xl font-bold text-gray-900 mt-0.5">{result.verdict}</h2>
            </div>

            <div>
              <span
                className={`px-3 py-1 rounded-lg text-xs font-bold border ${
                  result.status === 'REGISTERED'
                    ? 'bg-emerald-50 text-emerald-700 border-emerald-300'
                    : result.status === 'UNCERTAIN'
                    ? 'bg-amber-50 text-amber-700 border-amber-300'
                    : 'bg-rose-50 text-rose-700 border-rose-300'
                }`}
              >
                STATUS: {result.status}
              </span>
            </div>
          </div>

          {/* Metric Bar */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono">
            <div className="p-4 rounded-xl bg-gray-50 border border-gray-200">
              <span className="text-xs text-gray-500">128-D Euclidean Distance</span>
              <div className="text-3xl font-extrabold text-gray-900 mt-1">
                {result.distance.toFixed(4)}
              </div>
              <p className="text-[10px] text-gray-500 mt-1">
                Match range: 0.00 – 0.58
              </p>
            </div>

            <div className="p-4 rounded-xl bg-gray-50 border border-gray-200">
              <span className="text-xs text-gray-500">Calibrated Match Threshold</span>
              <div className="text-3xl font-extrabold text-blue-600 mt-1">
                {result.best_threshold.toFixed(2)}
              </div>
              <p className="text-[10px] text-gray-500 mt-1">
                Dlib ResNet boundary
              </p>
            </div>

            <div className="p-4 rounded-xl bg-gray-50 border border-gray-200">
              <span className="text-xs text-gray-500">Uncertainty Upper Bound</span>
              <div className="text-3xl font-extrabold text-amber-600 mt-1">
                {result.uncertain_threshold.toFixed(2)}
              </div>
              <p className="text-[10px] text-gray-500 mt-1">
                Above 0.65 = strictly unregistered
              </p>
            </div>
          </div>

          {/* Scientific Explanation */}
          <div className="p-4 rounded-xl bg-blue-50/50 border border-blue-200 space-y-1.5">
            <div className="flex items-center space-x-2 text-xs font-semibold text-blue-900">
              <Info className="w-4 h-4 text-blue-600" />
              <span>Scientific Explanation</span>
            </div>
            <p className="text-sm text-gray-700 leading-relaxed font-sans">
              {result.explanation}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

