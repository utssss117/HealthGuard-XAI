'use client';

import { useState } from 'react';
import { useAuth } from '@clerk/nextjs';
import { predictRisk, type Biomarkers, type PredictResponse } from '@/lib/api';
import ApiError from '@/components/ui/ApiError';

type Props = {
  biomarkers: Biomarkers | null;
  prediction: PredictResponse | null;
};

const FIELD_LABELS: Record<keyof Biomarkers, { label: string; unit: string; step: number }> = {
  Pregnancies:              { label: 'Pregnancies',       unit: 'count',  step: 1 },
  Glucose:                  { label: 'Glucose',           unit: 'mg/dL',  step: 1 },
  BloodPressure:            { label: 'Blood Pressure',    unit: 'mmHg',   step: 1 },
  SkinThickness:            { label: 'Skin Thickness',    unit: 'mm',     step: 1 },
  Insulin:                  { label: 'Insulin',           unit: 'μU/mL',  step: 1 },
  BMI:                      { label: 'BMI',               unit: 'kg/m²',  step: 0.1 },
  DiabetesPedigreeFunction: { label: 'Diabetes Pedigree', unit: 'score',  step: 0.001 },
  Age:                      { label: 'Age',               unit: 'years',  step: 1 },
};

const RISK_STYLES = {
  Low:    'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20',
  Medium: 'bg-amber-500/10 text-amber-400 border border-amber-500/20',
  High:   'bg-rose-500/10 text-rose-400 border border-rose-500/20',
};

export default function CounterfactualTab({ biomarkers, prediction }: Props) {
  const { getToken } = useAuth();
  const [modified, setModified] = useState<Biomarkers | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialise modified values from original on first render
  const working: Biomarkers = modified ?? (biomarkers ? { ...biomarkers } : {} as Biomarkers);

  if (!biomarkers || !prediction) {
    return (
      <div className="glass-panel p-12 flex flex-col items-center text-center justify-center min-h-[300px]">
        <span className="text-5xl mb-4 opacity-75 filter drop-shadow-[0_0_15px_rgba(59,130,246,0.2)]">🔄</span>
        <p className="text-slate-400 text-sm max-w-xs">Run a <strong className="text-slate-300">Risk Assessment</strong> first to explore what-if scenarios.</p>
      </div>
    );
  }

  const update = (k: keyof Biomarkers, v: string) =>
    setModified(p => ({ ...(p ?? biomarkers), [k]: parseFloat(v) || 0 }));

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const token = await getToken();
      const res = await predictRisk(working, token);
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => { setModified(null); setResult(null); };

  return (
    <div className="space-y-6">
      <div className="glass-panel p-6">
        <h2 className="font-bold text-white text-lg mb-1">What-If Analysis</h2>
        <p className="text-slate-400 text-sm mb-6">
          Adjust biomarker levels in real-time to analyze how lifestyle or physiological modifications impact total risk.
        </p>

        {error && <ApiError message={error} />}

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          {(Object.keys(FIELD_LABELS) as (keyof Biomarkers)[]).map(key => {
            const f = FIELD_LABELS[key];
            const orig = biomarkers[key];
            const curr = working[key];
            const changed = orig !== curr;
            return (
              <div key={key} className={`rounded-xl border p-4 transition-colors duration-200 ${changed ? 'border-blue-500/30 bg-blue-500/5' : 'border-slate-800 bg-slate-900/10'}`}>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-xs font-semibold text-slate-300">{f.label}</label>
                  {changed && (
                    <span className="text-[10px] text-slate-500 font-mono">orig: {orig}</span>
                  )}
                </div>
                <input
                  type="number"
                  step={f.step}
                  value={curr}
                  onChange={e => update(key, e.target.value)}
                  className="input-premium w-full px-2.5 py-1.5 text-sm focus:outline-none font-mono"
                />
                <span className="text-[10px] text-slate-500 mt-1 block uppercase tracking-wide">{f.unit}</span>
              </div>
            );
          })}
        </div>

        <div className="flex items-center gap-3">
          <button
            id="btn-whatif-run"
            onClick={handleRun}
            disabled={loading}
            className="btn-premium px-5 py-3 text-sm font-semibold transition-all duration-200 cursor-pointer disabled:opacity-50"
          >
            {loading ? 'Running simulation…' : '▶ Run What-If Prediction'}
          </button>
          <button
            onClick={handleReset}
            className="border border-slate-800 text-slate-400 hover:text-slate-200 hover:bg-white/[0.03] text-sm font-semibold px-5 py-3 rounded-lg transition-colors cursor-pointer"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Comparison panel */}
      {result && (
        <div className="glass-panel p-6">
          <h3 className="font-bold text-white text-base mb-6">Simulation Comparison</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-6">
            {[
              { label: 'Original Assessment Profile', pred: prediction },
              { label: 'Modified What-If Profile', pred: result },
            ].map(({ label, pred }) => (
              <div key={label} className="border border-slate-800/80 bg-slate-900/10 p-5 rounded-xl text-center">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4">{label}</p>
                <span className={`px-4 py-2 rounded-full text-base font-bold tracking-wide ${RISK_STYLES[pred.risk_level]}`}>
                  {pred.risk_level.toUpperCase()} RISK — {Math.round(pred.risk_probability * 100)}%
                </span>
              </div>
            ))}
          </div>

          <div className={`px-5 py-4 rounded-xl text-sm font-semibold text-center border ${
            result.risk_probability < prediction.risk_probability
              ? 'bg-emerald-500/5 text-emerald-400 border-emerald-500/10'
              : result.risk_probability > prediction.risk_probability
              ? 'bg-rose-500/5 text-rose-400 border-rose-500/10'
              : 'bg-slate-900 text-slate-400 border-slate-800'
          }`}>
            {result.risk_probability < prediction.risk_probability
              ? `✅ Risk successfully reduced by ${Math.round((prediction.risk_probability - result.risk_probability) * 100)} percentage points!`
              : result.risk_probability > prediction.risk_probability
              ? `⚠️ Risk increased by ${Math.round((result.risk_probability - prediction.risk_probability) * 100)} percentage points.`
              : `No change in total risk output.`}
          </div>
        </div>
      )}
    </div>
  );
}
