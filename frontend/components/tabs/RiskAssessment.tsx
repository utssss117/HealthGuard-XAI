'use client';

import { useState } from 'react';
import { useAuth } from '@clerk/nextjs';
import { predictRisk, type Biomarkers, type PredictResponse } from '@/lib/api';
import ApiError from '@/components/ui/ApiError';

type Props = {
  onResult: (bm: Biomarkers, pred: PredictResponse) => void;
  prediction: PredictResponse | null;
};

const FIELDS: { key: keyof Biomarkers; label: string; unit: string; min: number; max: number; step: number; hint: string }[] = [
  { key: 'Pregnancies',              label: 'Pregnancies',              unit: 'count',  min: 0,  max: 20,  step: 1,    hint: 'Number of times pregnant' },
  { key: 'Glucose',                  label: 'Plasma Glucose',           unit: 'mg/dL',  min: 0,  max: 300, step: 1,    hint: '2hr oral glucose tolerance test' },
  { key: 'BloodPressure',            label: 'Blood Pressure',           unit: 'mmHg',   min: 0,  max: 200, step: 1,    hint: 'Diastolic blood pressure' },
  { key: 'SkinThickness',            label: 'Skin Thickness',           unit: 'mm',     min: 0,  max: 100, step: 1,    hint: 'Triceps skinfold thickness' },
  { key: 'Insulin',                  label: '2hr Serum Insulin',        unit: 'μU/mL',  min: 0,  max: 900, step: 1,    hint: '2-hour serum insulin level' },
  { key: 'BMI',                      label: 'BMI',                      unit: 'kg/m²',  min: 0,  max: 80,  step: 0.1,  hint: 'Body Mass Index' },
  { key: 'DiabetesPedigreeFunction', label: 'Diabetes Pedigree',        unit: 'score',  min: 0,  max: 3,   step: 0.001,hint: 'Genetic diabetes likelihood function' },
  { key: 'Age',                      label: 'Age',                      unit: 'years',  min: 1,  max: 120, step: 1,    hint: 'Age in years' },
];

const DEFAULT: Biomarkers = {
  Pregnancies: 2, Glucose: 120, BloodPressure: 72, SkinThickness: 25,
  Insulin: 80, BMI: 28.5, DiabetesPedigreeFunction: 0.351, Age: 35,
};

const RISK_STYLES = {
  Low:    { text: 'text-emerald-400',  badge: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',  border: 'border-l-emerald-500' },
  Medium: { text: 'text-amber-400',   badge: 'bg-amber-500/10 text-amber-400 border-amber-500/20',    border: 'border-l-amber-500' },
  High:   { text: 'text-rose-400',    badge: 'bg-rose-500/10 text-rose-400 border-rose-500/20',      border: 'border-l-rose-500' },
};

export default function RiskAssessmentTab({ onResult, prediction }: Props) {
  const { getToken } = useAuth();
  const [form, setForm] = useState<Biomarkers>(DEFAULT);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [wakingUp, setWakingUp] = useState(false);

  const update = (k: keyof Biomarkers, v: string) =>
    setForm(p => ({ ...p, [k]: parseFloat(v) || 0 }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    const timer = setTimeout(() => setWakingUp(true), 5000);
    try {
      const token = await getToken();
      const result = await predictRisk(form, token);
      onResult(form, result);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      clearTimeout(timer);
      setLoading(false);
      setWakingUp(false);
    }
  };

  const rs = prediction ? RISK_STYLES[prediction.risk_level] : null;
  const pct = prediction ? Math.round(prediction.risk_probability * 100) : 0;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      {/* Form */}
      <div className="glass-panel p-6">
        <h2 className="font-bold text-white text-lg mb-1">Biomarker Input</h2>
        <p className="text-slate-400 text-sm mb-6">Enter the 8 clinical biomarkers to run the predictive analysis.</p>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-2 gap-4">
            {FIELDS.map(f => (
              <div key={f.key} className="flex flex-col">
                <label className="text-xs font-semibold text-slate-300 mb-1.5">
                  {f.label} <span className="text-slate-500 font-normal">({f.unit})</span>
                </label>
                <input
                  id={`input-${f.key}`}
                  type="number"
                  min={f.min} max={f.max} step={f.step}
                  value={form[f.key]}
                  onChange={e => update(f.key, e.target.value)}
                  required
                  className="input-premium px-3 py-2 text-sm focus:outline-none"
                />
                <span className="text-[10px] text-slate-500 mt-1">{f.hint}</span>
              </div>
            ))}
          </div>

          {error && <ApiError message={error} />}

          {wakingUp && (
            <div className="flex items-center gap-2.5 text-xs text-amber-400 bg-amber-500/5 border border-amber-500/10 rounded-xl px-4 py-3">
              <span className="animate-spin inline-block">⟳</span>
              Waking up the backend server (Render cold-start — up to 30s)…
            </div>
          )}

          <button
            id="btn-predict"
            type="submit"
            disabled={loading}
            className="btn-premium w-full py-3 text-sm font-semibold transition-all duration-200 cursor-pointer disabled:opacity-50"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Analyzing Biomarkers…
              </span>
            ) : '🔍 Predict Health Risk'}
          </button>
        </form>
      </div>

      {/* Result */}
      {prediction && rs ? (
        <div className={`glass-panel p-6 border-l-4 ${rs.border}`}>
          <h2 className="font-bold text-white text-lg mb-6">Assessment Result</h2>

          {/* Risk Gauge */}
          <div className="flex flex-col items-center mb-8">
            <div className="relative w-48 h-24 overflow-hidden mb-4">
              <div className="absolute inset-0 rounded-t-full border-[10px] border-slate-800" style={{ borderBottomColor: 'transparent' }} />
              <div
                className="absolute inset-0 rounded-t-full border-[10px] transition-all duration-1000 ease-out"
                style={{
                  borderColor: prediction.risk_level === 'Low' ? '#10b981' : prediction.risk_level === 'Medium' ? '#f59e0b' : '#f43f5e',
                  borderBottomColor: 'transparent',
                  transform: `rotate(${-180 + (prediction.risk_probability * 180)}deg)`,
                  transformOrigin: '50% 100%',
                  opacity: 0.9,
                }}
              />
              <div className="absolute bottom-0 left-1/2 -translate-x-1/2 text-center">
                <span className={`text-4xl font-extrabold tracking-tight ${rs.text}`}>{pct}%</span>
              </div>
            </div>
            <span className={`px-4 py-1.5 rounded-full text-xs font-bold border ${rs.badge}`}>
              {prediction.risk_level.toUpperCase()} RISK
            </span>
          </div>

          {/* Top features */}
          <div className="space-y-4">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Key Diagnostic Factors</h3>
            <div className="space-y-3.5">
              {Object.entries(prediction.top_features)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 5)
                .map(([feat, val]) => {
                  const max = Math.max(...Object.values(prediction.top_features));
                  const w = max > 0 ? (val / max) * 100 : 0;
                  return (
                    <div key={feat} className="group">
                      <div className="flex justify-between text-xs mb-1.5">
                        <span className="text-slate-300 font-medium">{feat}</span>
                        <span className="text-slate-400 font-mono">{(val * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 bg-slate-900 rounded-full overflow-hidden border border-white/[0.02]">
                        <div
                          className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full transition-all duration-700 group-hover:opacity-90"
                          style={{ width: `${w}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>

          <p className="mt-8 text-xs text-slate-500 italic text-center">
            Navigate to the <strong className="text-slate-400 font-medium">Explainability</strong> tab for SHAP-based local explainability analysis.
          </p>
        </div>
      ) : (
        <div className="glass-panel p-8 flex flex-col items-center justify-center text-center min-h-[350px]">
          <span className="text-5xl mb-4 opacity-80 filter drop-shadow-[0_0_15px_rgba(59,130,246,0.2)]">📋</span>
          <h3 className="text-white font-semibold text-base mb-1">Filing Pending</h3>
          <p className="text-slate-400 text-sm max-w-xs">Fill out the clinical parameters on the left and click predict to generate results.</p>
        </div>
      )}
    </div>
  );
}
