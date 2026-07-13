'use client';

import { useState } from 'react';
import { useAuth } from '@clerk/nextjs';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Cell,
} from 'recharts';
import { getExplanation, type Biomarkers, type PredictResponse, type ExplainResponse } from '@/lib/api';
import ApiError from '@/components/ui/ApiError';

type Props = {
  biomarkers: Biomarkers | null;
  prediction: PredictResponse | null;
  explanation: ExplainResponse | null;
  onExplanation: (e: ExplainResponse) => void;
};

export default function ExplainabilityTab({ biomarkers, prediction, explanation, onExplanation }: Props) {
  const { getToken } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [wakingUp, setWakingUp] = useState(false);

  const runExplain = async () => {
    if (!biomarkers) return;
    setLoading(true);
    setError(null);
    const timer = setTimeout(() => setWakingUp(true), 5000);
    try {
      const token = await getToken();
      const result = await getExplanation(biomarkers, token);
      onExplanation(result);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      clearTimeout(timer);
      setLoading(false);
      setWakingUp(false);
    }
  };

  if (!prediction || !biomarkers) {
    return (
      <EmptyState icon="🧠" message="Run a Risk Assessment first to unlock SHAP explainability." />
    );
  }

  const chartData = explanation
    ? Object.entries(explanation.feature_importances)
        .map(([name, value]) => ({ name, value: parseFloat(value.toFixed(4)) }))
        .sort((a, b) => b.value - a.value)
    : null;

  return (
    <div className="space-y-6">
      <div className="glass-panel p-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
          <div>
            <h2 className="font-bold text-white text-lg">SHAP Feature Attributions</h2>
            <p className="text-slate-400 text-sm mt-1">
              Positive values indicate risk-increasing elements. Negative values denote protective factors.
            </p>
          </div>
          {!explanation && (
            <button
              id="btn-explain"
              onClick={runExplain}
              disabled={loading}
              className="btn-premium px-5 py-2.5 text-sm font-semibold transition-all duration-200 cursor-pointer disabled:opacity-50 shrink-0"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Computing…
                </span>
              ) : 'Run SHAP'}
            </button>
          )}
        </div>

        {error && <ApiError message={error} />}
        {wakingUp && <WakeUpBanner />}

        {!explanation && !loading && (
          <div className="flex flex-col items-center justify-center py-12 text-slate-500">
            <span className="text-5xl mb-4 opacity-75">🔬</span>
            <p className="text-sm">Click <strong className="text-slate-300">Run SHAP</strong> to calculate custom local feature explanations.</p>
          </div>
        )}

        {loading && (
          <div className="flex flex-col items-center justify-center py-12 text-slate-500">
            <span className="text-4xl animate-spin mb-4">⚙️</span>
            <p className="text-sm text-slate-300">Evaluating SHAP tree kernels…</p>
          </div>
        )}

        {chartData && (
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 40, top: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.04)" />
              <XAxis type="number" tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
              <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 12, fill: '#e2e8f0' }} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{ background: '#0b1329', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12, color: '#f8fafc', fontSize: 12 }}
                formatter={(v) => [typeof v === 'number' ? v.toFixed(4) : v, 'SHAP value']}
              />
              <ReferenceLine x={0} stroke="rgba(255,255,255,0.15)" strokeWidth={1.5} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={20}>
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.value >= 0 ? '#f43f5e' : '#10b981'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {explanation && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <FactorCard title="⚠️ Top Risk Factors" items={explanation.top_positive_risk_factors} color="red" />
          <FactorCard title="✅ Protective Factors" items={explanation.protective_factors} color="green" />
        </div>
      )}
    </div>
  );
}

function FactorCard({ title, items, color }: { title: string; items: string[]; color: 'red' | 'green' }) {
  const cls = color === 'red'
    ? { bg: 'bg-rose-500/5', border: 'border-rose-500/10', title: 'text-rose-400', chip: 'bg-rose-500/10 text-rose-300 border border-rose-500/20' }
    : { bg: 'bg-emerald-500/5', border: 'border-emerald-500/10', title: 'text-emerald-400', chip: 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/20' };
  return (
    <div className={`glass-panel p-5 border-t-2 ${cls.border} ${cls.bg}`}>
      <h3 className={`font-bold text-sm mb-4 ${cls.title}`}>{title}</h3>
      {items.length === 0
        ? <p className="text-xs text-slate-500">No parameters identified in this band.</p>
        : <div className="flex flex-wrap gap-2.5">
            {items.map(item => (
              <span key={item} className={`px-2.5 py-1 rounded-lg text-xs font-semibold ${cls.chip}`}>{item}</span>
            ))}
          </div>
      }
    </div>
  );
}

function EmptyState({ icon, message }: { icon: string; message: string }) {
  return (
    <div className="glass-panel p-12 flex flex-col items-center text-center justify-center min-h-[300px]">
      <span className="text-5xl mb-4 opacity-75 filter drop-shadow-[0_0_15px_rgba(139,92,246,0.2)]">{icon}</span>
      <p className="text-slate-400 text-sm max-w-xs">{message}</p>
    </div>
  );
}

function WakeUpBanner() {
  return (
    <div className="flex items-center gap-2.5 text-xs text-amber-400 bg-amber-500/5 border border-amber-500/10 rounded-xl px-4 py-3 mb-5">
      <span className="animate-spin inline-block">⟳</span>
      Waking up the backend server (Render cold-start — up to 30s)…
    </div>
  );
}
