'use client';

import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts';

// Hardcoded from outputs/metrics/model_comparison.csv
const MODELS = [
  { name: 'Logistic Regression', accuracy: 0.7143, precision: 0.6087, recall: 0.5185, f1: 0.56,    roc: 0.823,  brier: 0.1679, cv: 0.8334 },
  { name: 'Random Forest',       accuracy: 0.7597, precision: 0.6809, recall: 0.5926, f1: 0.6337,  roc: 0.8147, brier: 0.1652, cv: 0.8120 },
  { name: 'XGBoost',             accuracy: 0.7338, precision: 0.6226, recall: 0.6111, f1: 0.6168,  roc: 0.8052, brier: 0.2025, cv: 0.7771 },
  { name: 'MLP',                 accuracy: 0.6104, precision: 0.2857, recall: 0.0741, f1: 0.1176,  roc: 0.3748, brier: 0.2469, cv: 0.7536 },
  { name: 'Stacking Ensemble',   accuracy: 0.7403, precision: 0.6667, recall: 0.5185, f1: 0.5833,  roc: 0.803,  brier: 0.1731, cv: 0.8137 },
];

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

const METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc'] as const;
type Metric = typeof METRICS[number];
const METRIC_LABELS: Record<Metric, string> = {
  accuracy: 'Accuracy', precision: 'Precision', recall: 'Recall', f1: 'F1-Score', roc: 'ROC-AUC',
};

// Bar chart data per metric
function barData(metric: Metric) {
  return MODELS.map((m, i) => ({
    name: m.name.replace(' ', '\n'),
    shortName: m.name.split(' ')[0],
    value: parseFloat((m[metric] * 100).toFixed(1)),
    fill: COLORS[i],
  }));
}

// Radar data for a given model
function radarData(m: typeof MODELS[number]) {
  return [
    { metric: 'Accuracy',  value: m.accuracy  * 100 },
    { metric: 'Precision', value: m.precision * 100 },
    { metric: 'Recall',    value: m.recall    * 100 },
    { metric: 'F1-Score',  value: m.f1        * 100 },
    { metric: 'ROC-AUC',   value: m.roc       * 100 },
    { metric: 'CV AUC',    value: m.cv        * 100 },
  ];
}

export default function ModelAnalyticsTab() {
  const bestRoc = MODELS.reduce((a, b) => a.roc > b.roc ? a : b);

  return (
    <div className="space-y-6">
      {/* Summary banner */}
      <div className="bg-blue-500/5 border border-blue-500/10 rounded-xl p-5 flex flex-wrap gap-6 items-center shadow-inner">
        <div>
          <p className="text-[10px] text-blue-400 font-bold uppercase tracking-widest">Optimal Pipeline Profile</p>
          <p className="text-xl font-black text-white">{bestRoc.name}</p>
        </div>
        <div className="h-8 w-px bg-slate-800 hidden sm:block" />
        {(['accuracy','precision','recall','f1','roc'] as Metric[]).map(m => (
          <div key={m} className="text-center sm:text-left">
            <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-wider">{METRIC_LABELS[m]}</p>
            <p className="text-base font-extrabold text-blue-400">{(bestRoc[m] * 100).toFixed(1)}%</p>
          </div>
        ))}
      </div>

      {/* Metric comparison bar charts */}
      <div className="glass-panel p-6">
        <h2 className="font-bold text-white text-lg mb-6">Model Comparison Analysis</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {METRICS.map(metric => (
            <div key={metric} className="border border-slate-900 bg-slate-950/20 rounded-xl p-4">
              <h3 className="text-xs font-semibold text-slate-300 mb-4 uppercase tracking-wider">{METRIC_LABELS[metric]} (%)</h3>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={barData(metric)} margin={{ left: -10, right: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.03)" />
                  <XAxis dataKey="shortName" tick={{ fontSize: 10, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 9, fill: '#475569' }} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{ background: '#0b1329', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12, color: '#f8fafc', fontSize: 12 }}
                    formatter={(v) => [typeof v === 'number' ? `${v}%` : v, METRIC_LABELS[metric]]}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={28}>
                    {barData(metric).map((entry, i) => <Cell key={i} fill={COLORS[i]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
      </div>

      {/* Radar charts per model */}
      <div className="glass-panel p-6">
        <h2 className="font-bold text-white text-lg mb-6">Pipeline Parameter Radar</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
          {MODELS.map((m, i) => (
            <div key={m.name} className="text-center border border-slate-900 bg-slate-950/20 rounded-xl p-3 flex flex-col justify-between">
              <p className="text-[10px] font-bold text-slate-300 uppercase tracking-wide mb-3">{m.name}</p>
              <ResponsiveContainer width="100%" height={140}>
                <RadarChart data={radarData(m)}>
                  <PolarGrid stroke="rgba(255,255,255,0.04)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 8, fill: '#475569' }} />
                  <Radar dataKey="value" stroke={COLORS[i]} fill={COLORS[i]} fillOpacity={0.2} />
                </RadarChart>
              </ResponsiveContainer>
              <p className="text-[10px] text-slate-500 font-semibold tracking-wider uppercase mt-3">ROC-AUC: {(m.roc * 100).toFixed(1)}%</p>
            </div>
          ))}
        </div>
      </div>

      {/* Full data table */}
      <div className="glass-panel p-6 overflow-x-auto">
        <h2 className="font-bold text-white text-lg mb-6">Aggregated Metrics Table</h2>
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 font-bold uppercase tracking-wider">
              <th className="text-left py-3 pr-4 font-bold">Model</th>
              <th className="text-right py-3 px-3 font-bold">Accuracy</th>
              <th className="text-right py-3 px-3 font-bold">Precision</th>
              <th className="text-right py-3 px-3 font-bold">Recall</th>
              <th className="text-right py-3 px-3 font-bold">F1</th>
              <th className="text-right py-3 px-3 font-bold">ROC-AUC</th>
              <th className="text-right py-3 px-3 font-bold">Brier↓</th>
              <th className="text-right py-3 pl-3 font-bold">CV AUC</th>
            </tr>
          </thead>
          <tbody>
            {MODELS.map((m, i) => (
              <tr key={m.name} className={`border-b border-slate-900/60 transition-colors duration-150 hover:bg-white/[0.02] ${m.name === bestRoc.name ? 'bg-blue-500/5 font-semibold text-blue-400 border-y border-blue-500/10' : ''}`}>
                <td className="py-3 pr-4 text-slate-300 flex items-center gap-2.5">
                  <span className="w-2 h-2 rounded-full inline-block shrink-0" style={{ background: COLORS[i] }} />
                  <span>{m.name}</span>
                  {m.name === bestRoc.name && <span className="text-[9px] font-bold text-blue-400 bg-blue-500/10 border border-blue-500/20 px-1.5 py-0.5 rounded-md uppercase tracking-wider">★ Best</span>}
                </td>
                {(['accuracy','precision','recall','f1','roc','brier','cv'] as (keyof typeof m)[]).map(col => (
                  <td key={col} className={`text-right py-3 px-3 font-mono ${m.name === bestRoc.name ? 'text-blue-300' : 'text-slate-400'}`}>
                    {typeof m[col] === 'number' ? (m[col] as number).toFixed(4) : m[col]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
