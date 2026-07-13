'use client';

import { useEffect, useState } from 'react';
import { useAuth } from '@clerk/nextjs';
import { getRecommendations, type Biomarkers, type PredictResponse, type ExplainResponse, type Recommendation } from '@/lib/api';
import ApiError from '@/components/ui/ApiError';

type Props = {
  biomarkers: Biomarkers | null;
  prediction: PredictResponse | null;
  explanation: ExplainResponse | null;
};

const PRIORITY_STYLE: Record<string, string> = {
  HIGH:   'bg-rose-500/10 text-rose-400 border-rose-500/20',
  MEDIUM: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  LOW:    'bg-blue-500/10 text-blue-400 border-blue-500/20',
};

const CATEGORY_ICON: Record<string, string> = {
  diet: '🥗', exercise: '🏃', weight: '⚖️', monitoring: '📊',
  medication: '💊', medical: '🏥', lifestyle: '🌿', sleep: '😴',
};

function categoryIcon(cat?: string) {
  if (!cat) return '💡';
  const key = cat.toLowerCase();
  for (const [k, v] of Object.entries(CATEGORY_ICON)) {
    if (key.includes(k)) return v;
  }
  return '💡';
}

export default function RecommendationsTab({ biomarkers, prediction, explanation }: Props) {
  const { getToken } = useAuth();
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [wellness, setWellness] = useState<string[]>([]);
  const [disclaimer, setDisclaimer] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fetched, setFetched] = useState(false);

  useEffect(() => {
    if (!biomarkers || !prediction || fetched) return;

    setLoading(true);
    setError(null);

    const loadData = async () => {
      try {
        const token = await getToken();
        const data = await getRecommendations({
          biomarkers,
          predicted_risks: { diabetes: prediction.risk_probability },
          top_positive_risk_factors: explanation?.top_positive_risk_factors ?? [],
          protective_factors: explanation?.protective_factors ?? [],
          use_llm: false,
        }, token);
        setRecs(data.prioritized_recommendations);
        setWellness(data.general_wellness_advice);
        setDisclaimer(data.disclaimer);
        setFetched(true);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, [biomarkers, prediction, explanation, fetched, getToken]);

  if (!prediction || !biomarkers) {
    return (
      <div className="glass-panel p-12 flex flex-col items-center text-center justify-center min-h-[300px]">
        <span className="text-5xl mb-4 opacity-75 filter drop-shadow-[0_0_15px_rgba(59,130,246,0.2)]">💊</span>
        <p className="text-slate-400 text-sm max-w-xs">Run a <strong className="text-slate-300">Risk Assessment</strong> first to generate personalized recommendations.</p>
      </div>
    );
  }

  const riskColor = prediction.risk_level === 'Low' ? 'text-emerald-400' : prediction.risk_level === 'Medium' ? 'text-amber-400' : 'text-rose-400';

  return (
    <div className="space-y-6">
      <div className="glass-panel p-6">
        <div className="flex justify-between items-start mb-6">
          <div>
            <h2 className="font-bold text-white text-lg">Personalized Recommendations</h2>
            <p className="text-slate-400 text-sm mt-1">
              Evidence-based clinical guidelines tailored to your <strong className={riskColor}>{prediction.risk_level.toLowerCase()} risk</strong> profile.
            </p>
          </div>
          {fetched && (
            <button
              onClick={() => { setFetched(false); setRecs([]); }}
              className="text-xs text-blue-400 hover:text-blue-300 font-semibold transition-colors cursor-pointer"
            >
              🔄 Refresh
            </button>
          )}
        </div>

        {error && <ApiError message={error} />}

        {loading && (
          <div className="flex flex-col items-center py-12 text-slate-500">
            <span className="text-4xl animate-spin mb-4">⚙️</span>
            <p className="text-sm text-slate-300">Formulating recommendations…</p>
          </div>
        )}

        {!loading && recs.length === 0 && !error && (
          <div className="flex flex-col items-center py-12 text-slate-500">
            <span className="text-4xl mb-3">📋</span>
            <p className="text-sm">Retrieving profile…</p>
          </div>
        )}

        {recs.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {recs.map((rec, i) => (
              <div key={i} className="glass-panel p-5 border border-slate-800/80 hover:border-slate-700/60 transition-all duration-300">
                <div className="flex items-start gap-4.5">
                  <span className="text-3xl mt-0.5 filter drop-shadow-[0_0_8px_rgba(255,255,255,0.08)]">{categoryIcon(rec.category)}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2 flex-wrap">
                      <h3 className="text-sm font-bold text-white leading-snug">{rec.recommendation || 'Recommendation'}</h3>
                      <span className={`px-2.5 py-0.5 rounded-lg text-[10px] font-bold border uppercase tracking-wider ${PRIORITY_STYLE[(rec.priority || 'LOW').toUpperCase()] ?? 'bg-slate-800 text-slate-400 border-slate-700'}`}>
                        {rec.priority || 'LOW'}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed mb-3">{rec.rationale}</p>
                    <span className="inline-block text-[10px] font-bold text-blue-400 bg-blue-500/10 border border-blue-500/20 px-2 py-0.5 rounded-lg uppercase tracking-wider">{rec.category}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {wellness.length > 0 && (
        <div className="bg-blue-950/10 border border-blue-500/10 rounded-xl p-5 shadow-inner">
          <h3 className="font-bold text-blue-400 text-sm mb-3">🌿 General Wellness Advice</h3>
          <ul className="space-y-2">
            {wellness.map((tip, i) => (
              <li key={i} className="text-xs text-blue-300/90 flex gap-2.5 items-start">
                <span className="text-blue-500 shrink-0 select-none">•</span>
                <span>{tip}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {disclaimer && (
        <p className="text-xs text-slate-500 italic border-t border-slate-900 pt-4 leading-relaxed">{disclaimer}</p>
      )}
    </div>
  );
}
