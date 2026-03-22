"use client";

import React, { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

interface RecommendationsProps {
  authHeaders: Record<string, string>;
  biomarkers: Record<string, number> | null;
  riskProbability: number | null;
}

export default function Recommendations({ authHeaders, biomarkers, riskProbability }: RecommendationsProps) {
  const [loading, setLoading] = useState(false);
  const [recs, setRecs] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!biomarkers || riskProbability === null) return;

    const fetchRecs = async () => {
      setLoading(true);
      setError(null);
      
      const payload = {
        biomarkers,
        predicted_risks: { diabetes: riskProbability, heart_disease: riskProbability * 0.8 },
        top_positive_risk_factors: [],
        protective_factors: [],
        use_llm: false,
      };

      try {
        const res = await fetch("/api/recommend", {
          method: "POST",
          headers: { "Content-Type": "application/json", ...authHeaders },
          body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Failed to load recommendations.");
        const data = await res.json();
        setRecs(data.prioritized_recommendations || []);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchRecs();
  }, [biomarkers, riskProbability, authHeaders]);

  if (!biomarkers) {
    return null; // Don't show anything if no prediction run yet
  }

  return (
    <section className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-6 sm:p-8 shadow-2xl relative transition-all hover:border-white/20">
      <h2 className="text-xl font-bold text-white mb-2">Recommendations</h2>
      <p className="text-sm text-[#94a3b8] mb-6">Personalized lifestyle guidance based on your risk profile</p>

      {loading ? (
        <div className="flex flex-col items-center justify-center p-8 gap-3 text-[#94a3b8] animate-pulse">
          <Loader2 className="animate-spin text-[#6c47ff]" size={32} />
          <span className="text-sm font-medium">Generating recommendations…</span>
        </div>
      ) : error ? (
        <div className="p-4 bg-red-500/10 border border-red-500/20 text-red-400 rounded-xl text-sm font-medium">
          {error}
        </div>
      ) : recs.length === 0 ? (
        <div className="p-4 bg-white/5 border border-white/10 rounded-xl text-[#94a3b8] text-sm text-center">
          No specific recommendations generated.
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          {recs.slice(0, 5).map((rec, i) => (
            <div 
              key={i} 
              className="group flex gap-4 p-4 rounded-2xl bg-white/5 border border-white/5 hover:bg-white/10 hover:border-white/10 transition-all cursor-default animate-in slide-in-from-bottom-2 duration-500 ease-out fill-mode-both"
              style={{ animationDelay: `${i * 100}ms` }}
            >
              <div className="w-8 h-8 rounded-full bg-[#1e293b] border border-[#334155] flex items-center justify-center text-xs font-bold text-[#64748b] shrink-0 group-hover:text-[#6c47ff] group-hover:border-[#6c47ff]/50 transition-colors">
                #{i + 1}
              </div>
              <div className="flex flex-col gap-2 pt-1 text-sm text-[#e2e8f0] leading-relaxed">
                {rec.recommendation}
                <span className="inline-block w-max px-2 py-0.5 rounded text-[10px] uppercase font-bold tracking-wider bg-white/10 text-[#94a3b8]">
                  {rec.related_risk || "General"}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
