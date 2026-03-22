"use client";

import React, { useEffect, useState } from "react";

interface FeatureChartProps {
  features: Record<string, number>;
}

export default function FeatureChart({ features }: FeatureChartProps) {
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    // Trigger animation shortly after mount
    const t = setTimeout(() => setAnimated(true), 100);
    return () => clearTimeout(t);
  }, [features]);

  const entries = Object.entries(features).slice(0, 8);
  const maxVal = Math.max(...entries.map(([, val]) => Math.abs(val)), 0.001);

  return (
    <section className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-6 sm:p-8 shadow-2xl relative transition-all hover:border-white/20">
      <h2 className="text-xl font-bold text-white mb-2">Feature Importance</h2>
      <p className="text-sm text-[#94a3b8] mb-6">Contribution of each biomarker to the risk score</p>
      
      <div className="flex flex-col gap-4 overflow-hidden">
        {entries.map(([name, val], index) => {
          const pct = (Math.abs(val) / maxVal) * 100;
          return (
            <div key={name} className="flex items-center w-full gap-3 group">
              <span className="w-1/3 text-xs font-medium text-[#cbd5e1] truncate group-hover:text-white transition-colors" title={name}>
                {name}
              </span>
              <div className="flex-1 bg-white/5 h-2.5 rounded-full overflow-hidden border border-white/5 shadow-inner">
                <div
                  className="h-full bg-gradient-to-r from-[#6c47ff] to-[#8b5cf6] rounded-full transition-all duration-1000 ease-out"
                  style={{ width: animated ? `${pct}%` : "0%" }}
                />
              </div>
              <span className="w-12 text-right text-xs font-mono font-bold text-[#f8fafc]">
                {Number(val).toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}
