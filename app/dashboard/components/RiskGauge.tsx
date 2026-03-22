"use client";

import React, { useEffect, useState } from "react";

interface RiskGaugeProps {
  probability: number;
  riskLevel: string;
}

export default function RiskGauge({ probability, riskLevel }: RiskGaugeProps) {
  const [rotation, setRotation] = useState(-90);
  const [offset, setOffset] = useState(251.2);
  const [color, setColor] = useState("#22c55e");
  const [animatedProb, setAnimatedProb] = useState(0);

  useEffect(() => {
    // Math logic mapped exactly from the Canvas app:
    const totalCircumference = 251.2;
    const targetOffset = totalCircumference - totalCircumference * probability;
    const targetAngle = -90 + probability * 180;
    
    let targetColor = "#ef4444"; // high
    if (probability <= 0.33) targetColor = "#22c55e"; // low
    else if (probability <= 0.66) targetColor = "#f59e0b"; // medium

    // A tiny timeout forces a repaint so the CSS transition triggers
    const t = setTimeout(() => {
      setOffset(targetOffset);
      setRotation(targetAngle);
      setColor(targetColor);
      setAnimatedProb(Math.round(probability * 100));
    }, 50);

    return () => clearTimeout(t);
  }, [probability]);

  // Dynamic class for the badge based on level
  const badgeClasses = {
    Low: "bg-green-500/10 text-green-400 border border-green-500/20",
    Medium: "bg-amber-500/10 text-amber-400 border border-amber-500/20",
    High: "bg-red-500/10 text-red-500 border border-red-500/20"
  }[riskLevel] || "bg-gray-500/10 text-gray-400 border border-gray-500/20";

  return (
    <section className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-6 sm:p-8 shadow-2xl relative flex flex-col items-center justify-between transition-all hover:border-white/20">
      <h2 className="text-xl font-bold text-white w-full text-left mb-6">Risk Assessment</h2>
      
      <div className="relative w-48 h-[110px] mb-8">
        <svg viewBox="0 0 200 110" className="w-full h-full drop-shadow-lg overflow-visible">
          <defs>
            <linearGradient id="gauge-grad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#22c55e" />
              <stop offset="50%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>
          </defs>
          
          {/* Background Track */}
          <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#1e293b" strokeWidth="18" strokeLinecap="round" />
          
          {/* Active Arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="url(#gauge-grad)"
            strokeWidth="18"
            strokeLinecap="round"
            strokeDasharray="251.2"
            strokeDashoffset={offset}
            className="transition-all duration-1000 ease-out"
          />
          
          {/* Needle pivot mechanism */}
          <g style={{ transform: `rotate(${rotation}deg)`, transformOrigin: "100px 100px" }} className="transition-all duration-1000 ease-[cubic-bezier(0.34,1.56,0.64,1)]">
            <line x1="100" y1="100" x2="100" y2="30" stroke={color} strokeWidth="4" strokeLinecap="round" className="transition-colors duration-1000" />
            <circle cx="100" cy="100" r="8" fill="#f8fafc" className="shadow-lg" />
            <circle cx="100" cy="100" r="3" fill={color} className="transition-colors duration-1000" />
          </g>
        </svg>

        {/* Floating Percentage */}
        <div className="absolute -bottom-4 left-0 w-full flex justify-center text-3xl font-extrabold text-white">
          {animatedProb}%
        </div>
      </div>

      <div className="w-full flex justify-between text-xs font-bold uppercase tracking-wider px-2 text-[#64748b] mb-6">
        <span>Low</span>
        <span>Med</span>
        <span>High</span>
      </div>

      <div className={`px-5 py-2 rounded-full font-bold text-sm tracking-wide shadow-inner ${badgeClasses}`}>
        {riskLevel} Risk
      </div>
    </section>
  );
}
