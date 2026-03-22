"use client";

import React, { useState } from "react";
import PredictionForm, { BiomarkerData } from "./PredictionForm";
import RiskGauge from "./RiskGauge";
import FeatureChart from "./FeatureChart";
import HealthRadar from "./HealthRadar";
import Recommendations from "./Recommendations";
import ChatAssistant from "./ChatAssistant";
import { Activity, MessageSquare } from "lucide-react";

interface DashboardClientProps {
  authToken: string;
}

export default function DashboardClient({ authToken }: DashboardClientProps) {
  const [mounted, setMounted] = useState(false);
  const [activeTab, setActiveTab] = useState<"predict" | "chat">("predict");
  const [loading, setLoading] = useState(false);
  const [syncStatus, setSyncStatus] = useState("Syncing with clinical engine...");
  const [biomarkers, setBiomarkers] = useState<BiomarkerData | null>(null);
  const [prediction, setPrediction] = useState<any>(null);

  // Mount check to prevent hydration errors
  React.useEffect(() => {
    setMounted(true);
  }, []);

  // Auto-sync on load
  React.useEffect(() => {
    if (!mounted) return;
    const sync = async () => {
      try {
        const res = await fetch("/api/auth/me", {
          headers: authHeaders
        });
        if (res.ok) {
          const data = await res.json();
          setSyncStatus(`Connected: Patient ID #${data.id}`);
        } else {
          setSyncStatus("Clinical Sync Failed");
        }
      } catch (e) {
        setSyncStatus("Backend Engine Offline");
      }
    };
    sync();
  }, [authToken]);

  const handlePredict = async (data: BiomarkerData) => {
    setLoading(true);
    setBiomarkers(data);
    
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders },
        body: JSON.stringify(data),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed");
      }

      const result = await res.json();
      setPrediction(result);
      
      // Smooth scroll to results
      setTimeout(() => {
        window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
      }, 100);

    } catch (err: any) {
      alert(`API Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  if (!mounted) return (
    <div className="min-h-[400px] flex items-center justify-center">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[#6c47ff]"></div>
    </div>
  );

  const authHeaders = { Authorization: `Bearer ${authToken}` };

  return (
    <div className="max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
      {/* Sync Status Header Info */}
      <div className="flex justify-end -mt-20 mb-8">
        <span className={`px-4 py-1.5 rounded-full text-xs font-bold tracking-wider shadow-sm border ${
          syncStatus.includes("Connected") 
          ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" 
          : "bg-amber-500/10 text-amber-500 border-amber-500/20"
        }`}>
          {syncStatus}
        </span>
      </div>

      {/* Tab Navigation */}
      <div className="flex justify-center gap-4 mb-8">
        <button
          onClick={() => setActiveTab("predict")}
          className={`flex items-center gap-2 px-6 py-3 rounded-full font-bold text-sm transition-all ${
            activeTab === "predict"
              ? "bg-[#6c47ff] text-white shadow-lg shadow-[#6c47ff]/30 ring-2 ring-[#6c47ff] ring-offset-2 ring-offset-[#020617]"
              : "bg-white/5 text-[#94a3b8] hover:text-white hover:bg-white/10 border border-white/5"
          }`}
        >
          <Activity size={18} />
          Risk Predictor
        </button>
        <button
          onClick={() => setActiveTab("chat")}
          className={`flex items-center gap-2 px-6 py-3 rounded-full font-bold text-sm transition-all ${
            activeTab === "chat"
              ? "bg-[#6c47ff] text-white shadow-lg shadow-[#6c47ff]/30 ring-2 ring-[#6c47ff] ring-offset-2 ring-offset-[#020617]"
              : "bg-white/5 text-[#94a3b8] hover:text-white hover:bg-white/10 border border-white/5"
          }`}
        >
          <MessageSquare size={18} />
          AI Assistant
        </button>
      </div>

      {activeTab === "predict" && (
        <div className="space-y-8">
          <PredictionForm onSubmit={handlePredict} loading={loading} />

          {/* Results Grid */}
          {prediction && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-in slide-in-from-bottom-8 duration-700 fade-in">
              <div className="col-span-1 border border-white/0 rounded-3xl overflow-hidden">
                <RiskGauge probability={prediction.risk_probability} riskLevel={prediction.risk_level} />
              </div>
              
              <div className="col-span-1 md:col-span-1 lg:col-span-2">
                <FeatureChart features={prediction.top_features} />
              </div>

              <div className="col-span-1 md:col-span-2 lg:col-span-2">
                <HealthRadar biomarkers={biomarkers as unknown as Record<string, number>} />
              </div>

              <div className="col-span-1 md:col-span-2 lg:col-span-1">
                <Recommendations 
                  authHeaders={authHeaders} 
                  biomarkers={biomarkers as unknown as Record<string, number>} 
                  riskProbability={prediction.risk_probability} 
                />
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === "chat" && (
        <div className="animate-in slide-in-from-bottom-4 duration-500 fade-in max-w-4xl mx-auto">
          <ChatAssistant 
            authHeaders={authHeaders} 
            patientData={prediction ? {
              predicted_risks: { 
                diabetes: prediction.risk_probability, 
                heart_disease: prediction.risk_probability * 0.8 
              },
              risk_level: prediction.risk_level,
              top_risk_factors: Object.keys(prediction.top_features).slice(0, 3),
              protective_factors: [],
              patient_profile: biomarkers,
            } : null}
          />
        </div>
      )}
    </div>
  );
}
