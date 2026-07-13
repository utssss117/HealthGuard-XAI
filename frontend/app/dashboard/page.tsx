'use client';

import { useState } from 'react';
import type { Biomarkers, PredictResponse, ExplainResponse } from '@/lib/api';
import RiskAssessmentTab from '@/components/tabs/RiskAssessment';
import ExplainabilityTab from '@/components/tabs/Explainability';
import CounterfactualTab from '@/components/tabs/Counterfactual';
import RecommendationsTab from '@/components/tabs/Recommendations';
import AIAssistantTab from '@/components/tabs/AIAssistant';
import ModelAnalyticsTab from '@/components/tabs/ModelAnalytics';

const TABS = [
  { id: 'risk',       label: 'Risk Assessment',  icon: '🔬' },
  { id: 'explain',    label: 'Explainability',    icon: '🧠' },
  { id: 'counterfact',label: 'What-If Analysis',  icon: '🔄' },
  { id: 'recs',       label: 'Recommendations',   icon: '💊' },
  { id: 'chat',       label: 'AI Assistant',      icon: '💬' },
  { id: 'analytics',  label: 'Model Analytics',   icon: '📊' },
] as const;

type TabId = typeof TABS[number]['id'];

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<TabId>('risk');

  // Shared state — prediction result flows into other tabs
  const [biomarkers, setBiomarkers] = useState<Biomarkers | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);

  const handlePrediction = (bm: Biomarkers, pred: PredictResponse) => {
    setBiomarkers(bm);
    setPrediction(pred);
    setExplanation(null); // reset explanation when new assessment is run
  };

  return (
    <div className="space-y-8">
      {/* Tab bar */}
      <div className="flex gap-1.5 bg-slate-900/40 border border-slate-800/80 rounded-xl p-1.5 overflow-x-auto backdrop-blur-md shadow-inner">
        {TABS.map(tab => (
          <button
            key={tab.id}
            id={`tab-${tab.id}`}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold whitespace-nowrap transition-all duration-200 cursor-pointer
              ${activeTab === tab.id
                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-[0_4px_15px_rgba(59,130,246,0.25)] border border-blue-500/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.03] border border-transparent'
              }`}
          >
            <span className="text-base">{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Tab panels with transition effect */}
      <div className="animate-[fadeIn_0.3s_ease-out] transition-all duration-300">
        {activeTab === 'risk' && (
          <RiskAssessmentTab onResult={handlePrediction} prediction={prediction} />
        )}
        {activeTab === 'explain' && (
          <ExplainabilityTab
            biomarkers={biomarkers}
            prediction={prediction}
            explanation={explanation}
            onExplanation={setExplanation}
          />
        )}
        {activeTab === 'counterfact' && (
          <CounterfactualTab biomarkers={biomarkers} prediction={prediction} />
        )}
        {activeTab === 'recs' && (
          <RecommendationsTab biomarkers={biomarkers} prediction={prediction} explanation={explanation} />
        )}
        {activeTab === 'chat' && (
          <AIAssistantTab biomarkers={biomarkers} prediction={prediction} />
        )}
        {activeTab === 'analytics' && (
          <ModelAnalyticsTab />
        )}
      </div>
    </div>
  );
}
