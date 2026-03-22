"use client";

import React, { useState } from "react";
import { Activity } from "lucide-react";

export interface BiomarkerData {
  Pregnancies: number;
  Glucose: number;
  BloodPressure: number;
  SkinThickness: number;
  Insulin: number;
  BMI: number;
  DiabetesPedigreeFunction: number;
  Age: number;
}

interface PredictionFormProps {
  onSubmit: (data: BiomarkerData) => void;
  loading: boolean;
}

const DEFAULT_DATA: BiomarkerData = {
  Pregnancies: 2,
  Glucose: 138,
  BloodPressure: 72,
  SkinThickness: 35,
  Insulin: 0,
  BMI: 33.6,
  DiabetesPedigreeFunction: 0.627,
  Age: 50,
};

export default function PredictionForm({ onSubmit, loading }: PredictionFormProps) {
  const [formData, setFormData] = useState<BiomarkerData>(DEFAULT_DATA);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: parseFloat(value) || 0,
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <section className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-6 sm:p-8 shadow-2xl relative overflow-hidden transition-all duration-300 hover:shadow-[#6c47ff]/10 hover:border-white/20">
      <h2 className="text-xl font-bold text-white mb-2">Patient Biomarkers</h2>
      <p className="text-sm text-[#94a3b8] mb-6">Enter clinical measurements to generate a risk assessment</p>
      
      <form onSubmit={handleSubmit} className="grid grid-cols-1 sm:grid-cols-2 gap-5">
        {[
          { key: "Pregnancies", label: "Pregnancies", min: 0, max: 20, step: "1" },
          { key: "Glucose", label: "Glucose (mg/dL)", min: 0, max: 300, step: "1" },
          { key: "BloodPressure", label: "Blood Pressure (mmHg)", min: 0, max: 200, step: "1" },
          { key: "SkinThickness", label: "Skin Thickness (mm)", min: 0, max: 100, step: "1" },
          { key: "Insulin", label: "Insulin (IU/mL)", min: 0, max: 900, step: "1" },
          { key: "BMI", label: "BMI", min: 0, max: 70, step: "0.1" },
          { key: "DiabetesPedigreeFunction", label: "Diabetes Pedigree", min: 0, max: 3, step: "0.001" },
          { key: "Age", label: "Age", min: 1, max: 120, step: "1" },
        ].map(({ key, label, min, max, step }) => (
          <div key={key} className="flex flex-col gap-1.5 focus-within:-translate-y-0.5 transition-transform duration-200">
            <label className="text-xs font-semibold text-[#cbd5e1] uppercase tracking-wider ml-1">{label}</label>
            <input
              type="number"
              name={key}
              value={formData[key as keyof BiomarkerData]}
              onChange={handleChange}
              min={min}
              max={max}
              step={step}
              required
              className="w-full bg-[#0f172a] border border-[#334155] rounded-xl px-4 py-2.5 text-white font-medium focus:outline-none focus:border-[#6c47ff] focus:ring-2 focus:ring-[#6c47ff]/30 transition-all hover:border-[#475569]"
            />
          </div>
        ))}

        <div className="col-span-1 sm:col-span-2 mt-4">
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-[#6c47ff] to-[#8b5cf6] text-white rounded-xl py-3.5 px-6 font-bold text-base flex items-center justify-center gap-2 hover:shadow-lg hover:shadow-[#6c47ff]/40 cursor-pointer transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed transform hover:-translate-y-1 active:translate-y-0"
          >
            {loading ? (
              <div className="flex gap-1.5 items-center">
                <span className="w-2 h-2 bg-white rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                <span className="w-2 h-2 bg-white rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                <span className="w-2 h-2 bg-white rounded-full animate-bounce"></span>
              </div>
            ) : (
              <>
                <Activity size={20} />
                Analyze Risk
              </>
            )}
          </button>
        </div>
      </form>
    </section>
  );
}
