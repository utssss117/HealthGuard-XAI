"use client";

import React from "react";
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from "chart.js";
import { Radar } from "react-chartjs-2";

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

interface HealthRadarProps {
  biomarkers: Record<string, number> | null;
}

const NORMS: Record<string, number> = {
  Pregnancies: 1, 
  Glucose: 100,
  BloodPressure: 80,
  SkinThickness: 20,
  Insulin: 100,
  BMI: 22,
  DiabetesPedigreeFunction: 0.3,
  Age: 40
};

export default function HealthRadar({ biomarkers }: HealthRadarProps) {
  if (!biomarkers) {
    return (
      <section className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-6 sm:p-8 shadow-2xl relative h-full flex flex-col items-center justify-center min-h-[300px]">
        <h2 className="text-xl font-bold text-white mb-2 self-start animate-pulse">Patient Health Profile</h2>
        <p className="text-sm text-[#94a3b8] mb-6 self-start">Waiting for patient data...</p>
      </section>
    );
  }

  const labels = Object.keys(biomarkers);
  
  // Calculate scaled data: (Actual / Normal) * 100
  // Capped at 250% so extreme outliers don't break the chart visual
  const actualData = labels.map(key => {
    const norm = NORMS[key] || 1;
    return Math.min((biomarkers[key] / norm) * 100, 250); 
  });
  
  const baselineData = labels.map(() => 100);

  const data = {
    labels: labels,
    datasets: [
      {
        label: "Patient Values",
        data: actualData,
        backgroundColor: "rgba(239, 68, 68, 0.4)", // transluscent red
        borderColor: "rgba(239, 68, 68, 1)",
        pointBackgroundColor: "rgba(239, 68, 68, 1)",
        borderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
      },
      {
        label: "Healthy Baseline",
        data: baselineData,
        backgroundColor: "rgba(34, 197, 94, 0.2)",
        borderColor: "rgba(34, 197, 94, 0.8)",
        pointBackgroundColor: "rgba(34, 197, 94, 1)",
        borderWidth: 2,
        borderDash: [5, 5],
        pointRadius: 0,
        pointHoverRadius: 0,
      }
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      r: {
        angleLines: { color: "rgba(255, 255, 255, 0.1)" },
        grid: { color: "rgba(255, 255, 255, 0.1)" },
        pointLabels: {
          color: "#94a3b8",
          font: { family: "Inter", size: 10, weight: 500 }
        },
        ticks: {
          display: false,
          min: 0,
          max: 200,
          stepSize: 50
        }
      }
    },
    plugins: {
      legend: {
        labels: { color: "#f8fafc", font: { family: "Inter", size: 12 } },
        position: "bottom" as const,
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            const label = context.dataset.label || '';
            const feature = context.chart.data.labels[context.dataIndex];
            
            if (label === "Healthy Baseline") {
              return `Baseline: ${NORMS[feature]}`;
            } else {
              return `Patient: ${biomarkers[feature]}`;
            }
          }
        }
      }
    }
  };

  return (
    <section className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-6 sm:p-8 shadow-2xl relative flex flex-col h-full min-h-[400px]">
      <h2 className="text-xl font-bold text-white mb-2">Patient Health Profile</h2>
      <p className="text-sm text-[#94a3b8] mb-6">Your biomarkers compared against healthy baselines</p>
      
      <div className="relative w-full flex-grow">
        <Radar data={data} options={options} />
      </div>
    </section>
  );
}
