"""
utils.py
────────
Synthetic patient simulation and evaluation framework for RW-AWRM.

Responsibilities:
  1. generate_synthetic_patients()   – 10 diverse patient profiles
  2. run_evaluation()                – Full pipeline over all patients
  3. compare_rule_vs_hybrid()        – Side-by-side priority score comparison
  4. validate_ranking_consistency()  – Assert monotonically decreasing scores
  5. print_patient_report()          – Formatted console output per patient
"""

import json
from typing import List, Dict, Any, Tuple
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Patient Profiles
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_patients() -> List[Dict[str, Any]]:
    """
    Returns 10 clinically realistic synthetic patient records spanning
    a range of risk profiles (low, moderate, high) and demographics.
    """
    patients = [
        {
            "patient_id": "P01",
            "label": "Young Low-Risk Male",
            "predicted_risks": {"diabetes": 0.10, "heart_disease": 0.08},
            "top_positive_risk_factors": ["BMI (26.5)", "Glucose (102.0)"],
            "protective_factors": ["Age (23.0)", "BloodPressure (75.0)"],
            "patient_profile": {"Age": 23, "BMI": 26.5, "Glucose": 102, "BloodPressure": 75, "Cholesterol": 185, "Insulin": 12, "DiabetesPedigreeFunction": 0.2, "Pregnancies": 0},
        },
        {
            "patient_id": "P02",
            "label": "Middle-Aged Pre-Diabetic Female",
            "predicted_risks": {"diabetes": 0.52, "heart_disease": 0.30},
            "top_positive_risk_factors": ["Glucose (118.0)", "BMI (31.0)", "Insulin (28.5)"],
            "protective_factors": ["Cholesterol (195.0)"],
            "patient_profile": {"Age": 44, "BMI": 31.0, "Glucose": 118, "BloodPressure": 85, "Cholesterol": 195, "Insulin": 28, "DiabetesPedigreeFunction": 0.55, "Pregnancies": 2},
        },
        {
            "patient_id": "P03",
            "label": "Elderly High-Risk Hypertensive Male",
            "predicted_risks": {"diabetes": 0.40, "heart_disease": 0.78},
            "top_positive_risk_factors": ["BloodPressure (152.0)", "Cholesterol (248.0)", "Age (68.0)"],
            "protective_factors": ["BMI (24.1)"],
            "patient_profile": {"Age": 68, "BMI": 24.1, "Glucose": 105, "BloodPressure": 152, "Cholesterol": 248, "Insulin": 16, "DiabetesPedigreeFunction": 0.30, "Pregnancies": 0},
        },
        {
            "patient_id": "P04",
            "label": "Obese Diabetic Female, Multi-Comorbid",
            "predicted_risks": {"diabetes": 0.85, "heart_disease": 0.72},
            "top_positive_risk_factors": ["Glucose (142.0)", "BMI (38.5)", "BloodPressure (138.0)", "Cholesterol (243.0)"],
            "protective_factors": [],
            "patient_profile": {"Age": 55, "BMI": 38.5, "Glucose": 142, "BloodPressure": 138, "Cholesterol": 243, "Insulin": 45, "DiabetesPedigreeFunction": 0.82, "Pregnancies": 4},
        },
        {
            "patient_id": "P05",
            "label": "Middle-Aged Borderline Cholesterol",
            "predicted_risks": {"diabetes": 0.18, "heart_disease": 0.41},
            "top_positive_risk_factors": ["Cholesterol (222.0)", "Age (48.0)"],
            "protective_factors": ["Glucose (92.0)", "BMI (23.8)"],
            "patient_profile": {"Age": 48, "BMI": 23.8, "Glucose": 92, "BloodPressure": 118, "Cholesterol": 222, "Insulin": 10, "DiabetesPedigreeFunction": 0.18, "Pregnancies": 0},
        },
        {
            "patient_id": "P06",
            "label": "Young Overweight Pre-Hypertensive",
            "predicted_risks": {"diabetes": 0.28, "heart_disease": 0.34},
            "top_positive_risk_factors": ["BMI (29.1)", "BloodPressure (126.0)"],
            "protective_factors": ["Age (31.0)", "Glucose (95.0)"],
            "patient_profile": {"Age": 31, "BMI": 29.1, "Glucose": 95, "BloodPressure": 126, "Cholesterol": 198, "Insulin": 14, "DiabetesPedigreeFunction": 0.34, "Pregnancies": 1},
        },
        {
            "patient_id": "P07",
            "label": "Elderly Metabolic Syndrome Profile",
            "predicted_risks": {"diabetes": 0.68, "heart_disease": 0.65},
            "top_positive_risk_factors": ["Glucose (131.0)", "BloodPressure (141.0)", "BMI (33.2)", "Cholesterol (238.0)"],
            "protective_factors": [],
            "patient_profile": {"Age": 63, "BMI": 33.2, "Glucose": 131, "BloodPressure": 141, "Cholesterol": 238, "Insulin": 36, "DiabetesPedigreeFunction": 0.61, "Pregnancies": 3},
        },
        {
            "patient_id": "P08",
            "label": "Young Healthy Baseline",
            "predicted_risks": {"diabetes": 0.05, "heart_disease": 0.04},
            "top_positive_risk_factors": [],
            "protective_factors": ["BMI (21.5)", "Glucose (88.0)", "BloodPressure (72.0)"],
            "patient_profile": {"Age": 26, "BMI": 21.5, "Glucose": 88, "BloodPressure": 72, "Cholesterol": 172, "Insulin": 8, "DiabetesPedigreeFunction": 0.10, "Pregnancies": 0},
        },
        {
            "patient_id": "P09",
            "label": "Insulin-Resistant Middle-Aged Female",
            "predicted_risks": {"diabetes": 0.61, "heart_disease": 0.38},
            "top_positive_risk_factors": ["Insulin (52.0)", "Glucose (124.0)", "BMI (30.8)"],
            "protective_factors": ["BloodPressure (88.0)"],
            "patient_profile": {"Age": 41, "BMI": 30.8, "Glucose": 124, "BloodPressure": 88, "Cholesterol": 208, "Insulin": 52, "DiabetesPedigreeFunction": 0.73, "Pregnancies": 5},
        },
        {
            "patient_id": "P10",
            "label": "Older Adult High Cardiovascular Risk",
            "predicted_risks": {"diabetes": 0.32, "heart_disease": 0.80},
            "top_positive_risk_factors": ["Cholesterol (255.0)", "BloodPressure (158.0)", "Age (72.0)"],
            "protective_factors": ["BMI (22.8)", "Glucose (96.0)"],
            "patient_profile": {"Age": 72, "BMI": 22.8, "Glucose": 96, "BloodPressure": 158, "Cholesterol": 255, "Insulin": 11, "DiabetesPedigreeFunction": 0.23, "Pregnancies": 0},
        },
    ]
    return patients


# ──────────────────────────────────────────────────────────────────────────────
# 2. Full Evaluation Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(patients: List[Dict[str, Any]], use_llm: bool = False) -> List[Dict[str, Any]]:
    """
    Run hybrid_recommender.generate_recommendations() on all patients.
    Returns list of (patient_id, label, output) tuples as dicts.
    """
    from phase3_recommendation_engine.hybrid_recommender import generate_recommendations

    results = []
    for p in patients:
        output = generate_recommendations(
            predicted_risks=p["predicted_risks"],
            top_positive_risk_factors=p["top_positive_risk_factors"],
            protective_factors=p["protective_factors"],
            patient_profile=p["patient_profile"],
            use_llm=use_llm,
        )
        results.append({
            "patient_id": p["patient_id"],
            "label": p["label"],
            "output": output,
        })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 3. Rule-Only vs Hybrid Comparison
# ──────────────────────────────────────────────────────────────────────────────

def compare_rule_vs_hybrid(patients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each patient, run both rule-only and hybrid pipelines.
    Returns a comparison table showing top-1 recommendation and its priority score.
    """
    from phase3_recommendation_engine.hybrid_recommender import (
        generate_recommendations,
        generate_rule_only_recommendations,
    )

    comparisons = []
    for p in patients:
        kwargs = dict(
            predicted_risks=p["predicted_risks"],
            top_positive_risk_factors=p["top_positive_risk_factors"],
            protective_factors=p["protective_factors"],
            patient_profile=p["patient_profile"],
        )
        rule_out   = generate_rule_only_recommendations(**kwargs)
        hybrid_out = generate_recommendations(**kwargs, use_llm=True)

        rule_recs   = rule_out["prioritized_recommendations"]
        hybrid_recs = hybrid_out["prioritized_recommendations"]

        comparisons.append({
            "patient_id": p["patient_id"],
            "label": p["label"],
            "rule_top1": {
                "recommendation": rule_recs[0]["recommendation"][:100] + "..." if rule_recs else "N/A",
                "priority_score": rule_recs[0]["priority_score"] if rule_recs else 0.0,
            },
            "hybrid_top1": {
                "recommendation": hybrid_recs[0]["recommendation"][:100] + "..." if hybrid_recs else "N/A",
                "priority_score": hybrid_recs[0]["priority_score"] if hybrid_recs else 0.0,
            },
        })
    return comparisons


# ──────────────────────────────────────────────────────────────────────────────
# 4. Ranking Consistency Validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_ranking_consistency(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validates that for every patient, recommendations are ordered by
    descending priority_score (monotonically non-increasing).

    Returns a summary dict with pass/fail per patient.
    """
    summary = {"total": len(results), "passed": 0, "failed": 0, "details": {}}

    for res in results:
        recs = res["output"]["prioritized_recommendations"]
        if not recs:
            summary["details"][res["patient_id"]] = "SKIP (no recommendations)"
            continue

        scores = [r["priority_score"] for r in recs]
        is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

        if is_sorted:
            summary["passed"] += 1
            summary["details"][res["patient_id"]] = f"PASS (n={len(recs)} recs)"
        else:
            summary["failed"] += 1
            summary["details"][res["patient_id"]] = f"FAIL - scores: {scores}"

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# 5. Console Report Printer
# ──────────────────────────────────────────────────────────────────────────────

def print_patient_report(result: Dict[str, Any]) -> None:
    """Print a formatted recommendation report for one patient."""
    print("\n" + "=" * 80)
    print(f"Patient: {result['patient_id']} — {result['label']}")
    print("=" * 80)

    ctx = result["output"]["patient_context"]
    print(f"  Age Group   : {ctx['age_group']}")
    print(f"  Risk Levels : {ctx['risk_levels']}")
    print()

    recs = result["output"]["prioritized_recommendations"]
    for i, rec in enumerate(recs, 1):
        print(f"  [{i}] Priority Score : {rec['priority_score']:.4f}")
        print(f"      Related Risk   : {rec['related_risk']}")
        # Wrap recommendation text at 70 chars for readability
        wrapped = "\n           ".join(
            [rec["recommendation"][j:j+70] for j in range(0, len(rec["recommendation"]), 70)]
        )
        print(f"      Recommendation : {wrapped}")
        print()
