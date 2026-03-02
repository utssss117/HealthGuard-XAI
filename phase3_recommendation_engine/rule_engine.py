"""
rule_engine.py
──────────────
Structured clinical rule engine that maps patient biomarker profiles
to evidence-based lifestyle recommendations.

Design principles:
  - Each rule is a named, self-contained function.
  - Rules fire only when the biomarker exceeds a clinically meaningful threshold.
  - No medication instructions. Only lifestyle guidance.
  - Each emitted recommendation carries a raw clinical importance score used
    downstream by the risk-weighting layer.
"""

from typing import List, Dict, Any

from phase3_recommendation_engine.clinical_thresholds import (
    BMI_THRESHOLDS,
    GLUCOSE_THRESHOLDS,
    BLOOD_PRESSURE_THRESHOLDS,
    CHOLESTEROL_THRESHOLDS,
    AGE_GROUPS,
    get_clinical_importance,
    classify_age_group,
)


# ──────────────────────────────────────────────────────────────────────────────
# Internal recommendation template builder
# ──────────────────────────────────────────────────────────────────────────────

def _make_rec(feature: str, recommendation: str, justification: str,
              related_risk: str, base_score: float) -> Dict[str, Any]:
    return {
        "feature":        feature,
        "recommendation": recommendation,
        "justification":  justification,
        "related_risk":   related_risk,
        "base_score":     base_score,          # raw clinical importance
    }


# ──────────────────────────────────────────────────────────────────────────────
# Individual clinical rule functions
# ──────────────────────────────────────────────────────────────────────────────

def _rule_bmi(profile: Dict[str, float]) -> List[Dict[str, Any]]:
    bmi = profile.get("BMI", 0.0)
    recs = []
    if bmi > BMI_THRESHOLDS["obese_class_i"]:
        recs.append(_make_rec(
            "BMI",
            "Implement a structured caloric-deficit diet (500–750 kcal/day below TDEE) "
            "combined with 150–300 min/week of moderate aerobic activity (e.g., brisk walking, cycling).",
            f"BMI {bmi:.1f} kg/m² indicates Class I+ obesity, significantly elevating "
            "cardiometabolic risk. A 5–10% body weight reduction improves insulin sensitivity "
            "and reduces cardiovascular event probability.",
            "diabetes, heart_disease",
            get_clinical_importance("BMI"),
        ))
    elif bmi > BMI_THRESHOLDS["overweight"]:
        recs.append(_make_rec(
            "BMI",
            "Adopt a Mediterranean-style dietary pattern with portion control. "
            "Target 150 min/week of moderate physical activity to prevent progression to obesity.",
            f"BMI {bmi:.1f} kg/m² is in the overweight range. Lifestyle modification at "
            "this stage prevents transition to obesity and downstream chronic disease.",
            "diabetes, heart_disease",
            get_clinical_importance("BMI"),
        ))
    return recs


def _rule_glucose(profile: Dict[str, float]) -> List[Dict[str, Any]]:
    glucose = profile.get("Glucose", 0.0)
    recs = []
    if glucose >= GLUCOSE_THRESHOLDS["diabetic"]:
        recs.append(_make_rec(
            "Glucose",
            "Follow a low-glycemic index (GI < 55) dietary regimen. Eliminate refined sugars "
            "and white starches. Incorporate resistance training 3×/week to enhance peripheral "
            "glucose uptake. Monitor fasting glucose regularly.",
            f"Fasting glucose {glucose:.0f} mg/dL meets ADA diagnostic criteria for diabetes "
            "(≥126 mg/dL). Dietary and exercise interventions reduce HbA1c by 0.5–2.0%.",
            "diabetes",
            get_clinical_importance("Glucose"),
        ))
    elif glucose >= GLUCOSE_THRESHOLDS["normal"]:
        recs.append(_make_rec(
            "Glucose",
            "Reduce intake of high-GI foods (white rice, sugary beverages, processed snacks). "
            "Consume complex carbohydrates and increase dietary fiber (≥25 g/day). "
            "Maintain 30 min of daily moderate exercise.",
            f"Fasting glucose {glucose:.0f} mg/dL is in the pre-diabetic range (100–125 mg/dL). "
            "The Diabetes Prevention Program demonstrates 58% risk reduction with lifestyle changes.",
            "diabetes",
            get_clinical_importance("Glucose"),
        ))
    return recs


def _rule_blood_pressure(profile: Dict[str, float]) -> List[Dict[str, Any]]:
    bp = profile.get("BloodPressure", 0.0)
    recs = []
    if bp > BLOOD_PRESSURE_THRESHOLDS["hypertension_1"]:
        recs.append(_make_rec(
            "BloodPressure",
            "Adopt DASH diet: reduce dietary sodium to <2,300 mg/day, increase potassium-rich "
            "foods (bananas, leafy greens). Add structured aerobic exercise (30 min/day, 5 days/week). "
            "Reduce alcohol consumption and manage chronic stress through mindfulness-based techniques.",
            f"Blood pressure {bp:.0f} mmHg meets Stage 1+ hypertension threshold (>130 mmHg). "
            "DASH diet can reduce systolic BP by 8–14 mmHg, significantly lowering ASCVD risk.",
            "heart_disease",
            get_clinical_importance("BloodPressure"),
        ))
    elif bp > BLOOD_PRESSURE_THRESHOLDS["normal"]:
        recs.append(_make_rec(
            "BloodPressure",
            "Limit sodium intake to <2,300 mg/day. Engage in 150 min/week of moderate-intensity "
            "aerobic exercise. Practice stress reduction (deep breathing, yoga).",
            f"Blood pressure {bp:.0f} mmHg is elevated (121–129 mmHg). Early lifestyle modification "
            "prevents progression to hypertension and reduces future cardiovascular events.",
            "heart_disease",
            get_clinical_importance("BloodPressure"),
        ))
    return recs


def _rule_cholesterol(profile: Dict[str, float]) -> List[Dict[str, Any]]:
    chol = profile.get("Cholesterol", 0.0)
    recs = []
    if chol >= CHOLESTEROL_THRESHOLDS["high"]:
        recs.append(_make_rec(
            "Cholesterol",
            "Transition to a plant-predominant diet: increase soluble fiber (oats, legumes, flaxseed), "
            "replace saturated fats with unsaturated fats (olive oil, avocado, nuts). "
            "Add ≥30 min/day of aerobic exercise. Avoid trans fats entirely.",
            f"Total cholesterol {chol:.0f} mg/dL exceeds the high threshold (≥240 mg/dL). "
            "Dietary changes and aerobic exercise can reduce LDL-C by 10–20% without medication.",
            "heart_disease",
            get_clinical_importance("Cholesterol"),
        ))
    elif chol >= CHOLESTEROL_THRESHOLDS["desirable"]:
        recs.append(_make_rec(
            "Cholesterol",
            "Reduce saturated fat intake (<7% of total calories). Increase omega-3 fatty acids "
            "(fatty fish 2×/week). Incorporate 20–30 min of daily physical activity.",
            f"Total cholesterol {chol:.0f} mg/dL is borderline-high (200–239 mg/dL). "
            "Preventive dietary changes at this stage can halt progression to high-risk range.",
            "heart_disease",
            get_clinical_importance("Cholesterol"),
        ))
    return recs


def _rule_age(profile: Dict[str, float]) -> List[Dict[str, Any]]:
    age = int(profile.get("Age", 0))
    recs = []
    age_group = classify_age_group(age)
    if age_group in ("older_adult", "elderly"):
        recs.append(_make_rec(
            "Age",
            "Prioritize fall-prevention exercises (balance and resistance training). "
            "Schedule annual cardiovascular and metabolic health screenings. "
            "Ensure adequate calcium (1,200 mg/day) and vitamin D (800–1,000 IU/day) intake.",
            f"Age {age} years places this individual in the {age_group.replace('_', ' ')} group, "
            "associated with higher baseline cardiovascular and metabolic risk. Proactive "
            "preventive screenings are evidence-supported for this cohort.",
            "heart_disease, diabetes",
            get_clinical_importance("Age"),
        ))
    return recs


def _rule_insulin(profile: Dict[str, float]) -> List[Dict[str, Any]]:
    insulin = profile.get("Insulin", 0.0)
    recs = []
    # Elevated fasting insulin (>25 µU/mL) as a proxy for insulin resistance
    if insulin > 25.0:
        recs.append(_make_rec(
            "Insulin",
            "Reduce dietary refined carbohydrates and added sugars. "
            "Incorporate high-intensity interval training (HIIT) 2–3×/week to improve insulin sensitivity. "
            "Adopt time-restricted eating (10–12 hour eating window) if clinically appropriate.",
            f"Fasting insulin level {insulin:.0f} µU/mL suggests potential insulin resistance. "
            "Lifestyle intervention addressing diet quality and exercise intensity can improve "
            "insulin sensitivity indices (HOMA-IR) significantly.",
            "diabetes",
            get_clinical_importance("Insulin"),
        ))
    return recs


def _rule_multimorbid(profile: Dict[str, float], fired_risks: List[str]) -> List[Dict[str, Any]]:
    """Fires when two or more co-morbid risk signals are simultaneously active."""
    recs = []
    unique_risks = set(r.strip() for r in ",".join(fired_risks).split(","))
    if "diabetes" in unique_risks and "heart_disease" in unique_risks:
        recs.append(_make_rec(
            "Comorbidity",
            "Your risk profile shows simultaneous elevation across both metabolic and cardiovascular "
            "pathways. Adopt an integrated cardiometabolic management plan: combine a Mediterranean-DASH "
            "hybrid diet, 200+ min/week progressive physical activity, and regular bi-annual "
            "cardiometabolic panels (HbA1c, lipid panel, BP monitoring).",
            "Co-elevated diabetes and cardiovascular risk amplify each other exponentially. "
            "Integrated lifestyle interventions addressing both simultaneously show compounded "
            "risk reduction vs. single-condition management.",
            "diabetes, heart_disease",
            max(get_clinical_importance("Glucose"), get_clinical_importance("BloodPressure")),
        ))
    return recs


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

RULES = [_rule_bmi, _rule_glucose, _rule_blood_pressure, _rule_cholesterol, _rule_age, _rule_insulin]


def apply_clinical_rules(patient_profile: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Apply all clinical rules to a patient biomarker profile.
    Returns a list of raw recommendation dicts (pre-scoring).
    """
    all_recs: List[Dict[str, Any]] = []
    fired_risks: List[str] = []

    for rule_fn in RULES:
        recs = rule_fn(patient_profile)
        all_recs.extend(recs)
        for r in recs:
            fired_risks.append(r["related_risk"])

    # Multi-morbidity rule (conditional on other rules firing)
    all_recs.extend(_rule_multimorbid(patient_profile, fired_risks))

    return all_recs
