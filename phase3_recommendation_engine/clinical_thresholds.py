"""
clinical_thresholds.py
─────────────────────
Evidence-based clinical thresholds and importance weights for the
Risk-Weighted Adaptive Wellness Recommendation Model (RW-AWRM).

References:
    - ADA Standards of Medical Care in Diabetes (2023)
    - JNC 8 / AHA 2017 Hypertension Guidelines
    - AHA/ACC Cardiovascular Risk Guidelines (2019)
    - WHO Global BMI Classification
    - NCEP ATP III Cholesterol Guidelines
"""

from typing import Dict, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# BMI thresholds (WHO classification, kg/m²)
# ──────────────────────────────────────────────────────────────────────────────
BMI_THRESHOLDS: Dict[str, float] = {
    "underweight":    18.5,
    "normal":         24.9,
    "overweight":     29.9,
    "obese_class_i":  34.9,
    "obese_class_ii": 39.9,
}

# ──────────────────────────────────────────────────────────────────────────────
# Fasting plasma glucose thresholds (mg/dL, ADA 2023)
# ──────────────────────────────────────────────────────────────────────────────
GLUCOSE_THRESHOLDS: Dict[str, float] = {
    "normal":       99.0,
    "pre_diabetes": 125.0,
    "diabetic":     126.0,
}

# ──────────────────────────────────────────────────────────────────────────────
# Systolic blood pressure thresholds (mmHg, AHA 2017)
# ──────────────────────────────────────────────────────────────────────────────
BLOOD_PRESSURE_THRESHOLDS: Dict[str, float] = {
    "normal":         120.0,
    "elevated":       129.0,
    "hypertension_1": 139.0,
    "hypertension_2": 180.0,
}

# ──────────────────────────────────────────────────────────────────────────────
# Total cholesterol thresholds (mg/dL, NCEP ATP III)
# ──────────────────────────────────────────────────────────────────────────────
CHOLESTEROL_THRESHOLDS: Dict[str, float] = {
    "desirable":       200.0,
    "borderline_high": 239.0,
    "high":            240.0,
}

# ──────────────────────────────────────────────────────────────────────────────
# Age group stratification (years)
# ──────────────────────────────────────────────────────────────────────────────
AGE_GROUPS: Dict[str, Tuple[int, int]] = {
    "young_adult": (18, 35),
    "middle_aged": (36, 55),
    "older_adult": (56, 70),
    "elderly":     (71, 120),
}

# ──────────────────────────────────────────────────────────────────────────────
# Risk severity probability bands
# ──────────────────────────────────────────────────────────────────────────────
RISK_BANDS: Dict[str, Tuple[float, float]] = {
    "low":    (0.00, 0.33),
    "medium": (0.34, 0.66),
    "high":   (0.67, 1.00),
}

# ──────────────────────────────────────────────────────────────────────────────
# Clinical importance weights per biomarker feature
# Derived from systematic reviews of feature-outcome associations.
# Used in: score = risk_prob × |shap_weight| × clinical_importance
# ──────────────────────────────────────────────────────────────────────────────
CLINICAL_IMPORTANCE_WEIGHTS: Dict[str, float] = {
    "BMI":                      0.85,
    "Glucose":                  0.92,
    "BloodPressure":            0.88,
    "Cholesterol":              0.83,
    "Age":                      0.72,
    "Insulin":                  0.76,
    "DiabetesPedigreeFunction": 0.79,
    "SkinThickness":            0.55,
    "Pregnancies":              0.50,
    "_default":                 0.60,
}

# ──────────────────────────────────────────────────────────────────────────────
# Canonical alias mapping (handles lowercase/underscore column variants)
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_ALIAS_MAP: Dict[str, str] = {
    "bmi":                        "BMI",
    "blood_pressure":             "BloodPressure",
    "bloodpressure":              "BloodPressure",
    "glucose":                    "Glucose",
    "cholesterol":                "Cholesterol",
    "age":                        "Age",
    "insulin":                    "Insulin",
    "skinthickness":              "SkinThickness",
    "skin_thickness":             "SkinThickness",
    "diabetespedigreefunction":   "DiabetesPedigreeFunction",
    "diabetes_pedigree_function": "DiabetesPedigreeFunction",
    "pregnancies":                "Pregnancies",
}


def get_clinical_importance(feature: str) -> float:
    """Return the clinical importance weight for a feature."""
    canonical = FEATURE_ALIAS_MAP.get(feature.lower(), feature)
    return CLINICAL_IMPORTANCE_WEIGHTS.get(canonical, CLINICAL_IMPORTANCE_WEIGHTS["_default"])


def classify_risk_band(probability: float) -> str:
    """Map a predicted probability to a named risk band."""
    for band, (lo, hi) in RISK_BANDS.items():
        if lo <= probability <= hi:
            return band
    return "high"


def classify_age_group(age: int) -> str:
    """Map numeric age to an age group label."""
    for group, (lo, hi) in AGE_GROUPS.items():
        if lo <= age <= hi:
            return group
    return "elderly"
